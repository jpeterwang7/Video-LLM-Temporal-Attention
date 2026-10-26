"""
Record one query position's attention row in every layer by wrapping torch's scaled_dot_product_attention.

Models that compute attention with F.scaled_dot_product_attention (Qwen2 under attn_implementation="sdpa",
LLaDA without flash-attn) hand it the queries and keys after rotary embedding, so softmax(q k^T / sqrt(d) +
mask) for one query row, averaged over heads, is exactly that layer's attention from that position. While
recording, every call appends one row, so a forward pass of the language model gives one row per layer.
"""
import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F


class SdpaRowRecorder:
    def __init__(self):
        self.rows = []

    @contextmanager
    def record(self, query):
        """Record the attention row of position `query` (mean over heads) for every attention call inside."""
        self.rows = []
        original = F.scaled_dot_product_attention

        def wrapped(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
            self.rows.append(self._row(q, k, attn_mask, is_causal, kwargs.get("scale"), query))
            return original(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)

        F.scaled_dot_product_attention = wrapped
        try:
            yield self
        finally:
            F.scaled_dot_product_attention = original

    @staticmethod
    def _row(q, k, mask, is_causal, scale, n):
        query = q[0, :, n:n + 1].float()                        # [heads, 1, head_dim]
        keys = k[0].float()                                     # [kv_heads, keys, head_dim]
        if keys.shape[0] != query.shape[0]:
            keys = keys.repeat_interleave(query.shape[0] // keys.shape[0], dim=0)
        scores = query @ keys.transpose(1, 2) * (scale or 1 / math.sqrt(q.shape[-1]))
        if mask is not None:
            row = mask[0, :, n:n + 1] if mask.dim() == 4 else mask[..., n:n + 1, :]
            scores = scores.masked_fill(~row, float("-inf")) if row.dtype == torch.bool else scores + row.float()
        if is_causal:
            scores[..., n + 1:] = float("-inf")
        return torch.softmax(scores, dim=-1)[:, 0].mean(dim=0).cpu()
