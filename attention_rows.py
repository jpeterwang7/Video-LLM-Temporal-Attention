"""
Record (SdpaRowRecorder) or steer (SdpaSteerer) one query position's attention in every layer, by wrapping
torch's scaled_dot_product_attention.

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


class SdpaSteerer:
    """Add a bias to the attention logits of chosen keys in one query row, in every attention call over the
    full sequence (one per layer of the language model), and recompute that row's output; every other row is
    left as the model computed it. A bias b multiplies those keys' attention weights by e^b before the row is
    renormalised. Each call also records the row's attention share on named sets of keys (mean over heads)."""

    def __init__(self):
        self.calls = 0
        self.shares = []

    @contextmanager
    def steer(self, seq_len, row, targets, bias, measure):
        """targets: key positions to boost. measure: {name: key positions} whose attention share to record."""
        self.calls, self.shares = 0, []
        original = F.scaled_dot_product_attention

        def wrapped(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
            out = original(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)
            if q.dim() != 4 or q.shape[-2] != seq_len or k.shape[-2] != seq_len:
                return out                                           # not the language model over the sequence
            self.calls += 1
            query = q[:, :, row:row + 1].float()                     # [batch, heads, 1, head_dim]
            keys, values = k.float(), v.float()
            if keys.shape[1] != query.shape[1]:
                keys = keys.repeat_interleave(query.shape[1] // keys.shape[1], dim=1)
                values = values.repeat_interleave(query.shape[1] // values.shape[1], dim=1)
            scores = query @ keys.transpose(-1, -2) * (kwargs.get("scale") or 1 / math.sqrt(q.shape[-1]))
            if attn_mask is not None:
                mask = attn_mask[:, None, None, :] if attn_mask.dim() == 2 else attn_mask[..., row:row + 1, :seq_len]
                scores = scores.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else scores + mask.float()
            if is_causal:
                scores[..., row + 1:] = float("-inf")
            if len(targets):
                scores[..., torch.as_tensor(targets, device=scores.device)] += bias
            probs = torch.softmax(scores, dim=-1)
            self.shares.append({name: float(probs[0, :, 0, torch.as_tensor(keys_, device=probs.device)].sum(-1).mean())
                                for name, keys_ in measure.items()})
            out = out.clone()
            out[:, :, row:row + 1] = (probs @ values).to(out.dtype)
            return out

        F.scaled_dot_product_attention = wrapped
        try:
            yield self
        finally:
            F.scaled_dot_product_attention = original
