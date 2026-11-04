#!/bin/bash
# Distil Qwen3-VL's answers into Dynin-Omni, with and without an anchor on Dynin-Omni's own attention, then compare
# the runs with the unchanged model on held-out videos.
#   ./run_distill.sh smoke    a quick check: 30 Qwen3-VL answers, then 20 training videos in outputs/distillation/smoke
#   ./run_distill.sh full     Qwen3-VL's answers for all training videos, the three training runs and the comparison;
#                             anything already finished is not repeated
cd "$(dirname "$0")"
PY=.venv-dynin/bin/python
DATA=outputs/distillation
FILES=$(cd labels && ls labels_peter*.json)
LABELS=$(for f in $FILES; do echo "labels/$f"; done)

[ -f $DATA/split.json ] || $PY prepare_distillation.py $LABELS || exit 1

if [ "$1" = "smoke" ]; then
    .venv/bin/python qwen_answer_probs.py --split $DATA/split.json --limit 30 --out $DATA/smoke_qwen_answers.jsonl || exit 1
    $PY train_dynin_distill.py --out-dir $DATA/smoke --qwen-answers $DATA/smoke_qwen_answers.jsonl --qwen-weight 0.5 \
        --attn-weight 1 --limit 20 --epochs 1 --patience 2
    exit
fi

.venv/bin/python qwen_answer_probs.py --split $DATA/split.json || exit 1

train() {   # name, then options
    [ -f $DATA/$1/finished ] || { $PY train_dynin_distill.py --out-dir $DATA/$1 "${@:2}" && touch $DATA/$1/finished; } || exit 1
    $PY run_attention_dynin.py $LABELS --split $DATA/split.json --checkpoint $DATA/$1/best.pt --out-dir $DATA/$1 || exit 1
}
train answers_only
train qwen_answers --qwen-weight 0.5
train qwen_answers_anchored --qwen-weight 0.5 --attn-weight 1

.venv/bin/python distill_results.py --run "Unchanged" outputs/dynin_omni --run "Answers only" $DATA/answers_only \
    --run "Qwen3-VL answers" $DATA/qwen_answers --run "Qwen3-VL answers, anchored" $DATA/qwen_answers_anchored || exit 1

SKIP="--skip video_14052_reversed.mp4 --out-name tracking"
for run in answers_only qwen_answers qwen_answers_anchored; do
    .venv/bin/python plot_time_layer.py $(for f in $FILES; do echo "$DATA/$run/labels/$f"; done) --run-dir $DATA/$run \
        --model-name "Dynin-Omni-8B, $run" $SKIP
done

tar czf outputs/distillation_results.tar.gz $DATA/results.txt $DATA/results.csv \
    $DATA/{answers_only,qwen_answers,qwen_answers_anchored}/log.jsonl \
    $DATA/{answers_only,qwen_answers,qwen_answers_anchored}/analysis/tracking/tracking_the_object.png
echo "Bundle: outputs/distillation_results.tar.gz"
