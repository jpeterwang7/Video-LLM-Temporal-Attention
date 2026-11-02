#!/bin/bash
# Fine-tune Dynin-Omni with Qwen3-VL's attention, with controls, then compare them on held-out videos.
#   ./run_distill.sh smoke    a quick check on 20 videos, written to outputs/distillation/smoke
#   ./run_distill.sh full     the three training runs and the comparison; finished runs are not repeated
cd "$(dirname "$0")"
PY=.venv-dynin/bin/python
DATA=outputs/distillation
FILES=$(cd labels && ls labels_peter*.json)
LABELS=$(for f in $FILES; do echo "labels/$f"; done)

[ -f $DATA/split.json ] || $PY prepare_distillation.py $LABELS || exit 1

if [ "$1" = "smoke" ]; then
    $PY train_dynin_distill.py --out-dir $DATA/smoke --attn-weight 1 --limit 20 --epochs 1 --patience 2
    exit
fi

train() {   # name, then options
    [ -f $DATA/$1/finished ] || { $PY train_dynin_distill.py --out-dir $DATA/$1 "${@:2}" && touch $DATA/$1/finished; } || exit 1
    $PY run_attention_dynin.py $LABELS --split $DATA/split.json --checkpoint $DATA/$1/best.pt --out-dir $DATA/$1 || exit 1
}
train answers_only --attn-weight 0
train qwen_attention --attn-weight 1
train shuffled_attention --attn-weight 1 --teacher shuffled

.venv/bin/python distill_results.py --run "Unchanged" outputs/dynin_omni --run "Answers only" $DATA/answers_only \
    --run "Qwen3-VL attention" $DATA/qwen_attention --run "Shuffled attention" $DATA/shuffled_attention || exit 1

SKIP="--skip video_14052_reversed.mp4 --out-name tracking"
for run in answers_only qwen_attention shuffled_attention; do
    .venv/bin/python plot_time_layer.py $(for f in $FILES; do echo "$DATA/$run/labels/$f"; done) --run-dir $DATA/$run \
        --model-name "Dynin-Omni-8B, $run" $SKIP
done

tar czf outputs/distillation_results.tar.gz $DATA/results.txt $DATA/results.csv \
    $DATA/*/log.jsonl $DATA/*/analysis/tracking/tracking_the_object.png
echo "Bundle: outputs/distillation_results.tar.gz"
