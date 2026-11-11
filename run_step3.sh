#!/bin/bash
# Step 3: build Dynin-Omni's training set. LLaDA-8B-Instruct plans and Qwen3-VL carries out every training and
# validation video; traces that end in the right answer are the examples Dynin-Omni will learn from.
#   ./run_step3.sh check   rerun the 20 sample videos with the executor's fixes (outputs/qwen_execution_v2)
#   ./run_step3.sh build   200 videos at a time, in a fixed random order: plan them (outputs/plans_train), then carry
#                          them out (outputs/traces_train). Stop any time; running it again resumes.
cd "$(dirname "$0")"
LABELS=$(ls labels/labels_peter*.json)

if [ "$1" = "check" ]; then
    .venv/bin/python execute_plans_qwen.py --out-dir outputs/qwen_execution_v2
    exit
fi

[ -f outputs/split.json ] || .venv/bin/python make_split.py $LABELS || exit 1
TOTAL=$(.venv/bin/python -c "import json; s = json.load(open('outputs/split.json')); print(len(s['train']) + len(s['val']))")
for ((n = 200; n < TOTAL + 200; n += 200)); do
    echo "=== Videos 1-$((n < TOTAL ? n : TOTAL)) of $TOTAL ==="
    .venv-dynin/bin/python make_plans_llada.py --split outputs/split.json --limit $n --out-dir outputs/plans_train || exit 1
    .venv/bin/python execute_plans_qwen.py --plans outputs/plans_train/plans.jsonl --out-dir outputs/traces_train || exit 1
done
echo "Training set: outputs/traces_train/traces.jsonl (traces with \"correct\": true are kept for training)"
