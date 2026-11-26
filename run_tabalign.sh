#!/bin/bash
# TabAlign for video, end to end (see README, "TabAlign for video"):
#   1. LLaDA-8B-Instruct plans every question (and Qwen3-VL-8B too, for the planner comparison)
#   2. Dynin-Omni grounds every plan target, on one frame per Qwen3-VL time step
#   3. Qwen3-VL carries out the validation plans without feedback; TabAttn is fit on them
#   4. Qwen3-VL carries out the test plans in every configuration, most important first
#   5. summary.txt, and results.tar.gz to copy back
#
#   ./run_tabalign.sh smoke       the whole pipeline on 6 validation and 6 test videos (about 20 minutes), to check it runs
#   ./run_tabalign.sh preview N   early numbers: LLaDA plans and Qwen3-VL carries out the first N test videos (default
#                                 30, one pair per category in turn), without the verifier, which needs the
#                                 calibration; about 20 minutes for 30
#   ./run_tabalign.sh full        everything (many hours); stop it any time, running it again resumes where it stopped
#   ./run_tabalign.sh summary     only step 5
#
# After the full run, steering Qwen3-VL's attention onto Dynin-Omni's regions while it answers directly
# (steer_qwen.py; uses the full run's Dynin-Omni maps):
#   ./run_tabalign.sh steer-smoke the 6 debugging videos, every region at strength 10 (a few minutes), to check it runs
#   ./run_tabalign.sh steer       strengths 3, 10, 30 on the 260 validation videos, then the best one on the 620 test
#                                 videos: unsteered, Dynin-Omni's region, a random region of the same size, the human
#                                 boxes (about 2 hours); stop it any time, running it again resumes
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     # less fragmentation on the 24 GB card
MAIN=.venv/bin/python
DYNIN=.venv-dynin/bin/python
MODE=${1:-smoke}
case "$MODE" in
    smoke) CAL=smoke_calib; TEST=smoke_test; OUT=outputs/tabalign_smoke ;;
    full|summary|preview|steer|steer-smoke) CAL=calib; TEST=test; OUT=outputs/tabalign ;;
    *) echo "Usage: $0 smoke|preview [N]|full|summary|steer-smoke|steer"; exit 1 ;;
esac
mkdir -p "$OUT"
LOG="$OUT/log.txt"

run() {
    echo "=== $(date '+%F %T') $*" | tee -a "$LOG"
    "$@" 2>&1 | tee -a "$LOG"
    status=${PIPESTATUS[0]}
    if [ "$status" -ne 0 ]; then
        echo "FAILED (exit $status): $*" | tee -a "$LOG"
        exit "$status"
    fi
}

if [ "$MODE" = preview ]; then
    N=${2:-30}
    [ -f outputs/split.json ] || run $MAIN make_split.py labels/labels_peter*.json
    run $MAIN tabalign_video.py --out-dir "$OUT"
    run $DYNIN plan_tabalign.py --planner llada --set $TEST --limit $N --exclude-set smoke_test --out-dir "$OUT"
    run $MAIN execute_tabalign.py --set $TEST --plan llada --reward none --limit $N --exclude-set smoke_test \
        --run-name preview --out-dir "$OUT"
    echo "PREVIEW DONE: accuracy above; traces in $OUT/runs/$TEST/preview/traces.jsonl"
    exit 0
fi

if [ "$MODE" = steer-smoke ]; then
    run $MAIN steer_qwen.py --set smoke_test --configs none dynin:10 random:10 human:10 --out-dir "$OUT"
    run $MAIN steer_qwen.py --set smoke_test --report --out-dir "$OUT"
    echo "STEER SMOKE DONE: table above, in $OUT/steer/smoke_test/summary.txt"
    exit 0
fi

if [ "$MODE" = steer ]; then
    run $MAIN steer_qwen.py --set $CAL --configs none dynin:3 dynin:10 dynin:30 --out-dir "$OUT"
    run $MAIN steer_qwen.py --set $CAL --report --out-dir "$OUT"
    ALPHA=$($MAIN steer_qwen.py --set $CAL --best dynin --out-dir "$OUT") || { echo "FAILED: choosing the strength" | tee -a "$LOG"; exit 1; }
    echo "=== $(date '+%F %T') strength chosen on $CAL: $ALPHA" | tee -a "$LOG"
    run $MAIN steer_qwen.py --set $TEST --exclude-set smoke_test --configs none dynin:$ALPHA random:$ALPHA human:$ALPHA \
        --out-dir "$OUT"
    run $MAIN steer_qwen.py --set $TEST --exclude-set smoke_test --report --out-dir "$OUT"
    tar czf "$OUT/steer_results.tar.gz" -C "$OUT" steer
    echo "Results to copy back: $OUT/steer_results.tar.gz"
    exit 0
fi

if [ "$MODE" != summary ]; then
    [ -f outputs/split.json ] || run $MAIN make_split.py labels/labels_peter*.json
    run $MAIN tabalign_video.py --out-dir "$OUT"
    run $DYNIN plan_tabalign.py --planner llada --set $CAL $TEST --out-dir "$OUT"
    run $MAIN plan_tabalign.py --planner qwen --set $TEST --out-dir "$OUT"
    run $DYNIN ground_dynin.py --set $CAL $TEST --out-dir "$OUT"
    run $MAIN execute_tabalign.py --set $CAL --plan llada --reward none --record-qwen --out-dir "$OUT"
    run $MAIN fit_tabattn.py --run "$OUT/runs/$CAL/llada_none" --out-dir "$OUT"
    run $MAIN execute_tabalign.py --set $TEST --plan llada --reward dynin --out-dir "$OUT"
    run $MAIN execute_tabalign.py --set $TEST --plan llada --reward none --record-qwen --out-dir "$OUT"
    run $MAIN execute_tabalign.py --set $TEST --plan qwen --reward dynin --out-dir "$OUT"
    run $MAIN execute_tabalign.py --set $TEST --plan llada --reward qwen --out-dir "$OUT"
    run $MAIN execute_tabalign.py --set $TEST --plan llada --reward oracle --only-labelled --out-dir "$OUT"
    run $MAIN execute_tabalign.py --set $TEST --plan llada --reward oracle_noise --only-labelled --out-dir "$OUT"
fi
run $MAIN summarize_tabalign.py --set $TEST --out-dir "$OUT"
tar czf "$OUT/results.tar.gz" -C "$OUT" summary.txt per_video.csv log.txt $(cd "$OUT" && ls calibration.json plans_*.jsonl \
    dynin/results_*.jsonl runs/*/*/traces.jsonl 2>/dev/null)
echo "Results to copy back: $OUT/results.tar.gz"
