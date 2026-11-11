#!/bin/bash
# TabAlign step 1 with LLaDA-8B-Instruct as the planner: Dynin-Omni describes each frame, then LLaDA-8B-Instruct plans
# from the question alone and from the question plus the descriptions, each with and without the worked example.
# Finished videos are skipped, so it can be run again after a stop. Ends by bundling the results for download.
cd "$(dirname "$0")"
PY=.venv-dynin/bin/python
LABELS=$(ls labels/labels_peter*.json)

$PY caption_frames_dynin.py $LABELS || exit 1
$PY make_plans_llada.py $LABELS --out-dir outputs/plans_llada || exit 1
$PY make_plans_llada.py $LABELS --no-example --out-dir outputs/plans_llada_no_example || exit 1
$PY make_plans_llada.py $LABELS --captions outputs/frame_captions.jsonl --out-dir outputs/plans_llada_captions || exit 1
$PY make_plans_llada.py $LABELS --captions outputs/frame_captions.jsonl --no-example \
    --out-dir outputs/plans_llada_captions_no_example || exit 1

tar czf outputs/plans_llada.tar.gz -C outputs frame_captions.jsonl plans_llada plans_llada_no_example \
    plans_llada_captions plans_llada_captions_no_example
echo "Bundle: outputs/plans_llada.tar.gz"
