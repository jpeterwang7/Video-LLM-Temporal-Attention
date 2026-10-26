#!/bin/bash
# Run steer_attention.py for all five models, each in its own environment (see README). Each model resumes
# where it stopped, and one model failing does not stop the others.
cd "$(dirname "$0")"

run() {
    local env=$1; shift
    (source "$env/bin/activate" && python steer_attention.py "$@") || echo "FAILED: $1"
}

run .venv qwen3vl labels/labels_peter.json labels/labels_peter_labeling_round2.json
run .venv-vidlada llava_video outputs/llava_video_7b/labels/labels_peter.json outputs/llava_video_7b/labels/labels_peter_labeling_round2.json
run .venv-vidlada vidlada outputs/vidlada_8b/labels/labels_peter.json outputs/vidlada_8b/labels/labels_peter_labeling_round2.json
run .venv-dynin dynin outputs/dynin_omni/labels/labels_peter.json outputs/dynin_omni/labels/labels_peter_labeling_round2.json
run .venv-dreamvl dreamvl labels/labels_peter.json labels/labels_peter_labeling_round2.json
