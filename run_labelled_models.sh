#!/bin/bash
# Run the five models on every labelled video, then write their accuracy and tracking-the-object charts.
# Videos a model has already answered are skipped, so this is safe to run again after adding or fixing labels.
# video_14052_reversed's boxes are on the wrong object, so it is left out of the charts until it is fixed.
cd "$(dirname "$0")"
FILES=$(cd labels && ls labels_peter*.json)
ORIGINAL=$(for f in $FILES; do echo "labels/$f"; done)
converted() { for f in $FILES; do echo "outputs/$1/labels/$f"; done; }

.venv-vidlada/bin/python run_attention_llava_video.py $ORIGINAL
.venv-vidlada/bin/python run_attention_vidlada.py $ORIGINAL
.venv-dynin/bin/python run_attention_dynin.py $ORIGINAL
.venv-dreamvl/bin/python run_attention_dreamvl.py $ORIGINAL

.venv/bin/python labelled_accuracy.py $ORIGINAL

SKIP="--skip video_14052_reversed.mp4 --out-name time_layer_250"
.venv/bin/python plot_time_layer.py $ORIGINAL --run-dir outputs/qwen3vl_8b $SKIP
.venv/bin/python plot_time_layer.py $(converted llava_video_7b) --run-dir outputs/llava_video_7b $SKIP
.venv/bin/python plot_time_layer.py $(converted vidlada_8b) --run-dir outputs/vidlada_8b $SKIP
.venv/bin/python plot_time_layer.py $(converted dynin_omni) --run-dir outputs/dynin_omni $SKIP
.venv/bin/python plot_time_layer.py $ORIGINAL --run-dir outputs/dreamvl_7b $SKIP

tar czf outputs/labelled_250_results.tar.gz -C outputs model_comparison/accuracy_labelled.txt \
    model_comparison/accuracy_labelled.csv model_comparison/accuracy_labelled.png \
    {qwen3vl_8b,llava_video_7b,vidlada_8b,dynin_omni,dreamvl_7b}/analysis/time_layer_250/tracking_the_object.png
echo "Bundle: outputs/labelled_250_results.tar.gz"
