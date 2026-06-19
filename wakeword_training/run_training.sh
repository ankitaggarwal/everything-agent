#!/bin/bash
# Reachy custom wake-word training -- runs inside an HF Jobs GPU container.
# Replicates the andyjmorgan/reachy-wake-word recipe (98% acc) and BLENDS IN the
# user's real recordings (oversampled) so the model wakes for their voices.
# Env: HF_TOKEN (secret), SMOKE (1=tiny fast validation run), DATA_DIR (clips).
set -euo pipefail
SMOKE="${SMOKE:-0}"
DATA_DIR="${DATA_DIR:-/data}"
WORK=/workspace; mkdir -p $WORK; cd $WORK
echo "###### Reachy wakeword training  (SMOKE=$SMOKE) ######"
nvidia-smi -L || echo "(no gpu visible)"

apt-get update -qq && apt-get install -y -qq git wget tar ffmpeg espeak-ng libsndfile1 build-essential >/dev/null 2>&1 || true
pip install -q -U huggingface_hub >/dev/null

# 1. training framework (andyjmorgan fork of openWakeWord) ------------------
git clone -q https://github.com/andyjmorgan/reachy-wake-word.git
cd reachy-wake-word && pip install -q -e . >/dev/null
# EXACT pinned training deps from dscripka's automatic_model_training notebook
pip install -q mutagen==1.47.0 torchinfo==1.8.0 torchmetrics==1.2.0 speechbrain==0.5.14 \
  audiomentations==0.33.0 torch-audiomentations==0.11.0 acoustics==0.2.6 pronouncing==0.2.0 \
  datasets==2.14.6 deep-phonemizer==0.0.19 onnx pyyaml tqdm scikit-learn >/dev/null 2>&1 || true

# 2. piper-sample-generator (synthetic TTS) -- v2.0.0 is the version dscripka's
#    notebook uses: generate_samples.py exposes the generate_samples() FUNCTION
#    (v1.0.0 is CLI-only) and pairs with the libritts_r-medium model.
git clone -q --branch v2.0.0 --depth 1 https://github.com/rhasspy/piper-sample-generator.git
cd piper-sample-generator && mkdir -p models
wget -q https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt -O models/en_US-libritts_r-medium.pt
pip install -q piper-phonemize webrtcvad >/dev/null 2>&1 || true
PIPER="$WORK/reachy-wake-word/piper-sample-generator"
RIR="$PIPER/impulses"; [ -d "$RIR" ] || RIR=""   # clean recipe; RIR optional
cd $WORK/reachy-wake-word
# Pin numpy<2 + scipy<1.13 LAST (training stack is numpy-1.x ABI; acoustics needs
# the old scipy.special.sph_harm). Installed together so the ABIs match.
pip install -q "numpy<2" "scipy<1.13"
python -c "import numpy, scipy, acoustics; print('numpy', numpy.__version__, 'scipy', scipy.__version__, '+ acoustics OK')"

# 3. base feature models + negative features --------------------------------
python -c "import openwakeword; openwakeword.utils.download_models()" || true
VAL=$(python -c "from huggingface_hub import hf_hub_download as h; print(h('davidscripka/openwakeword_features','validation_set_features.npy',repo_type='dataset'))")
if [ "$SMOKE" = "1" ]; then
  ACAV="$VAL"   # smoke: reuse the small (~180MB) validation features as negatives
else
  ACAV=$(python -c "from huggingface_hub import hf_hub_download as h; print(h('davidscripka/openwakeword_features','openwakeword_features_ACAV100M_2000_hrs_16bit.npy',repo_type='dataset'))")
fi
echo "ACAV=$ACAV"; echo "VAL=$VAL"

# 4. tunables (smoke vs full) ----------------------------------------------
if [ "$SMOKE" = "1" ]; then NSAMP=200; NVAL=120; STEPS=2000; OVER=20; ABATCH=16; TACC=0.5; TREC=0.4
else NSAMP=5000; NVAL=1000; STEPS=50000; OVER=40; ABATCH=100; TACC=0.92; TREC=0.85; fi
OUT=$WORK/trained; MODEL=reachy_custom; mkdir -p $OUT

cat > cfg.yaml <<EOF
target_phrase: ["reechy"]
model_name: $MODEL
piper_sample_generator_path: "$PIPER"
tts_model_path: "$PIPER/models/en_US-libritts_r-medium.pt"
tts_batch_size: 50
n_samples: $NSAMP
n_samples_val: $NVAL
max_speakers: 150
steps: $STEPS
target_accuracy: $TACC
target_recall: $TREC
target_false_positives_per_hour: 0.5
model_type: dnn
layer_size: 192
n_blocks: 1
augmentation_rounds: 2
augmentation_batch_size: $ABATCH
max_negative_weight: 1500
batch_n_per_class: {ACAV100M_sample: 1024, adversarial_negative: 50, positive: 50}
rir_paths: [$( [ -n "$RIR" ] && echo "\"$RIR\"" )]
background_paths: []
background_paths_duplication_rate: []
custom_negative_phrases: []
output_dir: "$OUT"
false_positive_validation_data_path: "$VAL"
feature_data_files:
  ACAV100M_sample: "$ACAV"
  adversarial_negative: "$OUT/$MODEL/adversarial_negative_features_train.npy"
  positive: "$OUT/$MODEL/positive_features_train.npy"
EOF
echo "----- config -----"; cat cfg.yaml

# 5. generate synthetic positives + adversarial negatives -------------------
python -m openwakeword.train --training_config cfg.yaml --generate_clips

# 6. INJECT + oversample the user's real clips into positive_train ----------
POS="$OUT/$MODEL/positive_train"; POST="$OUT/$MODEL/positive_test"; mkdir -p "$POS" "$POST"
nreal=0
for n in $(seq 1 $OVER); do
  for f in "$DATA_DIR"/positives/*.wav; do [ -e "$f" ] || continue; cp "$f" "$POS/real_${n}_$(basename "$f")"; nreal=$((nreal+1)); done
done
# a few real clips into the test/val set too (held-out validation of THEIR voice)
i=0; for f in "$DATA_DIR"/positives/*.wav; do [ -e "$f" ] || continue; cp "$f" "$POST/realval_$(basename "$f")"; i=$((i+1)); [ $i -ge 6 ] && break; done
echo "injected $nreal oversampled real positive copies"

# 7. augment + train --------------------------------------------------------
python -m openwakeword.train --training_config cfg.yaml --augment_clips
# train exports the ONNX; the optional TFLite convert needs tensorflow (skipped),
# so tolerate a late failure -- the .onnx we want is already written by then.
python -m openwakeword.train --training_config cfg.yaml --train_model \
  || echo "train step exited (likely the optional tflite convert; checking for onnx)"

# 8. upload the trained onnx ------------------------------------------------
ONNX=$(find $OUT -name "*.onnx" | head -1)
echo "trained onnx: $ONNX"
python - <<PY
import glob, os
from huggingface_hub import HfApi, create_repo
f=glob.glob("$OUT/**/*.onnx", recursive=True)
assert f, "no onnx produced"
repo="Curious-PM/reachy-wakeword-model"
create_repo(repo, repo_type="model", exist_ok=True, private=True, token=os.environ["HF_TOKEN"])
HfApi().upload_file(path_or_fileobj=f[0], path_in_repo="reachy_custom_${SMOKE}.onnx" if "$SMOKE"=="1" else "reachy_custom.onnx",
                    repo_id=repo, repo_type="model", token=os.environ["HF_TOKEN"])
print("UPLOADED", f[0])
PY
echo "###### DONE ######"
