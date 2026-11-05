set -euo pipefail

python - <<'PY'
from huggingface_hub import login
import os
# Prefer: export HF_TOKEN=... and use login(token=os.environ["HF_TOKEN"])
login(token='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')
print("logged_in")
PY

# pip install -U torch torchvision torchaudio
# pip install -U peft soxr bitsandbytes sentencepiece
# pip install -e "git+https://github.com/huggingface/diffusers.git#egg=diffusers"
# pip install -U "accelerate>=1.0.0" "transformers>=4.47.0"

# Don't fail the whole script if apex isn't installed
pip uninstall -y apex || true

export MODEL_NAME="stabilityai/stable-diffusion-3.5-medium"

export INSTANCE_DIR="C:/Users/2232570/Downloads/memgrad_sedev (2)/memgrad/SurgUn/distractor_new-20260724T002325Z-1-001/distractor_new"

OUTPUT_BASE="G:/pilot_flux/obj_dist_noise_dist_model_input_sd3.5_1e-4_all_layers_precond_off"
# ----------------------------------------------------------
# Training data
# ----------------------------------------------------------
DIST_DIRS=(
    "C:\Users\2232570\Downloads\memgrad_sedev (2)\memgrad\surgun_flux\sd35_base_objects_no_style\Cats"
)

PROMPTS=(
  "cat"
)

VALID_PROMPTS=(
  "cat"

  )

# FLUX transformer blocks 0-18
BLOCK=($(seq 12 23))

sanitize() {
  echo "$1" | sed 's/\./_/g' | tr -cd '[:alnum:]_-'
}

# ----------------------------------------------------------
# Train each style independently over every transformer block
# ----------------------------------------------------------
for i in "${!PROMPTS[@]}"; do

  PROMPT="${PROMPTS[$i]}"
  VALID="${VALID_PROMPTS[$i]}"
  DIST_DIR="${DIST_DIRS[$i]}"

  # Make  output name safe:
  # "crayon cartoon" -> "Crayon_Cartoon"
  SAFE_PROMPT="$(echo "$PROMPT" | sed 's/ /_/g')"
  SAFE_PROMPT="${SAFE_PROMPT^}"

  for b in "${BLOCK[@]}"; do

    SAFE_BLOCK="$(sanitize "$b")"

    OUTPUT_DIR="${OUTPUT_BASE}/${SAFE_BLOCK}/${SAFE_PROMPT}_post_mean"

    echo "======================================================"
    echo "STYLE      = $PROMPT"
    echo "BLOCK      = $b"
    echo "OUTPUT_DIR = $OUTPUT_DIR"
    echo "DIST_DIR   = $DIST_DIR"
    echo "======================================================"

    accelerate launch --num_processes=1 train_dreambooth_lora_sd3.py \
      --pretrained_model_name_or_path="$MODEL_NAME" \
      --instance_data_dir="$INSTANCE_DIR" \
      --lora_blocks="$SAFE_BLOCK" \
      --lora_layers="attn.to_q,attn.to_k,attn.to_v,attn.to_out.0,attn.add_q_proj,attn.add_k_proj,attn.add_v_proj,attn.to_add_out" \
      --dist_data_dir="$DIST_DIR" \
      --output_dir="$OUTPUT_DIR" \
      --mixed_precision="bf16" \
      --instance_prompt="$PROMPT" \
      --resolution=1024 \
      --checkpointing_steps=5 \
      --rank=64 \
      --train_batch_size=1 \
      --gradient_accumulation_steps=1 \
      --learning_rate=1e-4 \
      --lr_scheduler="constant" \
      --lr_warmup_steps=0 \
      --max_train_steps=1000 \
      --precondition_outputs=0 \
      --seed=0

  done
done