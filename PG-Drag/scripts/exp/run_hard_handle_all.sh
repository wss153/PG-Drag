#!/usr/bin/env bash
# Run hard_handle only for all cases in apap_3d.txt

set -euo pipefail
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate YAPAP
cd "$(dirname "$0")/../.."
export JAXTYPING_TYPECHECKER=beartype WANDB_MODE=offline PYTHONPATH=. HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

APAP_3D_TXT="${APAP_3D_TXT:-../JAPAP/configs/deform_meshes/data/apap_3d.txt}"
NUM_GPUS="${NUM_GPUS:-2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

STAMP=$(date +%Y%m%d_%H%M%S)
ROOT="outputs/hard_handle_ablation/hard_all-${STAMP}"
LOGDIR="outputs/batch_logs"
mkdir -p "$ROOT" "$LOGDIR"
MASTER="${LOGDIR}/hard_handle_all_${STAMP}.log"

COMMON=(
  pipe-cfg:stable-diffusion3-d-pipe-config
  --pipe-cfg.stage2-steps 600
  --pipe-cfg.hard-handle
)

declare -a CASES=()
while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line// /}"
  [[ -z "$line" ]] && continue
  mesh_path="${line%%,*}"
  name=$(echo "$mesh_path" | sed -n 's|.*/processed/\([^/]*\)/mesh\.obj|\1|p')
  kp=$(echo "$line" | sed -n 's|.*keypoints/\([^/]*\)/user_single_keypoints\.txt.*|\1|p')
  lora=$(echo "$line" | awk -F', ' '{print $NF}' | sed 's|.*/||')
  [[ -n "$name" && -n "$kp" && -n "$lora" ]] || continue
  CASES+=("${name}|${kp}|${lora}")
done < "$APAP_3D_TXT"

echo "Parsed ${#CASES[@]} cases from ${APAP_3D_TXT}" | tee "$MASTER"
echo "ROOT=$ROOT NUM_GPUS=$NUM_GPUS SKIP_EXISTING=$SKIP_EXISTING (hard_handle only)" | tee -a "$MASTER"

run_one() {
  local name=$1 kp=$2 lora=$3 gpu=$4
  local out="${ROOT}/hard_handle/${name}_handle-${kp}_anchor-${kp}"
  local log="${LOGDIR}/hard_all_${name}_${STAMP}.log"
  if [[ "$SKIP_EXISTING" == "1" && -f "${out}/metrics/eval_metrics.json" ]]; then
    echo "[$(date +%H:%M:%S)] SKIP ${name} (exists)" | tee -a "$MASTER"
    return 0
  fi
  mkdir -p "$out"
  echo "[$(date +%H:%M:%S)] START ${name} gpu=${gpu}" | tee -a "$MASTER" "$log"
  CUDA_VISIBLE_DEVICES=$gpu python scripts/exp/deform_meshes.py \
    --mesh-file "data/apap_3d/processed/${name}/mesh.obj" \
    --handle-file "data/apap_3d/processed/${name}/keypoints/${kp}/user_single_keypoints.txt" \
    --anchor-file "data/apap_3d/processed/${name}/keypoints/${kp}/constraint_single_keypoints.txt" \
    --out-dir "$out" \
    --lora-dir "data/lora_ckpts/apap_3d/legacy/${lora}" \
    "${COMMON[@]}" \
    >> "$log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE ${name} exit=$?" | tee -a "$MASTER" "$log"
}

worker() {
  local gpu=$1 idx=$2
  local total=${#CASES[@]}
  while (( idx < total )); do
    IFS='|' read -r name kp lora <<< "${CASES[$idx]}"
    run_one "$name" "$kp" "$lora" "$gpu"
    idx=$((idx + NUM_GPUS))
  done
}

for ((g=0; g<NUM_GPUS; g++)); do
  worker "$g" "$g" &
done
wait

python scripts/eval/aggregate_hard_handle_ablation.py "$ROOT" || true

echo "ALL DONE ROOT=$ROOT (${#CASES[@]} hard_handle cases)" | tee -a "$MASTER"
ln -sfn "$(realpath "$ROOT")" outputs/hard_handle_ablation/hard_all_latest
