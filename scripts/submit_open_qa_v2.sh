#!/bin/bash
# Both 1% training+evaluation jobs must pass before any larger training run starts.
set -euo pipefail
: "${SONOMED_PROJECT_ROOT:?Set SONOMED_PROJECT_ROOT}"
cd "$SONOMED_PROJECT_ROOT"
mkdir -p logs
receipt=submission-openqa-v2.tsv
[[ ! -e "$receipt" ]] || { echo "Receipt already exists; inspect before retrying." >&2; exit 1; }
printf 'kind\tfamily\tscale\tjob_id_cluster\ttime_limit\n' > "$receipt"
all_jobs=()
gates=()
for family in qwen medgemma; do
  job=$(sbatch --parsable --qos=gpu-rtx6k-s --job-name="oqa2-${family}-1pct" \
    --time=03:00:00 slurm/open_qa_v2.slurm "$family" 1)
  gates+=("${job%%;*}")
  all_jobs+=("${job%%;*}")
  printf 'train_eval\t%s\t1\t%s\t03:00:00\n' "$family" "$job" | tee -a "$receipt"
done
dep="afterok:${gates[0]}:${gates[1]}"
for family in qwen medgemma; do
  for scale in 5 10 25 50 100; do
    if [[ "$family" == qwen ]]; then
      case "$scale" in 5) limit=03:00:00;; 10) limit=04:00:00;; 25) limit=06:00:00;; 50) limit=09:00:00;; 100) limit=14:00:00;; esac
    else
      case "$scale" in 5) limit=04:00:00;; 10) limit=05:00:00;; 25) limit=08:00:00;; 50) limit=16:00:00;; 100) limit=24:00:00;; esac
    fi
    job=$(sbatch --parsable --qos=gpu-rtx6k-s --job-name="oqa2-${family}-${scale}pct" \
      --time="$limit" --dependency="$dep" --kill-on-invalid-dep=yes slurm/open_qa_v2.slurm "$family" "$scale")
    all_jobs+=("${job%%;*}")
    printf 'train_eval\t%s\t%s\t%s\t%s\n' "$family" "$scale" "$job" "$limit" | tee -a "$receipt"
  done
  job=$(sbatch --parsable --qos=gpu-rtx6k-s --job-name="oqa2-${family}-baselines" \
    --time=06:00:00 --dependency="$dep" --kill-on-invalid-dep=yes slurm/open_qa_v2.slurm "$family" baselines)
  all_jobs+=("${job%%;*}")
  printf 'baselines\t%s\toriginal+legacy100\t%s\t06:00:00\n' "$family" "$job" | tee -a "$receipt"
done
# Same-cluster dependencies avoid ambiguous cross-cluster job IDs. CRC requires
# one GPU even for this CPU scorer; reserve the minimum for a short final job.
summary_dep="afterok:$(IFS=:; echo "${all_jobs[*]}")"
job=$(sbatch --parsable --qos=gpu-rtx6k-s --dependency="$summary_dep" \
  --kill-on-invalid-dep=yes slurm/open_qa_summary.slurm)
printf 'summary\tboth\tall\t%s\t00:30:00\n' "$job" | tee -a "$receipt"
