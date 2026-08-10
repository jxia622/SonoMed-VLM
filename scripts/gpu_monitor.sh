#!/bin/bash
set -euo pipefail

prefix="${1:?Usage: gpu_monitor.sh OUTPUT_PREFIX}"
host="$(hostname)"
output="${prefix}-${host}.csv"
printf '%s\n' 'timestamp,hostname,index,uuid,name,utilization_gpu_percent,memory_used_mib,memory_total_mib,power_draw_watts' > "$output"
while true; do
  timestamp="$(date --iso-8601=seconds)"
  while IFS= read -r row; do
    printf '%s,%s,%s\n' "$timestamp" "$host" "$row" >> "$output"
  done < <(
    nvidia-smi \
      --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw \
      --format=csv,noheader,nounits
  )
  sleep 10
done
