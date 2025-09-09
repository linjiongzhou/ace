#!/usr/bin/env bash
# start_mem_logger.sh — per-node CPU/GPU memory logger (exclusive job mode, aligned output)
# Adds Δ since last sample, HWM, and optional burst detection with temporary faster sampling.

start_mem_logger() {
  local interval="${LOG_INTERVAL:-10}"         # base cadence (s)

  # Optional burst zoom-in (set BURST_GB or BURST_PCT to enable)
  local burst_gb="${BURST_GB:-0}"              # e.g., 8 means trigger if jump > 8 GB
  local burst_pct="${BURST_PCT:-0}"            # e.g., 5 means trigger if jump >5% of total
  local adapt_interval="${ADAPT_INTERVAL:-2}"  # faster cadence after burst (s)
  local adapt_cycles="${ADAPT_CYCLES:-10}"     # how many fast samples after burst

  local logroot="${MEM_LOG_DIR:-/scratch4/GFDL/gfdlscr/Linjiong.Zhou/ace/mem_logs}"

  # Use Slurm’s %x.o%j -> $SLURM_JOB_NAME.o$SLURM_JOB_ID
  local jobname="${SLURM_JOB_NAME:-unknown}"
  # sanitize just in case (spaces/slashes)
  jobname="${jobname//[[:space:]]/_}"
  jobname="${jobname//\//_}"

  local jobtag="${jobname}.o${SLURM_JOB_ID:-manual}"
  local dir="${logroot}/${jobtag}"
  mkdir -p "$dir"

  local logfile="${dir}/$(hostname).log"

  # Discover GPU totals once
  mapfile -t GPU_INFO < <(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits 2>/dev/null || true)
  local gpu_count="${#GPU_INFO[@]}"
  declare -a gpu_total_gb gpu_prev_used_gb gpu_hwm_gb
  for ((i=0;i<gpu_count;i++)); do
    gpu_total_gb[i]=$(awk -F, '{printf "%.1f", $2/1024}' <<<"${GPU_INFO[i]}")
    gpu_prev_used_gb[i]="0.0"
    gpu_hwm_gb[i]="0.0"
  done

  # CPU prev/HWM
  local cpu_prev_used_gb="0.0"
  local cpu_hwm_gb="0.0"

  _pct() { awk -v u="$1" -v t="$2" 'BEGIN{printf "%3d", (t>0? int((u*100)/t+0.5):0)}'; }

  local turbo_left=0

  (
    echo "=== logger start: $(date) ==="
    echo "jobid=${SLURM_JOB_ID:-N/A} node=$(hostname) GPUs=${SLURM_STEP_GPUS:-N/A}"

    while true; do
      ts="$(date '+%Y-%m-%d %H:%M:%S')"

      # --- CPU memory (GB) ---
      mem_total=$(awk '/MemTotal:/ {printf "%.1f",$2/1048576}' /proc/meminfo)
      mem_avail=$(awk '/MemAvailable:/ {printf "%.1f",$2/1048576}' /proc/meminfo)
      mem_used=$(awk -v t="$mem_total" -v a="$mem_avail" 'BEGIN {printf "%.1f", t-a}')
      mem_left=$(awk -v t="$mem_total" -v u="$mem_used" 'BEGIN {printf "%.1f", t-u}')
      mem_used_pct=$(_pct "$mem_used" "$mem_total")
      cpu_delta=$(awk -v a="$mem_used" -v b="$cpu_prev_used_gb" 'BEGIN{printf "%.1f", a-b}')
      cpu_hwm_gb=$(awk -v a="$mem_used" -v h="$cpu_hwm_gb" 'BEGIN{printf "%.1f", (a>h?a:h)}')

      burst=""
      if (( burst_gb > 0 || burst_pct > 0 )); then
        cpu_abs_burst=$(awk -v d="$cpu_delta" -v th="$burst_gb" 'BEGIN{print (th>0 && d>th)?1:0}')
        cpu_pct_delta=$(_pct "$cpu_delta" "$mem_total")
        cpu_pct_burst=$(( (burst_pct>0 && cpu_pct_delta>=burst_pct) ? 1 : 0 ))
        if (( cpu_abs_burst==1 || cpu_pct_burst==1 )); then
          burst="[BURST!]"
          turbo_left=$adapt_cycles
        fi
      fi

      line=$(printf "ts=%s  |  CPU used=%6.1fGB/%6.1fGB(%3s%%) left=%6.1fGB Δcpu=%+8.1fGB cpu_hwm=%6.1fGB" \
                   "$ts" "$mem_used" "$mem_total" "$mem_used_pct" "$mem_left" "$cpu_delta" "$cpu_hwm_gb")

      # --- GPU memory (GB) ---
      mapfile -t GPU_USED_MB < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || true)
      for ((i=0;i<gpu_count;i++)); do
        used=$(awk -v m="${GPU_USED_MB[i]}" 'BEGIN{printf "%.1f", m/1024}')
        total="${gpu_total_gb[i]}"
        left=$(awk -v u="$used" -v t="$total" 'BEGIN{printf "%.1f", t-u}')
        pct=$(_pct "$used" "$total")
        delta=$(awk -v a="$used" -v b="${gpu_prev_used_gb[i]}" 'BEGIN{printf "%.1f", a-b}')
        gpu_hwm_gb[i]=$(awk -v a="$used" -v h="${gpu_hwm_gb[i]}" 'BEGIN{printf "%.1f", (a>h?a:h)}')

        if (( burst_gb > 0 || burst_pct > 0 )); then
          g_abs_burst=$(awk -v d="$delta" -v th="$burst_gb" 'BEGIN{print (th>0 && d>th)?1:0}')
          g_pct_delta=$(_pct "$delta" "$total")
          g_pct_burst=$(( (burst_pct>0 && g_pct_delta>=burst_pct) ? 1 : 0 ))
          if (( g_abs_burst==1 || g_pct_burst==1 )); then
            burst="[BURST!]"
            turbo_left=$adapt_cycles
          fi
        fi

        line+=" $(printf " |  gpu%-1s used=%6.1fGB/%6.1fGB(%3s%%) left=%6.1fGB Δgpu%-1s=%+6.1fGB gpu%-1s_hwm=%5.1fGB" \
                          "$i" "$used" "$total" "$pct" "$left" "$i" "$delta" "$i" "${gpu_hwm_gb[i]}")"
        gpu_prev_used_gb[i]="$used"
      done

      [[ -n "$burst" ]] && line+="  $burst"
      echo "$line"

      cpu_prev_used_gb="$mem_used"

      if (( turbo_left > 0 )); then
        sleep "$adapt_interval"
        turbo_left=$((turbo_left-1))
      else
        sleep "$interval"
      fi
    done
  ) >> "$logfile" 2>&1 &

  export MEM_LOGGER_PID=$!
  trap "kill $MEM_LOGGER_PID >/dev/null 2>&1 || true" EXIT
  echo "Mem logger started (pid=$MEM_LOGGER_PID), interval=${interval}s -> $logfile"
}
