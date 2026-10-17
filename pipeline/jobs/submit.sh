#!/usr/bin/env bash
# PathQueue guarded job submission: the master's ONLY way to submit Slurm jobs (BUILD_PLAN §A19, §E; Agent 2).
#
#   pipeline/jobs/submit.sh <job> [--after JOBID] [--print-only]
#
# Run on a Delta login node from anywhere (paths resolve from this file). It never loops and never resubmits.
# Checks, in order (any failure refuses the submission, exit 1, nothing submitted):
#   1. <job> is on the allow-list of reviewed scripts and pipeline/jobs/<job>.sbatch exists;
#   2. the script's #SBATCH lines: --time present; no --reservation; GPU jobs use bfqi-delta-gpu / gpuA100x4,
#      one GPU, <= 16 CPUs, <= 60G and <= 01:00:00; CPU jobs use bfqi-delta-cpu;
#   3. event windows (CDT): now + --time must not cross Sun Oct 18 07:59 (end of the overnight reserved block),
#      the 07:59-08:00 gap, or Sun 18:00 (end of the event); --deadline is added for the block the job is in;
#   4. GPU jobs: no other GPU job of ours may be running or pending unless this one is chained after the TAIL of
#      that chain with --after (one GPU per team at a time);
#   5. GPU jobs: cumulative requested GPU time in the submission log + this job <= 7 h (420 min);
#   6. `sbatch --test-only` accepts the job. It validates syntax only: it ignores the magnetic reservations and
#      reports a far-future start, which is NOT an error.
# Every submission (and every refusal) is appended to logs/submissions.log (tab-separated). --print-only runs
# all checks and prints the sbatch command without submitting or logging.
# Bash 3.2 compatible (the tests run on macOS).
set -euo pipefail

JOBS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$JOBS_DIR/../.." && pwd)"
LOG="$CODE_DIR/logs/submissions.log"

ALLOWED="download_bach prep_a prep_b smoke embed_core embed_extra embed_aux text_cpu cells_spike cells finetune"
GPU_CAP_MIN=420                  # 7 h of Rishabh's 10 GPU-hours (BUILD_PLAN §G)
GPU_MAX_MIN=60                   # realistic 1 h GPU requests (CLAUDE.md, §C.2)
OVERNIGHT_END=1792328340         # Sun 2026-10-18 07:59:00 CDT (12:59:00Z): overnight reserved block ends
DAY_START=1792328400             # Sun 2026-10-18 08:00:00 CDT: Sunday reserved block starts
DAY_END=1792364400               # Sun 2026-10-18 18:00:00 CDT: Sunday block, hosted servers and event end
OVERNIGHT_DEADLINE=2026-10-18T07:59:00
DAY_DEADLINE=2026-10-18T18:00:00

usage() { echo "usage: $0 <job> [--after JOBID] [--print-only]   jobs: $ALLOWED" >&2; exit 64; }

JOB="${1:-}"
[ -n "$JOB" ] || usage
shift
AFTER=""
PRINT_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --after) [ $# -ge 2 ] || usage; AFTER="$2"; shift 2 ;;
        --after=*) AFTER="${1#--after=}"; shift ;;
        --print-only) PRINT_ONLY=1; shift ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
done

NOW="$(date +%s)"
CHECKS=""
note() { CHECKS="${CHECKS:+$CHECKS,}$1"; }

log_line() {  # status job_id gpu minutes extra
    mkdir -p "$(dirname "$LOG")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$JOB" "$2" "$3" "$4" \
        "${AFTER:--}" "${DEADLINE:--}" "$5" >> "$LOG"
}

refuse() {
    echo "REFUSED $JOB: $*" >&2
    if [ "$PRINT_ONLY" -eq 0 ]; then log_line refused - "${IS_GPU:-?}" "${MINUTES:-?}" "$*"; fi
    exit 1
}

# 1. allow-list ---------------------------------------------------------------------------------------------
case " $ALLOWED " in
    *" $JOB "*) ;;
    *) refuse "job '$JOB' is not on the allow-list ($ALLOWED)" ;;
esac
SCRIPT="$JOBS_DIR/$JOB.sbatch"
[ -f "$SCRIPT" ] || refuse "script $SCRIPT does not exist"
note allow_list
if [ -n "$AFTER" ]; then
    case "$AFTER" in *[!0-9]*|"") refuse "--after needs a numeric job id, got '$AFTER'" ;; esac
fi

# 2. #SBATCH directives ---------------------------------------------------------------------------------------
directive() {  # long-name short-flag -> value of the first matching #SBATCH line ('' if absent)
    sed -n -E "s/^#SBATCH[[:space:]]+(--$1=|--$1[[:space:]]+${2:+|-$2[[:space:]]*})([^[:space:]#]+).*/\2/p" "$SCRIPT" | head -n 1
}
grep -Eq '^#SBATCH[[:space:]]+--reservation' "$SCRIPT" && refuse "--reservation is not allowed (magnetic reservations attach automatically)"
TIME_RAW="$(directive time t)"
PARTITION="$(directive partition p)"
ACCOUNT="$(directive account A)"
[ -n "$TIME_RAW" ] || refuse "no #SBATCH --time in $SCRIPT"
[ -n "$PARTITION" ] || refuse "no #SBATCH --partition in $SCRIPT"

to_minutes() {  # Slurm time: M | M:S | H:M:S | D-H | D-H:M | D-H:M:S -> whole minutes (rounded up)
    local t="$1" d=0 h=0 m=0 s=0 rest
    case "$t" in *-*) d="${t%%-*}"; rest="${t#*-}" ;; *) rest="$t" ;; esac
    IFS=: read -r a b c <<EOF
$rest
EOF
    if [ "$t" != "$rest" ]; then h="${a:-0}"; m="${b:-0}"; s="${c:-0}"
    elif [ -n "${c:-}" ]; then h="$a"; m="$b"; s="$c"
    elif [ -n "${b:-}" ]; then m="$a"; s="$b"
    else m="$a"; fi
    for v in "$d" "$h" "$m" "$s"; do case "$v" in *[!0-9]*|"") echo "bad"; return; esac; done
    echo $(( (10#$d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s + 59) / 60 ))
}
MINUTES="$(to_minutes "$TIME_RAW")"
[ "$MINUTES" != "bad" ] || refuse "cannot parse --time=$TIME_RAW"
[ "$MINUTES" -gt 0 ] || refuse "--time must be positive"

IS_GPU=0
case "$PARTITION" in *gpu*) IS_GPU=1 ;; esac
if [ "$IS_GPU" -eq 1 ]; then
    [ "$ACCOUNT" = "bfqi-delta-gpu" ] || refuse "GPU job must use --account=bfqi-delta-gpu (got '$ACCOUNT')"
    [ "$PARTITION" = "gpuA100x4" ] || refuse "GPU job must use --partition=gpuA100x4 (got '$PARTITION')"
    GPUS="$(directive gpus-per-node)"
    [ "$GPUS" = "1" ] || refuse "GPU job must request exactly --gpus-per-node=1 (got '${GPUS:-none}')"
    CPUS="$(directive cpus-per-task c)"
    case "$CPUS" in ''|*[!0-9]*) refuse "GPU job needs a numeric --cpus-per-task" ;; esac
    [ "$CPUS" -le 16 ] || refuse "--cpus-per-task=$CPUS exceeds 16 per GPU"
    MEM="$(directive mem)"
    case "$MEM" in
        *[Gg]) MEM_G="${MEM%[Gg]}" ;;
        *[Mm]) MEM_G=$(( ${MEM%[Mm]} / 1024 )) ;;
        *) refuse "GPU job needs --mem in G or M (got '${MEM:-none}')" ;;
    esac
    case "$MEM_G" in ''|*[!0-9]*) refuse "cannot parse --mem=$MEM" ;; esac
    [ "$MEM_G" -le 60 ] || refuse "--mem=$MEM exceeds 60G per GPU"
    [ "$MINUTES" -le "$GPU_MAX_MIN" ] || refuse "GPU --time=$TIME_RAW exceeds 01:00:00"
else
    [ "$ACCOUNT" = "bfqi-delta-cpu" ] || refuse "CPU job must use --account=bfqi-delta-cpu (got '$ACCOUNT')"
fi
note directives

# 3. event windows ---------------------------------------------------------------------------------------------
END=$(( NOW + MINUTES * 60 ))
DEADLINE=""
if [ "$NOW" -ge "$DAY_END" ]; then
    refuse "the event windows ended Sun Oct 18 18:00 CDT"
elif [ "$NOW" -lt "$OVERNIGHT_END" ]; then
    [ "$END" -le "$OVERNIGHT_END" ] || refuse "now + --time=$TIME_RAW crosses Sun 07:59 CDT (end of the overnight block)"
    DEADLINE="$OVERNIGHT_DEADLINE"
elif [ "$NOW" -lt "$DAY_START" ]; then
    refuse "now is inside the 07:59-08:00 gap between reserved blocks; wait for 08:00"
else
    [ "$END" -le "$DAY_END" ] || refuse "now + --time=$TIME_RAW crosses Sun 18:00 CDT (end of the event)"
    DEADLINE="$DAY_DEADLINE"
fi
note "window(deadline=$DEADLINE)"

# 4. one GPU at a time -------------------------------------------------------------------------------------------
if [ "$IS_GPU" -eq 1 ]; then
    QUEUE="$(squeue -u "${USER:-$(id -un)}" -h -o '%i|%P|%T|%E')" || refuse "squeue failed"
    GPU_IDS=""
    while IFS='|' read -r jid part state dep; do
        [ -n "$jid" ] || continue
        case "$part" in *gpu*) GPU_IDS="$GPU_IDS $jid" ;; esac
    done <<EOF
$QUEUE
EOF
    if [ -n "$GPU_IDS" ]; then
        [ -n "$AFTER" ] || refuse "GPU job(s)$GPU_IDS already running/pending; chain with --after <tail job id>"
        case " $GPU_IDS " in *" $AFTER "*) ;; *) refuse "--after $AFTER is not one of our queued GPU jobs:$GPU_IDS" ;; esac
        ROOTS=0
        while IFS='|' read -r jid part state dep; do
            case "$part" in *gpu*) ;; *) continue ;; esac
            DEP_IDS=" $(printf '%s' "$dep" | tr -c '0-9' ' ') "
            case "$DEP_IDS" in *" $AFTER "*) refuse "job $jid already depends on $AFTER; chain after the tail of the chain" ;; esac
            IN_CHAIN=0
            for g in $GPU_IDS; do case "$DEP_IDS" in *" $g "*) IN_CHAIN=1 ;; esac; done
            [ "$IN_CHAIN" -eq 1 ] || ROOTS=$(( ROOTS + 1 ))
        done <<EOF
$QUEUE
EOF
        [ "$ROOTS" -le 1 ] || refuse "queued GPU jobs$GPU_IDS do not form one dependency chain"
    fi
    note one_gpu

    # 5. cumulative GPU time cap ---------------------------------------------------------------------------------
    USED=0
    if [ -f "$LOG" ]; then
        USED="$(awk -F'\t' '$2 == "submitted" && $5 == "1" { s += $6 } END { print s + 0 }' "$LOG")"
    fi
    [ $(( USED + MINUTES )) -le "$GPU_CAP_MIN" ] || \
        refuse "requested GPU time would reach $(( USED + MINUTES )) min > cap $GPU_CAP_MIN min (already $USED)"
    note "gpu_cap(${USED}+${MINUTES}/${GPU_CAP_MIN}min)"
fi

# 6. sbatch --test-only (syntax only; its start estimate ignores the magnetic reservations) ---------------------------
EXTRA="--deadline=$DEADLINE"
[ -z "$AFTER" ] || EXTRA="$EXTRA --dependency=afterok:$AFTER"
if ! TEST_OUT="$(sbatch --test-only $EXTRA "$SCRIPT" 2>&1)"; then
    refuse "sbatch --test-only rejected the job: $TEST_OUT"
fi
note test_only
echo "test-only (syntax ok; start estimate ignores reservations): $TEST_OUT"

if [ "$PRINT_ONLY" -eq 1 ]; then
    echo "PRINT-ONLY: sbatch --parsable $EXTRA $SCRIPT"
    echo "checks passed: $CHECKS"
    exit 0
fi
OUT="$(sbatch --parsable $EXTRA "$SCRIPT")"
JOB_ID="${OUT%%;*}"
case "$JOB_ID" in ''|*[!0-9]*) refuse "sbatch returned an unexpected job id: $OUT" ;; esac
log_line submitted "$JOB_ID" "$IS_GPU" "$MINUTES" "$CHECKS"
echo "SUBMITTED $JOB as job $JOB_ID (time ${TIME_RAW}, gpu=$IS_GPU, deadline $DEADLINE${AFTER:+, afterok:$AFTER})"
echo "checks passed: $CHECKS"
echo "log: $LOG"
