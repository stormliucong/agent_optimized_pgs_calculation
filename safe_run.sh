#!/usr/bin/env bash
# safe_run.sh — run a command inside a memory-bounded systemd scope, watched by
# a machine-wide memory watchdog, so a runaway job cannot take this host down.
#
# WHY TWO MECHANISMS (both measured on this box, 2026-08-14):
#
#   1. cgroup MemoryMax  — stops host-RAM runaways.  Verified: a python asking
#      for 3 GB under `-p MemoryMax=2G` was killed, rc=137.
#      It does NOT stop GPU runaways: this is a GB10, the GPU has no memory of
#      its own, and a 10 GB cupy allocation SURVIVED a 6 GB MemoryMax — the
#      nvidia driver's pages are not charged to the process cgroup.
#
#   2. MemAvailable watchdog — the backstop for exactly that hole.  Unified
#      memory does show up machine-wide: a 20 GB device buffer moved
#      MemAvailable from 119.0 GB to 97.4 GB.  So watching /proc/meminfo
#      catches what the cgroup cannot see.
#
#   A third, cheaper layer is exported here rather than enforced: cupy honours
#   CUPY_GPU_MEMORY_LIMIT (bytes or "50%" — NOT "48GB", that raises ValueError)
#   and raises OutOfMemoryError instead of eating the machine.  That turns a
#   host-killing allocation into a normal python traceback the caller can see.
#
# On 2026-08-11 an unguarded pressure-test run did exactly what this script
# prevents: NVRM ran out of memory at 06:21, the box thrashed for 2.5 hours,
# and the kernel OOM killer took out gdm3, systemd-logind, dbus, cron, avahi
# and the agent's own helpers before anything recovered.
#
# usage:  ./safe_run.sh [options] -- command [args...]
#
# options (all have env-var equivalents in caps):
#   --mem 72G       MEM        cgroup MemoryMax for the whole process tree
#   --floor 12      FLOOR_GB   kill when machine-wide MemAvailable drops below
#   --gpu 48G       GPU        CUPY_GPU_MEMORY_LIMIT (accepts G/GB/%/bytes)
#   --grace 20      GRACE      seconds below the floor before killing
#   --interval 5    INTERVAL   watchdog sampling period, seconds
#   --label name    LABEL      scope name / watchdog log name
#   --log path      WLOG       watchdog log (default ./<label>.watchdog.log)
#   --no-scope                 skip the cgroup, keep the watchdog
#
# exit codes: the command's own, except 137 (killed — see the watchdog log).

set -uo pipefail

MEM="${MEM:-72G}"
FLOOR_GB="${FLOOR_GB:-12}"
GPU="${GPU:-48G}"
GRACE="${GRACE:-20}"
INTERVAL="${INTERVAL:-5}"
LABEL="${LABEL:-job}"
WLOG="${WLOG:-}"
USE_SCOPE=1

while [ $# -gt 0 ]; do
  case "$1" in
    --mem)      MEM="$2";      shift 2 ;;
    --floor)    FLOOR_GB="$2"; shift 2 ;;
    --gpu)      GPU="$2";      shift 2 ;;
    --grace)    GRACE="$2";    shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --label)    LABEL="$2";    shift 2 ;;
    --log)      WLOG="$2";     shift 2 ;;
    --no-scope) USE_SCOPE=0;   shift ;;
    --)         shift; break ;;
    -h|--help)  sed -n '1,45p' "$0"; exit 0 ;;
    *)          echo "safe_run.sh: unknown option $1" >&2; exit 2 ;;
  esac
done
[ $# -gt 0 ] || { echo "safe_run.sh: nothing to run (use -- command ...)" >&2; exit 2; }

WLOG="${WLOG:-./${LABEL}.watchdog.log}"

# ── cupy's limit wants bytes or a percent string; be liberal about the input ──
to_bytes() {                       # 48G | 48GB | 48g | 51539607552 | 40%
  local v="${1//[Bb]/}"; v="${v// /}"
  case "$v" in
    *%)          printf '%s' "$v" ;;
    *[Gg])       printf '%s' $(( ${v%[Gg]} * 1024 * 1024 * 1024 )) ;;
    *[Mm])       printf '%s' $(( ${v%[Mm]} * 1024 * 1024 )) ;;
    ''|*[!0-9]*) printf '%s' "" ;;
    *)           printf '%s' "$v" ;;
  esac
}
GPU_BYTES="$(to_bytes "$GPU")"
if [ -n "$GPU_BYTES" ] && [ -z "${CUPY_GPU_MEMORY_LIMIT:-}" ]; then
  export CUPY_GPU_MEMORY_LIMIT="$GPU_BYTES"
fi

TOTAL_KB=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
FLOOR_KB=$(( FLOOR_GB * 1024 * 1024 ))
UNIT="safe-${LABEL}-$$.scope"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
wlog()  { printf '%s  %s\n' "$(stamp)" "$*" >>"$WLOG"; }

# ── watchdog ────────────────────────────────────────────────────────────────
# Runs OUTSIDE the scope on purpose: it must survive the thing it kills, and
# still be alive afterwards to write down why.
watchdog() {
  local target_pid="$1" strikes=0 need peak_used=0 avail used ticks=0 beat
  need=$(( GRACE / INTERVAL )); [ "$need" -lt 1 ] && need=1
  beat=$(( 600 / INTERVAL )); [ "$beat" -lt 1 ] && beat=1   # proof of life, 10 min
  while kill -0 "$target_pid" 2>/dev/null; do
    avail=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
    used=$(( TOTAL_KB - avail ))
    [ "$used" -gt "$peak_used" ] && peak_used=$used
    ticks=$(( ticks + 1 ))
    if [ $(( ticks % beat )) -eq 0 ]; then
      wlog "alive: using $(( used / 1048576 )) GB, peak $(( peak_used / 1048576 )) GB, MemAvailable $(( avail / 1048576 )) GB"
    fi
    if [ "$avail" -lt "$FLOOR_KB" ]; then
      strikes=$(( strikes + 1 ))
      wlog "LOW MEMORY: MemAvailable $(( avail / 1048576 )) GB < floor ${FLOOR_GB} GB  (strike ${strikes}/${need})"
      if [ "$strikes" -ge "$need" ]; then
        wlog "KILLING — top consumers at the moment of the kill:"
        ps -eo pid,ppid,rss,etime,comm --sort=-rss 2>/dev/null | head -10 >>"$WLOG"
        timeout 5 nvidia-smi --query-gpu=memory.used,memory.total \
          --format=csv,noheader 2>/dev/null >>"$WLOG"
        if [ "$USE_SCOPE" -eq 1 ]; then
          systemctl --user kill --kill-whom=all -s TERM "$UNIT" 2>/dev/null
          sleep 10
          systemctl --user kill --kill-whom=all -s KILL "$UNIT" 2>/dev/null
        else
          kill -TERM -"$target_pid" 2>/dev/null; sleep 10
          kill -KILL -"$target_pid" 2>/dev/null
        fi
        wlog "killed the job; the machine was NOT allowed to thrash"
        return 0
      fi
    else
      [ "$strikes" -gt 0 ] && wlog "recovered: MemAvailable $(( avail / 1048576 )) GB"
      strikes=0
    fi
    sleep "$INTERVAL"
  done
  wlog "job exited on its own; peak machine memory used $(( peak_used / 1048576 )) GB of $(( TOTAL_KB / 1048576 )) GB"
}

wlog "=== safe_run: ${LABEL} ==="
wlog "cmd: $*"
wlog "cgroup MemoryMax=${MEM} swap=0 | MemAvailable floor=${FLOOR_GB} GB (grace ${GRACE}s) | CUPY_GPU_MEMORY_LIMIT=${CUPY_GPU_MEMORY_LIMIT:-unset}"

# ── run ─────────────────────────────────────────────────────────────────────
if [ "$USE_SCOPE" -eq 1 ] && ! systemctl --user show-environment >/dev/null 2>&1; then
  wlog "WARN: no usable --user systemd instance; running with the watchdog only"
  USE_SCOPE=0
fi

if [ "$USE_SCOPE" -eq 1 ]; then
  # OOMPolicy=continue is load-bearing.  systemd's default for a scope is
  # OOMPolicy=stop: when the kernel memcg-OOM-kills ANY process in the cgroup,
  # systemd marks the unit "Failed with result 'oom-kill'" and tears the WHOLE
  # scope down.  That is what killed the 2026-09-01 variance run — one scorer's
  # worker fan-out hit MemoryMax and took `claude` with it (rc=143), 4.5 h in.
  # With `continue`, the kernel still kills the fattest process (the scorer, see
  # the oom_score_adj bump in run_1kg.py), the harness sees a dead child and
  # records status=OOM, and the agent keeps running.
  systemd-run --user --scope --quiet --unit="$UNIT" --collect \
    -p MemoryMax="$MEM" -p MemoryHigh="$MEM" -p MemorySwapMax=0 -p TasksMax=8192 \
    -p OOMPolicy=continue \
    -- "$@" &
else
  setsid "$@" &
fi
JOB=$!

watchdog "$JOB" &
WATCH=$!

trap 'kill -TERM "$JOB" 2>/dev/null; [ "$USE_SCOPE" -eq 1 ] && systemctl --user kill --kill-whom=all -s TERM "$UNIT" 2>/dev/null' INT TERM

wait "$JOB"; RC=$?
# give the watchdog one sampling period to notice the exit and write its peak
for _ in $(seq 1 $(( INTERVAL + 2 ))); do
  kill -0 "$WATCH" 2>/dev/null || break
  sleep 1
done
kill "$WATCH" 2>/dev/null; wait "$WATCH" 2>/dev/null

if [ "$RC" -eq 137 ]; then
  wlog "exit 137 — SIGKILL.  Either the cgroup cap (host RAM) or the watchdog (machine-wide) fired; the lines above say which."
elif [ "$RC" -eq 143 ]; then
  wlog "exit 143 — SIGTERM.  If the watchdog logged no kill above, systemd tore the scope down; check: journalctl -k --since '-1h' | grep -i oom"
else
  wlog "exit ${RC}"
fi
exit "$RC"
