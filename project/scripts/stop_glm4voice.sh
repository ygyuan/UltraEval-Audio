#!/usr/bin/env bash
# Force-kill any leftover GLM-4-Voice eval processes.
#
# Use this if ``run_eval_glm4voice.sh`` was killed in a way that left the
# ``model_server.py`` / ``ultraeval_adapter_server.py`` (or the eval client)
# orphaned -- e.g. power loss, ``kill -9`` on the wrapper shell, etc.
#
# Strategy:
#   1. Kill anything still listening on the model_server / adapter ports.
#   2. Pattern-match on python command lines for the two server scripts and
#      ``audio_evals/main.py`` invocations using ``model glm-4-voice``.
#   3. As a final fallback, take down every process group recorded in the
#      most recent ``project/scripts/logs/glm4voice_*/server.pids`` file.
set -u

MODEL_SERVER_PORT="${MODEL_SERVER_PORT:-10000}"
ADAPTER_PORT="${ADAPTER_PORT:-10001}"

kill_pids() {
    local label="$1"; shift
    local pids="$*"
    for pid in ${pids}; do
        [ -z "${pid}" ] && continue
        if kill -0 "${pid}" 2>/dev/null; then
            echo "[stop] ${label}: SIGTERM pid=${pid}"
            kill -TERM "${pid}" 2>/dev/null
        fi
    done
    sleep 2
    for pid in ${pids}; do
        [ -z "${pid}" ] && continue
        if kill -0 "${pid}" 2>/dev/null; then
            echo "[stop] ${label}: SIGKILL pid=${pid}"
            kill -KILL "${pid}" 2>/dev/null
        fi
    done
}

# 1. Port-based discovery
for port in "${MODEL_SERVER_PORT}" "${ADAPTER_PORT}"; do
    pids=$(ss -ltnp 2>/dev/null | awk -v p=":${port}" '$4 ~ p {print $0}' | \
           grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
    if [ -n "${pids}" ]; then
        kill_pids "port ${port}" ${pids}
    fi
done

# 2. Pattern-based discovery
patterns=(
    "third_party/GLM-4-Voice/model_server.py"
    "third_party/GLM-4-Voice/ultraeval_adapter_server.py"
    "audio_evals/main.py.*--model glm-4-voice"
)
for pat in "${patterns[@]}"; do
    pids=$(pgrep -f "${pat}" 2>/dev/null || true)
    if [ -n "${pids}" ]; then
        kill_pids "pattern '${pat}'" ${pids}
    fi
done

# 3. Process-group fallback from the latest server.pids sidecar.
LATEST_PID_FILE=$(ls -t project/scripts/logs/glm4voice_*/server.pids 2>/dev/null | head -n 1 || true)
if [ -n "${LATEST_PID_FILE}" ] && [ -f "${LATEST_PID_FILE}" ]; then
    echo "[stop] using ${LATEST_PID_FILE}"
    while read -r line; do
        pgid=$(echo "${line}" | grep -oE 'pgid=[0-9]+' | cut -d= -f2)
        [ -z "${pgid}" ] && continue
        if kill -0 "-${pgid}" 2>/dev/null; then
            echo "[stop] SIGTERM process group ${pgid}"
            kill -TERM -- "-${pgid}" 2>/dev/null
            sleep 2
            if kill -0 "-${pgid}" 2>/dev/null; then
                echo "[stop] SIGKILL process group ${pgid}"
                kill -KILL -- "-${pgid}" 2>/dev/null
            fi
        fi
    done < "${LATEST_PID_FILE}"
    rm -f "${LATEST_PID_FILE}"
fi

echo "[stop] done."
