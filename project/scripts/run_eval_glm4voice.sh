#!/usr/bin/env bash
# Run UltraEval-Audio evaluation for GLM-4-Voice (THUDM / zai-org).
#
# GLM-4-Voice inference is HTTP-based: we need to start two servers before
# launching audio_evals/main.py:
#   1. model_server.py  (port 10000)  -- the GLM-4-Voice 9B LLM worker
#   2. ultraeval_adapter_server.py (port 10001) -- the speech tokenizer +
#      audio decoder adapter that audio_evals.models.glm4voice.GLM4Voice
#      talks to.
#
# Local checkpoints (under init_model/zai-org/):
#   - glm-4-voice-9b         : main LLM
#   - glm-4-voice-tokenizer  : Whisper-VQ speech tokenizer
#   - glm-4-voice-decoder    : CosyVoice flow + hifigan vocoder
#
# NOTE: ``--use_model_pool`` is intentionally NOT used here because the
# model is a remote APIModel (not an OfflineModel), so a single client can
# already issue concurrent requests via ``--workers``.
#
# Robustness notes (fixes for previous "Ctrl+C cannot stop" issues):
#   * ``set -m`` (job control) makes every ``&``-backgrounded server its
#     own process-group leader, so ``$!`` matches the python PID *and*
#     pgid.  Cleanup uses ``kill -- -<pgid>`` to take down the whole
#     subtree (uvicorn worker, torch CUDA threads, ffmpeg, ...).
#   * The trap handler runs cleanup *exactly once* and exits immediately on
#     SIGINT/SIGTERM, so a second Ctrl+C is not required.
#   * Before launching, ``free_port`` proactively kills any orphan listener
#     left behind by a previous run -- otherwise the new server would
#     crash with EADDRINUSE while ``wait_for_port`` happily connects to
#     the stale listener (this is what broke app-2026-05-21_17-58 run).
#   * ``wait_for_port`` only declares ready when *our* PID owns the port,
#     not when any process is listening on it.
#   * ``set -e`` is intentionally NOT used at the eval level: ``run_eval``
#     swallows per-dataset failures; if a server itself dies mid-run we
#     abort the script loudly instead of silently skipping every dataset.
#   * HuggingFace network access is disabled (offline mode) to avoid the
#     ~80s retry storms visible in app-2026-05-19_19-5*.log; all datasets
#     are loaded from ``init_model/`` instead.
set -uo pipefail
# Enable job control so every backgrounded job becomes its own process
# group leader (pgid == pid).  This is what lets us use
# ``kill -- -<pgid>`` to take down the whole subtree.  Without this, the
# children inherit the wrapper shell's pgid and partial cleanup leaves
# CUDA worker threads / uvicorn workers behind.
set -m

current_dir=$(pwd)
cd "${current_dir}"

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
GLM_VOICE_REPO="${current_dir}/third_party/GLM-4-Voice"
MODEL_DIR="${current_dir}/init_model/zai-org/glm-4-voice-9b"
TOKENIZER_DIR="${current_dir}/init_model/zai-org/glm-4-voice-tokenizer"
DECODER_DIR="${current_dir}/init_model/zai-org/glm-4-voice-decoder"

MODEL_SERVER_HOST="127.0.0.1"
MODEL_SERVER_PORT="${MODEL_SERVER_PORT:-10000}"
ADAPTER_HOST="127.0.0.1"
ADAPTER_PORT="${ADAPTER_PORT:-10001}"

# GPU layout: model_server.py and ultraeval_adapter_server.py both need GPUs.
# By default put model_server on cuda:0 and adapter on cuda:1; the eval
# client itself is CPU-only.
MODEL_SERVER_GPU="${MODEL_SERVER_GPU:-0}"
ADAPTER_GPU="${ADAPTER_GPU:-1}"
EVAL_VISIBLE_DEVICES="${EVAL_VISIBLE_DEVICES:-0,1,2,3}"

LOG_DIR="${current_dir}/project/scripts/logs/glm4voice_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
PID_FILE="${LOG_DIR}/server.pids"
: > "${PID_FILE}"

stage="${stage:-1}"
stop_stage="${stop_stage:-3}"
WORKERS="${WORKERS:-4}"

# Force HF/datasets offline -- the deployment has no internet access, and
# online retries cost ~80s per dataset (see app-2026-05-19_19-55-26.log).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# -----------------------------------------------------------------------------
# Server lifecycle helpers
# -----------------------------------------------------------------------------
MODEL_SERVER_PID=""
MODEL_SERVER_PGID=""
ADAPTER_PID=""
ADAPTER_PGID=""
CLEANED_UP=0

kill_pgid() {
    local pgid="$1"
    local name="$2"
    [ -z "${pgid}" ] && return 0
    if kill -0 "-${pgid}" 2>/dev/null; then
        echo "[cleanup] SIGTERM process group ${pgid} (${name})"
        kill -TERM -- "-${pgid}" 2>/dev/null
        # Give the children a couple of seconds to exit gracefully.
        for _ in 1 2 3 4 5; do
            kill -0 "-${pgid}" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "-${pgid}" 2>/dev/null; then
            echo "[cleanup] SIGKILL process group ${pgid} (${name})"
            kill -KILL -- "-${pgid}" 2>/dev/null
        fi
    fi
}

cleanup() {
    [ "${CLEANED_UP}" -eq 1 ] && return 0
    CLEANED_UP=1
    set +e
    echo "[cleanup] tearing down servers..."
    kill_pgid "${ADAPTER_PGID}" "ultraeval_adapter_server"
    kill_pgid "${MODEL_SERVER_PGID}" "model_server"
    # Belt-and-suspenders: also kill anything still listening on our ports.
    for port in "${ADAPTER_PORT}" "${MODEL_SERVER_PORT}"; do
        local pids
        pids=$(ss -ltnp 2>/dev/null | awk -v p=":${port}" '$4 ~ p {print $0}' | \
               grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
        for pid in ${pids}; do
            if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
                echo "[cleanup] SIGKILL leftover pid=${pid} on port ${port}"
                kill -KILL "${pid}" 2>/dev/null
            fi
        done
    done
    rm -f "${PID_FILE}"
}

on_signal() {
    local sig="$1"
    echo "[trap] received ${sig}, cleaning up and exiting..."
    cleanup
    # Use 130 for SIGINT, 143 for SIGTERM (standard convention).
    case "${sig}" in
        INT)  exit 130 ;;
        TERM) exit 143 ;;
        *)    exit 1   ;;
    esac
}

trap 'on_signal INT'  INT
trap 'on_signal TERM' TERM
trap cleanup EXIT

# List the PIDs currently listening on a TCP port.
port_listener_pids() {
    local port="$1"
    ss -ltnp 2>/dev/null \
        | awk -v p=":${port}" '$4 ~ p {print $0}' \
        | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u
}

# Kill any leftover process still bound to the given port.  A previous
# Ctrl+C / kill -9 of the wrapper shell can leave the model_server or
# adapter_server orphaned and bound to 10000/10001 -- in that case our
# fresh server crashes immediately with EADDRINUSE while wait_for_port
# happily connects to the *old* server (this is exactly what happened in
# app-2026-05-21_17-58 run).  So always free the port before starting.
free_port() {
    local port="$1"
    local name="$2"
    local pids
    pids=$(port_listener_pids "${port}")
    if [ -z "${pids}" ]; then
        return 0
    fi
    echo "[free_port] port ${port} (${name}) is occupied by: ${pids}"
    for pid in ${pids}; do
        echo "[free_port] SIGTERM pid=${pid}"
        kill -TERM "${pid}" 2>/dev/null
    done
    # Give them a few seconds to release the port gracefully.
    for _ in 1 2 3 4 5; do
        pids=$(port_listener_pids "${port}")
        [ -z "${pids}" ] && break
        sleep 1
    done
    pids=$(port_listener_pids "${port}")
    for pid in ${pids}; do
        echo "[free_port] SIGKILL pid=${pid}"
        kill -KILL "${pid}" 2>/dev/null
    done
    # Final wait for the kernel to actually release the socket.
    for _ in 1 2 3 4 5; do
        pids=$(port_listener_pids "${port}")
        [ -z "${pids}" ] && return 0
        sleep 1
    done
    if [ -n "$(port_listener_pids "${port}")" ]; then
        echo "[free_port] ERROR: port ${port} still occupied after SIGKILL"
        return 1
    fi
    return 0
}

wait_for_port() {
    local host="$1"
    local port="$2"
    local pid="$3"
    local name="$4"
    local timeout="${5:-1800}"   # default 30 min
    local elapsed=0
    echo "[wait] ${name} at ${host}:${port} (pid=${pid}, timeout=${timeout}s)"
    while :; do
        # If our server is dead -> abort immediately, don't get fooled by an
        # orphan listener from a previous run.
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "[wait] FATAL: ${name} (pid=${pid}) died before listening on ${port}"
            echo "[wait] tail of its log:"
            tail -n 80 "${LOG_DIR}/${name}.log" 2>/dev/null || true
            return 1
        fi
        # Then check whether *our* PID owns the port -- not just whether
        # something is listening on it.
        local listeners
        listeners=$(port_listener_pids "${port}")
        if echo "${listeners}" | grep -qx "${pid}"; then
            echo "[wait] ${name} ready after ${elapsed}s (pid=${pid} owns port)"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        if [ ${elapsed} -ge ${timeout} ]; then
            echo "[wait] ERROR: ${name} not ready after ${timeout}s"
            return 1
        fi
    done
}

start_servers() {
    # Free both ports first -- a previous run may have left an orphan
    # uvicorn worker bound to them, which would make a fresh launch crash
    # with EADDRINUSE while wait_for_port still happily connects to the
    # old listener.
    free_port "${MODEL_SERVER_PORT}" "model_server" || return 1
    free_port "${ADAPTER_PORT}" "ultraeval_adapter_server" || return 1

    # 1. model_server.py -- launched as its own process group leader
    # (thanks to ``set -m`` + ``&``, the backgrounded job gets its own pgid).
    echo "[start] launching model_server.py on cuda:${MODEL_SERVER_GPU}, port ${MODEL_SERVER_PORT}"
    CUDA_VISIBLE_DEVICES="${MODEL_SERVER_GPU}" \
        python -u "${GLM_VOICE_REPO}/model_server.py" \
            --host "${MODEL_SERVER_HOST}" \
            --port "${MODEL_SERVER_PORT}" \
            --device "cuda:0" \
            --dtype "bfloat16" \
            --model-path "${MODEL_DIR}" \
            > "${LOG_DIR}/model_server.log" 2>&1 < /dev/null &
    MODEL_SERVER_PID=$!
    # With ``set -m`` enabled the job is its own group leader -> pgid == pid.
    MODEL_SERVER_PGID="${MODEL_SERVER_PID}"
    MODEL_SERVER_PGID="${MODEL_SERVER_PID}"
    echo "model_server pid=${MODEL_SERVER_PID} pgid=${MODEL_SERVER_PGID}" >> "${PID_FILE}"
    echo "[start] model_server pid=${MODEL_SERVER_PID} pgid=${MODEL_SERVER_PGID}"

    if ! wait_for_port "${MODEL_SERVER_HOST}" "${MODEL_SERVER_PORT}" \
                       "${MODEL_SERVER_PID}" "model_server"; then
        return 1
    fi

    # 2. ultraeval_adapter_server.py -- separate process group as well.
    echo "[start] launching ultraeval_adapter_server.py on cuda:${ADAPTER_GPU}, port ${ADAPTER_PORT}"
    CUDA_VISIBLE_DEVICES="${ADAPTER_GPU}" \
        python -u "${GLM_VOICE_REPO}/ultraeval_adapter_server.py" \
            --host "${ADAPTER_HOST}" \
            --port "${ADAPTER_PORT}" \
            --device "cuda:0" \
            --glm-server-url "http://${MODEL_SERVER_HOST}:${MODEL_SERVER_PORT}/generate_stream" \
            --model-path "${MODEL_DIR}" \
            --tokenizer-path "${TOKENIZER_DIR}" \
            --flow-path "${DECODER_DIR}" \
            > "${LOG_DIR}/ultraeval_adapter_server.log" 2>&1 < /dev/null &
    ADAPTER_PID=$!
    ADAPTER_PGID="${ADAPTER_PID}"
    echo "adapter_server pid=${ADAPTER_PID} pgid=${ADAPTER_PGID}" >> "${PID_FILE}"
    echo "[start] adapter_server pid=${ADAPTER_PID} pgid=${ADAPTER_PGID}"

    if ! wait_for_port "${ADAPTER_HOST}" "${ADAPTER_PORT}" \
                       "${ADAPTER_PID}" "ultraeval_adapter_server"; then
        return 1
    fi
    return 0
}

server_alive() {
    kill -0 "${MODEL_SERVER_PID}" 2>/dev/null && \
    kill -0 "${ADAPTER_PID}" 2>/dev/null
}

run_eval() {
    # Wrap audio_evals/main.py so a single failure does not abort the rest
    # of the evaluation matrix.  If a server died mid-run we abort the
    # whole script (continuing to silently skip every dataset would be
    # worse than failing loudly).
    if ! server_alive; then
        echo "[eval] FATAL: a GLM-4-Voice server died; aborting."
        echo "[eval]   model_server.log tail:"
        tail -n 40 "${LOG_DIR}/model_server.log" 2>/dev/null || true
        echo "[eval]   ultraeval_adapter_server.log tail:"
        tail -n 40 "${LOG_DIR}/ultraeval_adapter_server.log" 2>/dev/null || true
        exit 1
    fi
    CUDA_VISIBLE_DEVICES="${EVAL_VISIBLE_DEVICES}" \
        python audio_evals/main.py "$@" \
        || echo "[eval] FAILED: audio_evals/main.py $*"
}

# Sanity checks before booting servers.
for d in "${GLM_VOICE_REPO}" "${MODEL_DIR}" "${TOKENIZER_DIR}" "${DECODER_DIR}"; do
    if [ ! -d "${d}" ]; then
        echo "[fatal] required path missing: ${d}"
        exit 1
    fi
done

if ! start_servers; then
    echo "[fatal] failed to bring up GLM-4-Voice servers; see ${LOG_DIR}"
    exit 1
fi

# Where to persist S2S waveforms emitted by the client (read by extract_audio).
export GLM4VOICE_AUDIO_OUT_DIR="${current_dir}/res/glm-4-voice/audio_out"
mkdir -p "${GLM4VOICE_AUDIO_OUT_DIR}"

# -----------------------------------------------------------------------------
# Stage 1: ASR
# -----------------------------------------------------------------------------
if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "start stage 1"
    # Chinese ASR datasets
    # for dataset in WenetSpeech-test-net WenetSpeech-test-meeting KeSpeech fleurs-zh; do
    for dataset in KeSpeech fleurs-zh; do
        run_eval --dataset "${dataset}" --model glm-4-voice \
                 --prompt glm-4-voice-asr-zh --workers "${WORKERS}"
    done
    # English ASR datasets
    for dataset in tedlium-release1 fleurs-en_us; do
        run_eval --dataset "${dataset}" --model glm-4-voice \
                 --prompt glm-4-voice-asr-en --workers "${WORKERS}"
    done
fi

# -----------------------------------------------------------------------------
# Stage 2: Speech Translation (S2TT)
# -----------------------------------------------------------------------------
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "start stage 2"
    for dataset in covost2-zh-en; do
        run_eval --dataset "${dataset}" --model glm-4-voice \
                 --prompt glm-4-voice-s2tt-zh2en --workers "${WORKERS}"
    done
    for dataset in covost2-en-zh; do
        run_eval --dataset "${dataset}" --model glm-4-voice \
                 --prompt glm-4-voice-s2tt-en2zh --workers "${WORKERS}"
    done
fi

# -----------------------------------------------------------------------------
# Stage 3: Emotion + Spoken QA
# -----------------------------------------------------------------------------
if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "start stage 3"
    # Emotion Recognition
    for dataset in meld-emo; do
        run_eval --dataset "${dataset}" --model glm-4-voice \
                 --prompt glm-4-voice-emotion --workers "${WORKERS}"
    done
    # Spoken QA datasets (audio-only S2S prompt, downstream extract_audio +
    # speech2text post-processing required).
    for dataset in speech-web-questions speech-triviaqa; do
        run_eval --dataset "${dataset}" --model glm-4-voice \
                 --prompt glm-4-voice-speech-qa --workers "${WORKERS}"
    done
    for dataset in speech-cmmlu; do
        run_eval --dataset "${dataset}" --model glm-4-voice \
                 --prompt glm-4-voice-speech-choice-qa-zh --workers "${WORKERS}"
    done
fi

echo "success on $(date)"
exit 0
