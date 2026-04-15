#!/bin/bash
set -exo

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
current_dir=$(cd "${script_dir}/../.." && pwd)
cd ${current_dir}

# ============================================================
# Auto-restart wrapper for model_server and adapter_server
# ============================================================
MODEL_SERVER_SCRIPT="${current_dir}/init_model/zai-org/GLM-4-Voice/model_server.py"
ADAPTER_SERVER_SCRIPT="${current_dir}/init_model/zai-org/GLM-4-Voice/ultraeval_adapter_server.py"
MODEL_SERVER_PORT=${MODEL_SERVER_PORT:-10000}
ADAPTER_SERVER_PORT=${ADAPTER_SERVER_PORT:-10001}
MODEL_SERVER_DEVICE=${MODEL_SERVER_DEVICE:-"cuda:0"}
MODEL_SERVER_DTYPE=${MODEL_SERVER_DTYPE:-"bfloat16"}

MODEL_SERVER_PID=""
ADAPTER_SERVER_PID=""
MODEL_RESTART_LOOP_PID=""
ADAPTER_RESTART_LOOP_PID=""

cleanup() {
    echo "[cleanup] Stopping all background server processes..."
    # Kill the restart-loop shells first, then the actual server processes
    for pid in $MODEL_RESTART_LOOP_PID $ADAPTER_RESTART_LOOP_PID $ADAPTER_SERVER_PID $MODEL_SERVER_PID; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    # Also kill any remaining child processes in the process group
    pkill -P $$ 2>/dev/null || true
    wait 2>/dev/null || true
    echo "[cleanup] Done."
}
trap cleanup EXIT INT TERM

wait_for_port() {
    local port=$1
    local timeout=${2:-300}
    local interval=3
    local elapsed=0
    echo "[wait_for_port] Waiting for port ${port} to be ready (timeout=${timeout}s)..."
    while ! (echo > /dev/tcp/127.0.0.1/${port}) 2>/dev/null; do
        sleep ${interval}
        elapsed=$((elapsed + interval))
        if [ ${elapsed} -ge ${timeout} ]; then
            echo "[wait_for_port] ERROR: port ${port} not ready after ${timeout}s"
            return 1
        fi
    done
    echo "[wait_for_port] Port ${port} is ready (took ~${elapsed}s)."
    return 0
}

# Start model_server with auto-restart loop in background
start_model_server_loop() {
    while true; do
        echo "[model_server] Starting model_server on port ${MODEL_SERVER_PORT} (device=${MODEL_SERVER_DEVICE}, dtype=${MODEL_SERVER_DTYPE})..."
        python "${MODEL_SERVER_SCRIPT}" \
            --host 127.0.0.1 \
            --port ${MODEL_SERVER_PORT} \
            --device ${MODEL_SERVER_DEVICE} \
            --dtype ${MODEL_SERVER_DTYPE}
        exit_code=$?
        echo "[model_server] model_server exited with code ${exit_code}. Restarting in 5s..."
        sleep 5
    done
}

# Start adapter_server with auto-restart loop in background
start_adapter_server_loop() {
    while true; do
        echo "[adapter_server] Starting adapter_server on port ${ADAPTER_SERVER_PORT}..."
        python "${ADAPTER_SERVER_SCRIPT}" \
            --host 127.0.0.1 \
            --port ${ADAPTER_SERVER_PORT} \
            --device ${MODEL_SERVER_DEVICE} \
            --glm-server-url "http://127.0.0.1:${MODEL_SERVER_PORT}/generate_stream"
        exit_code=$?
        echo "[adapter_server] adapter_server exited with code ${exit_code}. Restarting in 5s..."
        sleep 5
    done
}

# Launch both server loops in background
start_model_server_loop &
MODEL_RESTART_LOOP_PID=$!
echo "[main] model_server restart-loop PID: ${MODEL_RESTART_LOOP_PID}"

# Wait for model_server to be ready before starting adapter_server
wait_for_port ${MODEL_SERVER_PORT} 600

start_adapter_server_loop &
ADAPTER_RESTART_LOOP_PID=$!
echo "[main] adapter_server restart-loop PID: ${ADAPTER_RESTART_LOOP_PID}"

# Wait for adapter_server to be ready before running eval
wait_for_port ${ADAPTER_SERVER_PORT} 600

echo "[main] Both servers are ready. Starting evaluation..."

# ============================================================
# Evaluation stages
# ============================================================
stage=1
stop_stage=3

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in KeSpeech fleurs-zh; do
        for model in glm-4-voice; do
            for prompt in glm4voice-asr-zh; do
                python -m audio_evals.main --dataset ${dataset} --model ${model} --prompt ${prompt}
            done
        done
    done

    for dataset in tedlium-release1 fleurs-en_us; do
        for model in glm-4-voice; do
            for prompt in glm4voice-asr-en; do
                python -m audio_evals.main --dataset ${dataset} --model ${model} --prompt ${prompt}
            done
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in covost2-zh-en; do
        for model in glm-4-voice; do
            for prompt in glm4voice-s2tt-zh2en; do
                python -m audio_evals.main --dataset ${dataset} --model ${model} --prompt ${prompt}
            done
        done
    done

    for dataset in covost2-en-zh; do
        for model in glm-4-voice; do
            for prompt in glm4voice-s2tt-en2zh; do
                python -m audio_evals.main --dataset ${dataset} --model ${model} --prompt ${prompt}
            done
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in meld-emo; do
        for model in glm-4-voice; do
            for prompt in glm4voice-emotion; do
                python -m audio_evals.main --dataset ${dataset} --model ${model} --prompt ${prompt}
            done
        done
    done

    for dataset in speech-web-questions speech-triviaqa speech-chatbot-alpaca-eval; do
        for model in glm-4-voice; do
            python -m audio_evals.main --dataset ${dataset} --model ${model}
        done
    done
fi

echo "success on `date`"
exit 0
