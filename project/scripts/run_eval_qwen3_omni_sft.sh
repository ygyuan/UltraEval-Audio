#!/usr/bin/env bash
set -ex
current_dir=$(pwd)
cd ${current_dir}
#source ${current_dir}/.bashrc
#conda activate GPTSoVits

# ---------------------------------------------------------------------------
# GPU / memory tuning
#   - expandable_segments 缓解 PyTorch caching allocator 的显存碎片，
#     直接对应日志末尾官方给出的建议（见 CUDA OOM 报错）。
#   - CUDA_VISIBLE_DEVICES 允许调用者通过环境变量指定卡，避免误用被其他
#     进程占用的 GPU；默认使用 0 号卡。如需多卡请显式 export。
# ---------------------------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 日志目录，便于事后排查 OOM / 长音频等问题
log_dir="${current_dir}/logs/qwen3_omni_sft"
mkdir -p "${log_dir}"

# 单条 python 评测的最大重试次数。评测本身是断点续跑的（jsonl 追加式写入，
# 已完成 sample 会被跳过），因此 OOM 后重启 python 进程可以从中断点继续。
max_retries="${MAX_RETRIES:-3}"

run_eval() {
    # $1=dataset  $2=model  $3=prompt(可空)
    local dataset="$1"
    local model="$2"
    local prompt="$3"
    local log_file="${log_dir}/${model}__${dataset}.log"
    local attempt=0
    while : ; do
        attempt=$((attempt + 1))
        set +e
        if [ -z "${prompt}" ]; then
            python audio_evals/main.py \
                --dataset "${dataset}" \
                --model "${model}" \
                --use_model_pool off 2>&1 | tee -a "${log_file}"
            rc=${PIPESTATUS[0]}
        else
            python audio_evals/main.py \
                --dataset "${dataset}" \
                --model "${model}" \
                --prompt "${prompt}" \
                --use_model_pool off 2>&1 | tee -a "${log_file}"
            rc=${PIPESTATUS[0]}
        fi
        set -e
        if [ ${rc} -eq 0 ]; then
            echo "[run_eval] ${model} on ${dataset} finished OK (attempt ${attempt})" | tee -a "${log_file}"
            break
        fi
        if [ ${attempt} -ge ${max_retries} ]; then
            echo "[run_eval] ${model} on ${dataset} failed ${attempt} times, giving up" | tee -a "${log_file}"
            break
        fi
        echo "[run_eval] ${model} on ${dataset} failed (rc=${rc}), retrying in 30s ..." | tee -a "${log_file}"
        # 给驱动和其它进程一点时间释放显存
        sleep 30
    done
}

stage=4
stop_stage=4

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    # for dataset in tedlium-release1 fleurs-en_us; do
    for dataset in WenetSpeech-test-net WenetSpeech-test-meeting KeSpeech fleurs-zh; do
    # for dataset in KeSpeech fleurs-zh; do
        for model in qwen3-omni-audio_sft; do
            for prompt in qwen3-omni-asr-zh; do
                run_eval "${dataset}" "${model}" "${prompt}"
            done
        done
    done
    for dataset in tedlium-release1 fleurs-en_us; do
        for model in qwen3-omni-audio_sft; do
            for prompt in qwen3-omni-asr-zh; do
                run_eval "${dataset}" "${model}" "${prompt}"
            done
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in covost2-zh-en; do
        for model in qwen3-omni-audio_sft; do
            for prompt in qwen3-omni-s2tt-zh2en; do
                [ ! -d res/${model}/${dataset} ] && \
                    run_eval "${dataset}" "${model}" "${prompt}"
            done
        done
    done
    for dataset in covost2-en-zh; do
        for model in qwen3-omni-audio_sft; do
            for prompt in qwen3-omni-s2tt-en2zh; do
                [ ! -d res/${model}/${dataset} ] && \
                    run_eval "${dataset}" "${model}" "${prompt}"
            done
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in meld-emo; do
        for model in qwen3-omni-audio_sft; do
            for prompt in qwen3-omni-emotion; do
                [ ! -d res/${model}/${dataset} ] && \
                    run_eval "${dataset}" "${model}" "${prompt}"
            done
        done
    done
    for dataset in speech-web-questions speech-triviaqa speech-chatbot-alpaca-eval; do
        for model in qwen3-omni-speech; do
            for prompt in ""; do
                run_eval "${dataset}" "${model}" ""
            done
        done
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    # 恢复 `[ ! -d res/${model}/${dataset} ]` 断点保护：这样重跑时已完成
    # 的数据集不会重新推理。**注意**：如果历史目录里都是失败结果（fail_rate
    # 100%），请先手动删除对应 res 目录再重跑。
    for dataset in asc-moan; do
        for model in qwen3-omni-audio_sft; do
            for prompt in qwen3-omni-asc-moan; do
                [ ! -d res/${model}/${dataset} ] && \
                    run_eval "${dataset}" "${model}" "${prompt}"
            done
        done
    done
    for dataset in asc-multi; do
        for model in qwen3-omni-audio_sft; do
            for prompt in qwen3-omni-asc-multi; do
                [ ! -d res/${model}/${dataset} ] && \
                    run_eval "${dataset}" "${model}" "${prompt}"
            done
        done
    done
    for dataset in meld-emo; do
        for model in qwen3-omni-audio_sft; do
            for prompt in qwen3-omni-emotion; do
                [ ! -d res/${model}/${dataset} ] && \
                    run_eval "${dataset}" "${model}" "${prompt}"
            done
        done
    done
fi

echo "success on `date`"
exit 0
