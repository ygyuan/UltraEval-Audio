set -exo
current_dir=$(pwd)
cd ${current_dir}
#source ${current_dir}/.bashrc
#conda activate GPTSoVits

stage=3
stop_stage=3

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in seed_tts_eval_en; do
        for model in qwen3-tts-1.7b-base qwen3-tts-12hz-1.7b-base-xvec_only qwen3-tts-0.6b-base; do
            for prompt in qwen3-tts-voice-clone-english; do
                [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 8
            done
        done
    done
    for dataset in seed_tts_eval_zh; do
        for model in qwen3-tts-1.7b-base qwen3-tts-12hz-1.7b-base-xvec_only qwen3-tts-0.6b-base; do
            for prompt in qwen3-tts-voice-clone-chinese; do
                [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 8
            done
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for dataset in cv3_zero_shot_en; do
        for model in qwen3-tts-1.7b-base qwen3-tts-12hz-1.7b-base-xvec_only qwen3-tts-0.6b-base; do
            for prompt in qwen3-tts-voice-clone-english; do
                [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 8
            done
        done
    done
    for dataset in cv3_zero_shot_zh; do
        for model in qwen3-tts-1.7b-base qwen3-tts-12hz-1.7b-base-xvec_only qwen3-tts-0.6b-base; do
            for prompt in qwen3-tts-voice-clone-chinese; do
                [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 8
            done
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "start stage ${stage}, stop stage ${stage}"
    for dataset in cv3_zero_shot_hard_en; do
        for model in qwen3-tts-1.7b-base qwen3-tts-12hz-1.7b-base-xvec_only qwen3-tts-0.6b-base; do
            for prompt in qwen3-tts-voice-clone-english; do
                [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 8
            done
        done
    done
    for dataset in cv3_zero_shot_hard_zh; do
        for model in qwen3-tts-1.7b-base qwen3-tts-12hz-1.7b-base-xvec_only qwen3-tts-0.6b-base; do
            for prompt in qwen3-tts-voice-clone-chinese; do
                [ ! -d res/${model}/${dataset} ] && \
                    python audio_evals/main.py --dataset ${dataset} --model ${model} --prompt ${prompt} --use_model_pool --workers 8
            done
        done
    done
fi

echo "success on `date`"
exit 0
