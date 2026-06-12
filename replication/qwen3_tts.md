# Qwen3-TTS 复现文档与评测结果

**模型**: [Qwen3-TTS](../registry/model/qwen3tts.yaml)  
**评测日期**: 2026/02（CV3 / minimax_tts 更新于 2026/06/12）

**指标说明**:
- **WER⬇️ / CER⬇️**: ASR 识别错误率（越低越好）
- **SIM⬆️**: 说话人相似度（越高越好）
- **DNSMOS⬆️**: 语音质量打分（越高越好，范围 0–5）

---

## Seed-TTS-Eval（Voice Clone）复现结果

**Note**: 性能格式为 `reproduced_result(official_result)`，括号内为论文/官方结果（如有）。

> 下面命令会自动下载权重到 `init_model/` 并运行 voice clone；首次运行会较慢。

| 模型 | SEED-test-en (WER⬇️) | SEED-test-en (SIM⬆️) | SEED-test-zh (CER⬇️) | SEED-test-zh (SIM⬆️) | eval_cli |
|---|---:|---:|---:|---:|---|
| Qwen3-TTS-12Hz-1.7B-Base-official-infer-params | 1.58 (1.24) | 71.24 | 0.87 (0.78) | 76.89 | en:[1] zh:[2] |
| Qwen3-TTS-12Hz-1.7B-Base-official-infer-params-xvec_only | 1.56 (1.24) | 59.61 | 0.78 (0.78) | 72.92 | en:[3] zh:[4] |
| Qwen3-TTS-12Hz-0.6B-Base-official-infer-params | 1.69 (1.32) | 70.55 | 1.01 (0.92) | 76.48 | en:[5] zh:[6] |

## CV3-Eval（Zero-shot Voice Clone）复现结果

> CV3-Eval 在本项目中按 split 分开跑（`cv3_zero_shot_{en,zh}` 与 `cv3_zero_shot_hard_{en,zh}`），下方汇总最新一次评测结果。

| 模型 | en WER/%⬇️ | en SIM/%⬆️ | en P808⬆️ | zh CER/%⬇️ | zh SIM/%⬆️ | zh P808⬆️ | hard-en WER/%⬇️ | hard-en SIM/%⬆️ | hard-en P808⬆️ | hard-zh CER/%⬇️ | hard-zh SIM/%⬆️ | hard-zh P808⬆️ | eval_cli |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-TTS-12Hz-1.7B-Base-official-infer-params | 3.84 | 67.31 | 3.75 | 3.05 | 72.86 | 3.83 | 5.38 | 66.82 | 3.90 | 12.33 | 69.19 | 3.80 | en:[7] zh:[8] hard-en:[9] hard-zh:[10] |
| Qwen3-TTS-12Hz-1.7B-Base-official-infer-params-xvec_only | 2.69 | 61.21 | 3.79 | 2.96 | 69.90 | 3.82 | 2.11 | 59.99 | 3.91 | 8.39 | 66.46 | 3.78 | en:[15] zh:[16] hard-en:[17] hard-zh:[18] |
| Qwen3-TTS-12Hz-0.6B-Base-official-infer-params | 4.01 | 66.99 | 3.69 | 3.50 | 72.33 | 3.82 | 5.63 | 67.17 | 3.84 | 12.32 | 66.88 | 3.77 | en:[11] zh:[12] hard-en:[13] hard-zh:[14] |

---

## MiniMax TTS Multilingual Benchmark 复现结果

| 模型 | en WER/%⬇️ | en SIM-O⬆️ | zh CER/%⬇️ | zh SIM-O⬆️ | eval_cli |
|---|---:|---:|---:|---:|---|
| Qwen3-TTS-12Hz-1.7B-Base-official-infer-params | 0.76 | 78.98 | 1.10 | 81.12 | en:[19] zh:[20] |
| Qwen3-TTS-12Hz-1.7B-Base-official-infer-params-xvec_only | 0.68 | 66.16 | 0.74 | 76.12 | en:[21] zh:[22] |
| Qwen3-TTS-12Hz-0.6B-Base-official-infer-params | 0.66 | 81.62 | 1.64 | 81.27 | en:[23] zh:[24] |

---

## Evaluation Commands

[1] `python audio_evals/main.py --dataset seed_tts_eval_en --model qwen3-tts-1.7b-base --prompt qwen3-tts-voice-clone-english --use_model_pool --workers 8`  
[2] `python audio_evals/main.py --dataset seed_tts_eval_zh --model qwen3-tts-1.7b-base --prompt qwen3-tts-voice-clone-chinese --use_model_pool --workers 8`  
[3] `python audio_evals/main.py --dataset seed_tts_eval_en --model qwen3-tts-12hz-1.7b-base-xvec_only --prompt qwen3-tts-voice-clone-english --use_model_pool --workers 8`  
[4] `python audio_evals/main.py --dataset seed_tts_eval_zh --model qwen3-tts-12hz-1.7b-base-xvec_only --prompt qwen3-tts-voice-clone-chinese --use_model_pool --workers 8`  
[5] `python audio_evals/main.py --dataset seed_tts_eval_en --model qwen3-tts-0.6b-base --prompt qwen3-tts-voice-clone-english --use_model_pool --workers 8`  
[6] `python audio_evals/main.py --dataset seed_tts_eval_zh --model qwen3-tts-0.6b-base --prompt qwen3-tts-voice-clone-chinese --use_model_pool --workers 8`  

[7] `python audio_evals/main.py --dataset cv3_zero_shot_en --model qwen3-tts-1.7b-base --prompt qwen3-tts-voice-clone-english --use_model_pool --workers 8`  
[8] `python audio_evals/main.py --dataset cv3_zero_shot_zh --model qwen3-tts-1.7b-base --prompt qwen3-tts-voice-clone-chinese --use_model_pool --workers 8`  
[9] `python audio_evals/main.py --dataset cv3_zero_shot_hard_en --model qwen3-tts-1.7b-base --prompt qwen3-tts-voice-clone-english --use_model_pool --workers 8`  
[10] `python audio_evals/main.py --dataset cv3_zero_shot_hard_zh --model qwen3-tts-1.7b-base --prompt qwen3-tts-voice-clone-chinese --use_model_pool --workers 8`  

[11] `python audio_evals/main.py --dataset cv3_zero_shot_en --model qwen3-tts-0.6b-base --prompt qwen3-tts-voice-clone-english --use_model_pool --workers 8`  
[12] `python audio_evals/main.py --dataset cv3_zero_shot_zh --model qwen3-tts-0.6b-base --prompt qwen3-tts-voice-clone-chinese --use_model_pool --workers 8`  
[13] `python audio_evals/main.py --dataset cv3_zero_shot_hard_en --model qwen3-tts-0.6b-base --prompt qwen3-tts-voice-clone-english --use_model_pool --workers 8`  
[14] `python audio_evals/main.py --dataset cv3_zero_shot_hard_zh --model qwen3-tts-0.6b-base --prompt qwen3-tts-voice-clone-chinese --use_model_pool --workers 8`  

[15] `python audio_evals/main.py --dataset cv3_zero_shot_en --model qwen3-tts-12hz-1.7b-base-xvec_only --prompt qwen3-tts-voice-clone-english --use_model_pool --workers 8`  
[16] `python audio_evals/main.py --dataset cv3_zero_shot_zh --model qwen3-tts-12hz-1.7b-base-xvec_only --prompt qwen3-tts-voice-clone-chinese --use_model_pool --workers 8`  
[17] `python audio_evals/main.py --dataset cv3_zero_shot_hard_en --model qwen3-tts-12hz-1.7b-base-xvec_only --prompt qwen3-tts-voice-clone-english --use_model_pool --workers 8`  
[18] `python audio_evals/main.py --dataset cv3_zero_shot_hard_zh --model qwen3-tts-12hz-1.7b-base-xvec_only --prompt qwen3-tts-voice-clone-chinese --use_model_pool --workers 8`  

[19] `python audio_evals/main.py --dataset minimax_tts_english --model qwen3-tts-1.7b-base --prompt qwen3-tts-voice-clone-english --use_model_pool --workers 8`  
[20] `python audio_evals/main.py --dataset minimax_tts_chinese --model qwen3-tts-1.7b-base --prompt qwen3-tts-voice-clone-chinese --use_model_pool --workers 8`  
[21] `python audio_evals/main.py --dataset minimax_tts_english --model qwen3-tts-12hz-1.7b-base-xvec_only --prompt qwen3-tts-voice-clone-english --use_model_pool --workers 8`  
[22] `python audio_evals/main.py --dataset minimax_tts_chinese --model qwen3-tts-12hz-1.7b-base-xvec_only --prompt qwen3-tts-voice-clone-chinese --use_model_pool --workers 8`  
[23] `python audio_evals/main.py --dataset minimax_tts_english --model qwen3-tts-0.6b-base --prompt qwen3-tts-voice-clone-english --use_model_pool --workers 8`  
[24] `python audio_evals/main.py --dataset minimax_tts_chinese --model qwen3-tts-0.6b-base --prompt qwen3-tts-voice-clone-chinese --use_model_pool --workers 8`  
