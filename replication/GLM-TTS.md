# GLM-TTS Evaluation Results

**Model**: `glmtts` ([config](../registry/model/glmtts.yaml))
**Evaluation Date**: 2025/12 (CV3 / minimax_tts updated 2026/06/12)
**Paper/Repo**: [zai-org/GLM-TTS](https://github.com/zai-org/GLM-TTS)

**Metrics Legend**:
- **WER⬇️**: Word Error Rate (lower is better)
- **CER⬇️**: Character Error Rate (lower is better)
- **SIM⬆️**: Speaker Similarity (higher is better)
- **P808_MOS⬆️**: DNSMOS P.808 Mean Opinion Score (higher is better)

---


## Seed-TTS-Eval Benchmark

| task | dataset | WER/CER⬇️ | SIM⬆️ | eval_cli | note |
|------|---------|-----------|-------|----------|------|
| tts | seed_tts_eval_en | 2.12 | 67.27 | [1] | |
| tts | seed_tts_eval_zh | 1.08(1.03) | 76.00(76.1) | [2] | |

---

## CV3 Benchmark (Zero-Shot)

| task | dataset | WER/CER⬇️ | SIM⬆️ | P808_MOS⬆️ | eval_cli | note |
|------|---------|-----------|-------|------------|----------|------|
| tts | cv3_zero_shot_en | 4.61 | 72.02 | 3.69 | [3] | |
| tts | cv3_zero_shot_zh | 3.61 | 77.92 | 3.71 | [4] | |
| tts | cv3_zero_shot_hard_en | 4.52 | 73.61 | 3.71 | [5] | fail_rate: 3.12% |
| tts | cv3_zero_shot_hard_zh | 9.29 | 77.75 | 3.61 | [6] | |

---

## MiniMax TTS Multilingual Benchmark

| task | dataset | WER/CER⬇️ | SIM-O⬆️ | eval_cli | note |
|------|---------|-----------|---------|----------|------|
| tts | minimax_tts_english | 0.82 (WER) | 79.84 | [7] | |
| tts | minimax_tts_chinese | 0.96 (CER) | 78.88 | [8] | |

---

## Evaluation Commands

[1] `python audio_evals/main.py --dataset seed_tts_eval_en --model glmtts`
[2] `python audio_evals/main.py --dataset seed_tts_eval_zh --model glmtts`

[3] `python audio_evals/main.py --dataset cv3_zero_shot_en --model glmtts`
[4] `python audio_evals/main.py --dataset cv3_zero_shot_zh --model glmtts`
[5] `python audio_evals/main.py --dataset cv3_zero_shot_hard_en --model glmtts`
[6] `python audio_evals/main.py --dataset cv3_zero_shot_hard_zh --model glmtts`

[7] `python audio_evals/main.py --dataset minimax_tts_english --model glmtts`
[8] `python audio_evals/main.py --dataset minimax_tts_chinese --model glmtts`
