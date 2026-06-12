# VoxCPM2 Evaluation Results

**Model**: [VoxCPM2](../registry/model/voxcmp2.yaml)
**Evaluation Date**: 2026/04/10 (CV3 / minimax_tts updated 2026/06/12)
**Paper/Repo**: [openbmb/VoxCPM](https://github.com/OpenBMB/VoxCPM.git)

**Metrics Legend**:
- **WER⬇️**: Word Error Rate (lower is better)
- **CER⬇️**: Character Error Rate (lower is better)
- **SIM⬆️**: Speaker Similarity (higher is better)
- **P808_MOS⬆️**: DNSMOS P.808 Mean Opinion Score (higher is better)

---

## Seed-TTS-Eval Benchmark

| task | dataset | WER/CER⬇️ | SIM⬆️ | eval_cli | note |
|------|---------|-----------|-------|----------|------|
| tts | seed_tts_eval_en | 1.15(1.84) | 54.59(75.3) | [1] | |
| tts | seed_tts_eval_zh | 0.99(0.97) | 61.90(79.5) | [2] | |

---

## CV3 Benchmark (Zero-Shot)

| task | dataset | WER/CER⬇️ | SIM⬆️ | P808_MOS⬆️ | eval_cli | note |
|------|---------|-----------|-------|------------|----------|------|
| tts | cv3_zero_shot_en | 4.37 (5.00) | 71.92 | 3.65 | [3] | |
| tts | cv3_zero_shot_zh | 3.70 (3.65) | 75.37 | 3.74 | [4] | |
| tts | cv3_zero_shot_hard_en | 3.78 (8.48) | 69.55 | 3.75 | [5] | |
| tts | cv3_zero_shot_hard_zh | 8.17 (8.55) | 72.65 | 3.64 | [6] | |

---

## MiniMax TTS Multilingual Benchmark

| task | dataset | WER/CER⬇️ | SIM-O⬆️ | eval_cli | note |
|------|---------|-----------|---------|----------|------|
| tts | minimax_tts_english | 0.92 (WER) | 86.01 | [7] | |
| tts | minimax_tts_chinese | 1.14 (CER) | 82.23 | [8] | |

---

## Evaluation Commands

[1] `python audio_evals/main.py --dataset seed_tts_eval_en --model voxcpm2-vc --use_model_pool --workers 8`

[2] `python audio_evals/main.py --dataset seed_tts_eval_zh --model voxcpm2-vc --use_model_pool --workers 8`

[3] `python audio_evals/main.py --dataset cv3_zero_shot_en --model voxcpm2-vc --use_model_pool --workers 8`

[4] `python audio_evals/main.py --dataset cv3_zero_shot_zh --model voxcpm2-vc --use_model_pool --workers 8`

[5] `python audio_evals/main.py --dataset cv3_zero_shot_hard_en --model voxcpm2-vc --use_model_pool --workers 8`

[6] `python audio_evals/main.py --dataset cv3_zero_shot_hard_zh --model voxcpm2-vc --use_model_pool --workers 8`

[7] `python audio_evals/main.py --dataset minimax_tts_english --model voxcpm2-vc --use_model_pool --workers 8`

[8] `python audio_evals/main.py --dataset minimax_tts_chinese --model voxcpm2-vc --use_model_pool --workers 8`
