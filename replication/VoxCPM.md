# VoxCPM Evaluation Results

**Model**: [VoxCPM](../registry/model/voxcpm.yaml)
**Evaluation Date**: 2025/12/08 (CV3 / minimax_tts updated 2026/06/12)
**Paper/Repo**: [openbmb/VoxCPM](https://huggingface.co/openbmb/VoxCPM-0.5B)

**Metrics Legend**:
- **WER⬇️**: Word Error Rate (lower is better)
- **CER⬇️**: Character Error Rate (lower is better)
- **SIM⬆️**: Speaker Similarity (higher is better)
- **P808_MOS⬆️**: DNSMOS P.808 Mean Opinion Score (higher is better)

---

## Seed-TTS-Eval Benchmark

| task | dataset | WER/CER⬇️ | SIM⬆️ | eval_cli | note |
|------|---------|-----------|-------|----------|------|
| tts | seed_tts_eval_en | 1.53(1.85) | 73.20(72.9) | [1] | |
| tts | seed_tts_eval_zh | 0.99(0.93) | 77.25(77.2) | [2] | |

---

## CV3 Benchmark (Zero-Shot)

| task | dataset | WER/CER⬇️ | SIM⬆️ | P808_MOS⬆️ | eval_cli | note |
|------|---------|-----------|-------|------------|----------|------|
| tts | cv3_zero_shot_en | 6.42 | 67.98 | 3.78 | [3] | |
| tts | cv3_zero_shot_zh | 3.62 | 72.23 | 3.82 | [4] | |
| tts | cv3_zero_shot_hard_en | 4.68 | 65.06 | 3.87 | [5] | |
| tts | cv3_zero_shot_hard_zh | 13.04 | 65.69 | 3.68 | [6] | |

---

## MiniMax TTS Multilingual Benchmark

| task | dataset | WER/CER⬇️ | SIM-O⬆️ | eval_cli | note |
|------|---------|-----------|---------|----------|------|
| tts | minimax_tts_english | 1.01 (WER) | 83.16 | [7] | |
| tts | minimax_tts_chinese | 1.05 (CER) | 83.03 | [8] | |

---

## Evaluation Commands

[1] `python audio_evals/main.py --dataset seed_tts_eval_en --model voxcpm-vc`

[2] `python audio_evals/main.py --dataset seed_tts_eval_zh --model voxcpm-vc`

[3] `python audio_evals/main.py --dataset cv3_zero_shot_en --model voxcpm-vc`

[4] `python audio_evals/main.py --dataset cv3_zero_shot_zh --model voxcpm-vc`

[5] `python audio_evals/main.py --dataset cv3_zero_shot_hard_en --model voxcpm-vc`

[6] `python audio_evals/main.py --dataset cv3_zero_shot_hard_zh --model voxcpm-vc`

[7] `python audio_evals/main.py --dataset minimax_tts_english --model voxcpm-vc`

[8] `python audio_evals/main.py --dataset minimax_tts_chinese --model voxcpm-vc`
