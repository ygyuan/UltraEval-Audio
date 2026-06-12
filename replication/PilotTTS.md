# PilotTTS Evaluation Results

**Model**: `pilot_tts` ([config](../registry/model/pilot_tts.yaml))
**Evaluation Date**: 2026/06 (CV3 / minimax_tts updated 2026/06/12)

**Metrics Legend**:
- **WER⬇️**: Word Error Rate (lower is better)
- **CER⬇️**: Character Error Rate (lower is better)
- **SIM⬆️**: Speaker Similarity (higher is better)

---

## Seed-TTS-Eval Benchmark

| task | dataset | WER/CER⬇️ | SIM⬆️ | eval_cli | note |
|------|---------|-----------|-------|----------|------|
| tts | seed_tts_eval_en | 1.26 (WER) | 68.71 | [1] | |
| tts | seed_tts_eval_zh | 0.89 (CER) | 77.31 | [2] | |

---

## CV3 Benchmark (Zero-Shot)

| task | dataset | WER/CER⬇️ | SIM⬆️ | P808_MOS⬆️ | eval_cli | note |
|------|---------|-----------|-------|------------|----------|------|
| tts | cv3_zero_shot_en | 3.18 | 73.29 | 3.87 | [3] | |
| tts | cv3_zero_shot_zh | 3.36 | 79.57 | 3.87 | [4] | |
| tts | cv3_zero_shot_hard_en | 5.38 | 72.19 | 3.97 | [5] | |
| tts | cv3_zero_shot_hard_zh | 10.35 | 76.71 | 3.81 | [6] | |

---

## MiniMax TTS Multilingual Benchmark

| task | dataset | WER/CER⬇️ | SIM-O⬆️ | eval_cli | note |
|------|---------|-----------|---------|----------|------|
| tts | minimax_tts_english | 1.23 (WER) | 81.10 | [7] | |
| tts | minimax_tts_chinese | 1.22 (CER) | 80.99 | [8] | |

---

## Evaluation Commands

[1] `python audio_evals/main.py --dataset seed_tts_eval_en --model pilot_tts`
[2] `python audio_evals/main.py --dataset seed_tts_eval_zh --model pilot_tts`

[3] `python audio_evals/main.py --dataset cv3_zero_shot_en --model pilot_tts`
[4] `python audio_evals/main.py --dataset cv3_zero_shot_zh --model pilot_tts`
[5] `python audio_evals/main.py --dataset cv3_zero_shot_hard_en --model pilot_tts`
[6] `python audio_evals/main.py --dataset cv3_zero_shot_hard_zh --model pilot_tts`

[7] `python audio_evals/main.py --dataset minimax_tts_english --model pilot_tts`
[8] `python audio_evals/main.py --dataset minimax_tts_chinese --model pilot_tts`
