# OmniVoice Evaluation Results

**Model**: `omnivoice` ([config](../registry/model/omnivoice.yaml))
**Evaluation Date**: 2026/04 (CV3 / minimax_tts updated 2026/06/12)

**Metrics Legend**:
- **WER⬇️**: Word Error Rate (lower is better)
- **CER⬇️**: Character Error Rate (lower is better)
- **SIM⬆️**: Speaker Similarity (higher is better)

---

## Seed-TTS-Eval Benchmark

| task | dataset | WER/CER⬇️ | SIM⬆️ | eval_cli | note |
|------|---------|-----------|-------|----------|------|
| tts | seed_tts_eval_en | 0.98 (WER) | 73.60 | [1] | |
| tts | seed_tts_eval_zh | 0.85 (CER) | 76.83 | [2] | |

---

## CV3 Benchmark (Zero-Shot)

| task | dataset | WER/CER⬇️ | SIM⬆️ | P808_MOS⬆️ | eval_cli | note |
|------|---------|-----------|-------|------------|----------|------|
| tts | cv3_zero_shot_en | 2.85 | 70.28 | 3.77 | [3] | |
| tts | cv3_zero_shot_zh | 3.33 | 73.18 | 3.86 | [4] | |
| tts | cv3_zero_shot_hard_en | 2.57 | 71.03 | 3.83 | [5] | |
| tts | cv3_zero_shot_hard_zh | 9.03 | 71.64 | 3.81 | [6] | |

---

## MiniMax TTS Multilingual Benchmark

| task | dataset | WER/CER⬇️ | SIM-O⬆️ | eval_cli | note |
|------|---------|-----------|---------|----------|------|
| tts | minimax_tts_english | 1.24 (WER) | 87.06 | [7] | |
| tts | minimax_tts_chinese | 0.98 (CER) | 82.41 | [8] | fail_rate: 97% |

---

## Evaluation Commands

[1] `python audio_evals/main.py --dataset seed_tts_eval_en --model omnivoice`
[2] `python audio_evals/main.py --dataset seed_tts_eval_zh --model omnivoice`

[3] `python audio_evals/main.py --dataset cv3_zero_shot_en --model omnivoice`
[4] `python audio_evals/main.py --dataset cv3_zero_shot_zh --model omnivoice`
[5] `python audio_evals/main.py --dataset cv3_zero_shot_hard_en --model omnivoice`
[6] `python audio_evals/main.py --dataset cv3_zero_shot_hard_zh --model omnivoice`

[7] `python audio_evals/main.py --dataset minimax_tts_english --model omnivoice`
[8] `python audio_evals/main.py --dataset minimax_tts_chinese --model omnivoice`
