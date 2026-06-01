# OmniVoice Evaluation Results

**Model**: `omnivoice` ([config](../registry/model/omnivoice.yaml))
**Evaluation Date**: 2026/04

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

## Evaluation Commands

[1] `python audio_evals/main.py --dataset seed_tts_eval_en --model omnivoice`
[2] `python audio_evals/main.py --dataset seed_tts_eval_zh --model omnivoice`
