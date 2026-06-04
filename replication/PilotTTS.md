# PilotTTS Evaluation Results

**Model**: `pilot_tts` ([config](../registry/model/pilot_tts.yaml))
**Evaluation Date**: 2026/06

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

## Evaluation Commands

[1] `python audio_evals/main.py --dataset seed_tts_eval_en --model pilot_tts`
[2] `python audio_evals/main.py --dataset seed_tts_eval_zh --model pilot_tts`
