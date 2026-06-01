# fish-speech Evaluation Results

**Model**: `fishspeech` ([config](../registry/model/fishspeech.yaml))
**Evaluation Date**: 2025/12 (from `res/fishspeech/*/res-overall.json`)
**Paper/Repo**: [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)

**Metrics Legend**:
- **WER⬇️**: Word Error Rate (lower is better)
- **CER⬇️**: Character Error Rate (lower is better)
- **SIM⬆️**: Speaker Similarity (higher is better)

---


## Seed-TTS-Eval Benchmark

| task | dataset | WER/CER⬇️ | SIM⬆️ | eval_cli | note |
|------|---------|-----------|-------|----------|------|
| tts | seed_tts_eval_en | 1.12(0.99) | 65.15 | [1] | |
| tts | seed_tts_eval_zh | 0.97(0.54) | 73.28 | [2] | |

---



## Evaluation Commands

[1] `python audio_evals/main.py --dataset seed_tts_eval_en --model fishspeech`
[2] `python audio_evals/main.py --dataset seed_tts_eval_zh --model fishspeech`
