# fish-speech Evaluation Results

**Model**: `fishspeech` ([config](../registry/model/fishspeech.yaml))
**Evaluation Date**: 2025/12 (CV3 / minimax_tts updated 2026/06/12)
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

## CV3 Benchmark (Zero-Shot)

| task | dataset | WER/CER⬇️ | SIM⬆️ | P808_MOS⬆️ | eval_cli | note |
|------|---------|-----------|-------|------------|----------|------|
| tts | cv3_zero_shot_en | 2.71 | 61.67 | 3.68 | [3] | |
| tts | cv3_zero_shot_zh | 3.45 | 67.84 | 3.77 | [4] | |
| tts | cv3_zero_shot_hard_en | 3.41 | 60.94 | 3.81 | [5] | |
| tts | cv3_zero_shot_hard_zh | 10.62 | 65.64 | 3.78 | [6] | |

---

## MiniMax TTS Multilingual Benchmark

| task | dataset | WER/CER⬇️ | SIM-O⬆️ | eval_cli | note |
|------|---------|-----------|---------|----------|------|
| tts | minimax_tts_english | 0.69 (WER) | 78.74 | [7] | |
| tts | minimax_tts_chinese | 1.03 (CER) | 76.95 | [8] | |

---

## Evaluation Commands

[1] `python audio_evals/main.py --dataset seed_tts_eval_en --model fishspeech`
[2] `python audio_evals/main.py --dataset seed_tts_eval_zh --model fishspeech`

[3] `python audio_evals/main.py --dataset cv3_zero_shot_en --model fishspeech`
[4] `python audio_evals/main.py --dataset cv3_zero_shot_zh --model fishspeech`
[5] `python audio_evals/main.py --dataset cv3_zero_shot_hard_en --model fishspeech`
[6] `python audio_evals/main.py --dataset cv3_zero_shot_hard_zh --model fishspeech`

[7] `python audio_evals/main.py --dataset minimax_tts_english --model fishspeech`
[8] `python audio_evals/main.py --dataset minimax_tts_chinese --model fishspeech`
