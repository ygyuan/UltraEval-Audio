# IndexTTS Evaluation Results

**Model**: [IndexTTS2](../registry/model/indextts.yaml)
**Evaluation Date**: 2025/12/08 (CV3 / minimax_tts updated 2026/06/12)
**Paper/Repo**: [IndexTeam/IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2)

**Metrics Legend**:
- **WER⬇️**: Word Error Rate (lower is better)
- **CER⬇️**: Character Error Rate (lower is better)
- **SIM⬆️**: Speaker Similarity (higher is better)

---

## Seed-TTS-Eval Benchmark

### IndexTTS2

| task | dataset | WER/CER⬇️ | SIM⬆️ | eval_cli | note |
|------|---------|-----------|-------|----------|------|
| tts | seed_tts_eval_en | 1.38(1.52) | 70.32 | [3] | |
| tts | seed_tts_eval_zh | 1.04(1.01)| 76.09 | [4] | |

---

## CV3 Benchmark (Zero-Shot)

### IndexTTS2

| task | dataset | WER/CER⬇️ | SIM⬆️ | P808_MOS⬆️ | eval_cli | note |
|------|---------|-----------|-------|------------|----------|------|
| tts | cv3_zero_shot_en | 2.90 | 73.82 | 3.69 | [9] | |
| tts | cv3_zero_shot_zh | 3.54 | 77.82 | 3.75 | [10] | |
| tts | cv3_zero_shot_hard_en | 2.54 | 73.61 | 3.82 | [11] | |
| tts | cv3_zero_shot_hard_zh | 8.24 | 76.80 | 3.71 | [12] | |

---

## MiniMax TTS Multilingual Benchmark

### IndexTTS2

| task | dataset | WER/CER⬇️ | SIM-O⬆️ | eval_cli | note |
|------|---------|-----------|---------|----------|------|
| tts | minimax_tts_english | 0.83 (WER) | 83.04 | [13] | |
| tts | minimax_tts_chinese | 0.94 (CER) | 80.30 | [14] | |

---

## Evaluation Commands

### IndexTTS2

[3] `python audio_evals/main.py --dataset seed_tts_eval_en --model indextts2`

[4] `python audio_evals/main.py --dataset seed_tts_eval_zh --model indextts2`

[9] `python audio_evals/main.py --dataset cv3_zero_shot_en --model indextts2`

[10] `python audio_evals/main.py --dataset cv3_zero_shot_zh --model indextts2`

[11] `python audio_evals/main.py --dataset cv3_zero_shot_hard_en --model indextts2`

[12] `python audio_evals/main.py --dataset cv3_zero_shot_hard_zh --model indextts2`

[13] `python audio_evals/main.py --dataset minimax_tts_english --model indextts2`

[14] `python audio_evals/main.py --dataset minimax_tts_chinese --model indextts2`
