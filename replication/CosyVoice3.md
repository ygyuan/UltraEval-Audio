# CosyVoice3 Evaluation Results

**Model**: `cosyvoice3-latest` ([config](../registry/model/cosyvoice.yaml))
**Evaluation Date**: 2025/12/17 (CV3 / minimax_tts updated 2026/06/12)
**Paper/Repo**: [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)

**Metrics Legend**:
- **WER⬇️**: Word Error Rate (lower is better)
- **CER⬇️**: Character Error Rate (lower is better)
- **SIM⬆️**: Speaker Similarity (higher is better)
- **P808_MOS⬆️**: DNSMOS P.808 Mean Opinion Score (higher is better)

---

**Note**:
Update (2025/12/17): results may change as [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice.git) is updated.


## Seed-TTS-Eval Benchmark

| task | dataset | WER/CER⬇️ | SIM⬆️ | eval_cli | note |
|------|---------|-----------|-------|----------|------|
| tts | seed_tts_eval_en | 1.59(2.24) | 69.48(71.8) | [1] | |
| tts | seed_tts_eval_zh | 1.11(1.21) | 77.57(78.0) | [2] | |

---

## CV3 Benchmark (Zero-Shot)

| task | dataset | WER/CER⬇️ | SIM⬆️ | P808_MOS⬆️ | eval_cli | note |
|------|---------|-----------|-------|------------|----------|------|
| tts | cv3_zero_shot_en | 3.56 | 74.41 | 3.82 | [3] | |
| tts | cv3_zero_shot_zh | 4.05 | 79.96 | 3.86 | [4] | |
| tts | cv3_zero_shot_hard_en | 6.64 | 74.69 | 3.93 | [5] | |
| tts | cv3_zero_shot_hard_zh | 7.47 | 78.01 | 3.80 | [6] | |

---

## MiniMax TTS Multilingual Benchmark

| task | dataset | WER/CER⬇️ | SIM-O⬆️ | eval_cli | note |
|------|---------|-----------|---------|----------|------|
| tts | minimax_tts_english | 2.53 (WER) | 83.35 | [11] | |
| tts | minimax_tts_chinese | 1.53 (CER) | 80.68 | [12] | |

---

## Long-TTS-Eval Benchmark

| task | dataset | WER(%)⬇️ | eval_cli | note |
|------|---------|----------|----------|------|
| tts | long_tts_eval_en | 92.63 | [7] | |
| tts | long_tts_eval_zh | 87.17 | [8] | |
| tts | long_tts_eval_hard_en | 78.20 | [9] | |
| tts | long_tts_eval_hard_zh | 69.59 | [10] | |

---

## Long-TTS-Eval Benchmark (zero_shot-overall.json)

| task | dataset | WER(%)⬇️ | note |
|------|---------|----------|------|
| tts | long_tts_eval_en | - | error: list index out of range (see `res/cosyvoice3-latest/long_tts_eval_en/zero_shot-log.txt`) |
| tts | long_tts_eval_zh | - | error: list index out of range (see `res/cosyvoice3-latest/long_tts_eval_zh/zero_shot-log.txt`) |
| tts | long_tts_eval_hard_en | - | error: list index out of range (see `res/cosyvoice3-latest/long_tts_eval_hard_en/zero_shot-log.txt`) |
| tts | long_tts_eval_hard_zh | - | error: list index out of range (see `res/cosyvoice3-latest/long_tts_eval_hard_zh/zero_shot-log.txt`) |

## Evaluation Commands

[1] `python audio_evals/main.py --dataset seed_tts_eval_en --model cosyvoice3-latest --prompt cosyvoice3-vc`
[2] `python audio_evals/main.py --dataset seed_tts_eval_zh --model cosyvoice3-latest --prompt cosyvoice3-vc`

[3] `python audio_evals/main.py --dataset cv3_zero_shot_en --model cosyvoice3-latest --prompt cosyvoice3-vc`
[4] `python audio_evals/main.py --dataset cv3_zero_shot_zh --model cosyvoice3-latest --prompt cosyvoice3-vc`
[5] `python audio_evals/main.py --dataset cv3_zero_shot_hard_en --model cosyvoice3-latest --prompt cosyvoice3-vc`
[6] `python audio_evals/main.py --dataset cv3_zero_shot_hard_zh --model cosyvoice3-latest --prompt cosyvoice3-vc`

[7] `python audio_evals/main.py --dataset long_tts_eval_en --model cosyvoice3-latest --prompt cosyvoice3-tts`
[8] `python audio_evals/main.py --dataset long_tts_eval_zh --model cosyvoice3-latest --prompt cosyvoice3-tts`
[9] `python audio_evals/main.py --dataset long_tts_eval_hard_en --model cosyvoice3-latest --prompt cosyvoice3-tts`
[10] `python audio_evals/main.py --dataset long_tts_eval_hard_zh --model cosyvoice3-latest --prompt cosyvoice3-tts`

[11] `python audio_evals/main.py --dataset minimax_tts_english --model cosyvoice3-latest --prompt cosyvoice3-vc`
[12] `python audio_evals/main.py --dataset minimax_tts_chinese --model cosyvoice3-latest --prompt cosyvoice3-vc`
