# VibeVoice Evaluation Results

**Model**: `vibevioce_tts` / `vibevoice-asr-en` / `vibevoice-asr-zh` ([config](../registry/model/vibevoice.yaml))
**Evaluation Date**: 2026/05

**Metrics Legend**:
- **WER⬇️**: Word Error Rate (lower is better)
- **CER⬇️**: Character Error Rate (lower is better)
- **SIM⬆️**: Speaker Similarity (higher is better)

---

## TTS (Seed-TTS-Eval Benchmark) — `vibevioce_tts`

| task | dataset | WER/CER⬇️ | SIM⬆️ | eval_cli | note |
|------|---------|-----------|-------|----------|------|
| tts | seed_tts_eval_en | 4.87 (WER) | 59.62 | [1] | |
| tts | seed_tts_eval_zh | 2.58 (CER) | 68.88 | [2] | |

---

## ASR (English) — `vibevoice-asr-en`

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| asr(en) | tedlium-release1 | wer⬇️ | 3.30 | [3] | |
| asr(en) | fleurs-en_us | wer⬇️ | 6.09 | [4] | |

## ASR (Chinese) — `vibevoice-asr-zh`

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| asr(zh) | fleurs-zh | cer⬇️ | 5.05 | [5] | |
| asr(zh) | KeSpeech | cer⬇️ | 30.64 | [6] | |
| asr(zh) | WenetSpeech-test-net | cer⬇️ | 14.75 | [7] | |
| asr(zh) | WenetSpeech-test-meeting | cer⬇️ | 18.43 | [8] | |
| asr(zh) | asr_lianghui | cer⬇️ | 11.00 | [9] | fail_rate: 76.66% |
| asr(zh) | asr_qunliao_eyi | cer⬇️ | 12.93 | [10] | |
| asr(zh) | asr_shipinhao_long | cer⬇️ | 9.71 | [11] | |
| asr(zh) | asr_shipinhao_fangyan | cer⬇️ | 60.24 | [12] | dialect |
| asr(zh) | asr_badcase | cer⬇️ | 27.26 | [13] | hard cases |

---

## Evaluation Commands

[1] `python audio_evals/main.py --dataset seed_tts_eval_en --model vibevioce_tts`
[2] `python audio_evals/main.py --dataset seed_tts_eval_zh --model vibevioce_tts`

[3] `python audio_evals/main.py --dataset tedlium-release1 --model vibevoice-asr-en`
[4] `python audio_evals/main.py --dataset fleurs-en_us --model vibevoice-asr-en`

[5] `python audio_evals/main.py --dataset fleurs-zh --model vibevoice-asr-zh`
[6] `python audio_evals/main.py --dataset KeSpeech --model vibevoice-asr-zh`
[7] `python audio_evals/main.py --dataset WenetSpeech-test-net --model vibevoice-asr-zh`
[8] `python audio_evals/main.py --dataset WenetSpeech-test-meeting --model vibevoice-asr-zh`
[9] `python audio_evals/main.py --dataset asr_lianghui --model vibevoice-asr-zh`
[10] `python audio_evals/main.py --dataset asr_qunliao_eyi --model vibevoice-asr-zh`
[11] `python audio_evals/main.py --dataset asr_shipinhao_long --model vibevoice-asr-zh`
[12] `python audio_evals/main.py --dataset asr_shipinhao_fangyan --model vibevoice-asr-zh`
[13] `python audio_evals/main.py --dataset asr_badcase --model vibevoice-asr-zh`
