# MiMo-ASR Evaluation Results

**Model**: `mimo-asr-en` / `mimo-asr-zh` ([config](../registry/model/mimo-asr.yaml))
**Evaluation Date**: 2026/05

**Metrics Legend**:
- **WER⬇️**: Word Error Rate (lower is better)
- **CER⬇️**: Character Error Rate (lower is better)

---

## ASR (English) — `mimo-asr-en`

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| asr(en) | tedlium-release1 | wer⬇️ | 2.42 | [1] | |
| asr(en) | fleurs-en_us | wer⬇️ | 2.21 | [2] | |

## ASR (Chinese) — `mimo-asr-zh`

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| asr(zh) | fleurs-zh | cer⬇️ | 1.35 | [3] | |
| asr(zh) | KeSpeech | cer⬇️ | 7.74 | [4] | |
| asr(zh) | WenetSpeech-test-net | cer⬇️ | 5.37 | [5] | |
| asr(zh) | WenetSpeech-test-meeting | cer⬇️ | 5.94 | [6] | |
| asr(zh) | asr_lianghui | cer⬇️ | 5.37 | [7] | fail_rate: 0.04% |
| asr(zh) | asr_qunliao_eyi | cer⬇️ | 7.42 | [8] | |
| asr(zh) | asr_shipinhao_long | cer⬇️ | 3.80 | [9] | |
| asr(zh) | asr_shipinhao_fangyan | cer⬇️ | 33.25 | [10] | dialect |
| asr(zh) | asr_badcase | cer⬇️ | 15.61 | [11] | hard cases |

---

## Evaluation Commands

[1] `python audio_evals/main.py --dataset tedlium-release1 --model mimo-asr-en`
[2] `python audio_evals/main.py --dataset fleurs-en_us --model mimo-asr-en`
[3] `python audio_evals/main.py --dataset fleurs-zh --model mimo-asr-zh`
[4] `python audio_evals/main.py --dataset KeSpeech --model mimo-asr-zh`
[5] `python audio_evals/main.py --dataset WenetSpeech-test-net --model mimo-asr-zh`
[6] `python audio_evals/main.py --dataset WenetSpeech-test-meeting --model mimo-asr-zh`
[7] `python audio_evals/main.py --dataset asr_lianghui --model mimo-asr-zh`
[8] `python audio_evals/main.py --dataset asr_qunliao_eyi --model mimo-asr-zh`
[9] `python audio_evals/main.py --dataset asr_shipinhao_long --model mimo-asr-zh`
[10] `python audio_evals/main.py --dataset asr_shipinhao_fangyan --model mimo-asr-zh`
[11] `python audio_evals/main.py --dataset asr_badcase --model mimo-asr-zh`
