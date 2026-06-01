# GLM-ASR Evaluation Results

**Model**: `glm-asr-en` / `glm-asr-zh` ([config](../registry/model/glm-asr.yaml))
**Evaluation Date**: 2026/05

**Metrics Legend**:
- **WER⬇️**: Word Error Rate (lower is better)
- **CER⬇️**: Character Error Rate (lower is better)

---

## ASR (English) — `glm-asr-en`

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| asr(en) | tedlium-release1 | wer⬇️ | 3.10 | [1] | |
| asr(en) | fleurs-en_us | wer⬇️ | 6.00 | [2] | |

## ASR (Chinese) — `glm-asr-zh`

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| asr(zh) | fleurs-zh | cer⬇️ | 3.37 | [3] | |
| asr(zh) | KeSpeech | cer⬇️ | 9.18 | [4] | |
| asr(zh) | WenetSpeech-test-net | cer⬇️ | 6.61 | [5] | |
| asr(zh) | WenetSpeech-test-meeting | cer⬇️ | 8.43 | [6] | |
| asr(zh) | asr_lianghui | cer⬇️ | 6.14 | [7] | fail_rate: 0.04% |
| asr(zh) | asr_qunliao_eyi | cer⬇️ | 9.51 | [8] | |
| asr(zh) | asr_shipinhao_long | cer⬇️ | 4.34 | [9] | |
| asr(zh) | asr_shipinhao_fangyan | cer⬇️ | 42.80 | [10] | dialect |
| asr(zh) | asr_badcase | cer⬇️ | 17.16 | [11] | hard cases |

---

## Evaluation Commands

[1] `python audio_evals/main.py --dataset tedlium-release1 --model glm-asr-en`
[2] `python audio_evals/main.py --dataset fleurs-en_us --model glm-asr-en`
[3] `python audio_evals/main.py --dataset fleurs-zh --model glm-asr-zh`
[4] `python audio_evals/main.py --dataset KeSpeech --model glm-asr-zh`
[5] `python audio_evals/main.py --dataset WenetSpeech-test-net --model glm-asr-zh`
[6] `python audio_evals/main.py --dataset WenetSpeech-test-meeting --model glm-asr-zh`
[7] `python audio_evals/main.py --dataset asr_lianghui --model glm-asr-zh`
[8] `python audio_evals/main.py --dataset asr_qunliao_eyi --model glm-asr-zh`
[9] `python audio_evals/main.py --dataset asr_shipinhao_long --model glm-asr-zh`
[10] `python audio_evals/main.py --dataset asr_shipinhao_fangyan --model glm-asr-zh`
[11] `python audio_evals/main.py --dataset asr_badcase --model glm-asr-zh`
