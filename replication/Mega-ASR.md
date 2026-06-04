# Mega-ASR Evaluation Results

**Model**: `mega-asr-en` / `mega-asr-zh` ([config](../registry/model/mega_asr.yaml))
**Evaluation Date**: 2026/06

**Metrics Legend**:
- **WER⬇️**: Word Error Rate (lower is better)
- **CER⬇️**: Character Error Rate (lower is better)

---

## ASR (English) — `mega-asr-en`

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| asr(en) | tedlium-release1 | wer⬇️ | 2.30 | [1] | |
| asr(en) | fleurs-en_us | wer⬇️ | 4.85 | [2] | |

## ASR (Chinese) — `mega-asr-zh`

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| asr(zh) | fleurs-zh | cer⬇️ | 2.75 | [3] | |
| asr(zh) | KeSpeech | cer⬇️ | 5.21 | [4] | |
| asr(zh) | WenetSpeech-test-net | cer⬇️ | 5.11 | [5] | |
| asr(zh) | WenetSpeech-test-meeting | cer⬇️ | 6.05 | [6] | |
| asr(zh) | asr_lianghui | cer⬇️ | 5.41 | [7] | fail_rate: 0.11% |
| asr(zh) | asr_qunliao_eyi | cer⬇️ | 8.28 | [8] | fail_rate: 1.98% |
| asr(zh) | asr_shipinhao_long | cer⬇️ | 3.93 | [9] | fail_rate: 0.21% |
| asr(zh) | asr_shipinhao_fangyan | cer⬇️ | 28.05 | [10] | dialect |
| asr(zh) | asr_badcase | cer⬇️ | 16.53 | [11] | hard cases |

---

## Evaluation Commands

[1] `python audio_evals/main.py --dataset tedlium-release1 --model mega-asr-en`
[2] `python audio_evals/main.py --dataset fleurs-en_us --model mega-asr-en`
[3] `python audio_evals/main.py --dataset fleurs-zh --model mega-asr-zh`
[4] `python audio_evals/main.py --dataset KeSpeech --model mega-asr-zh`
[5] `python audio_evals/main.py --dataset WenetSpeech-test-net --model mega-asr-zh`
[6] `python audio_evals/main.py --dataset WenetSpeech-test-meeting --model mega-asr-zh`
[7] `python audio_evals/main.py --dataset asr_lianghui --model mega-asr-zh`
[8] `python audio_evals/main.py --dataset asr_qunliao_eyi --model mega-asr-zh`
[9] `python audio_evals/main.py --dataset asr_shipinhao_long --model mega-asr-zh`
[10] `python audio_evals/main.py --dataset asr_shipinhao_fangyan --model mega-asr-zh`
[11] `python audio_evals/main.py --dataset asr_badcase --model mega-asr-zh`
