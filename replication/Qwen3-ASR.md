# Qwen3-ASR Evaluation Results

**Model**: `qwen3-asr-en` / `qwen3-asr-zh` ([config](../registry/model/qwen3_asr.yaml))
**Evaluation Date**: 2026/06

**Metrics Legend**:
- **WER⬇️**: Word Error Rate (lower is better)
- **CER⬇️**: Character Error Rate (lower is better)

---

## ASR (English) — `qwen3-asr-en`

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| asr(en) | tedlium-release1 | wer⬇️ | 2.28 | [1] | |
| asr(en) | fleurs-en_us | wer⬇️ | 4.72 | [2] | |

## ASR (Chinese) — `qwen3-asr-zh`

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| asr(zh) | fleurs-zh | cer⬇️ | 2.78 | [3] | |
| asr(zh) | KeSpeech | cer⬇️ | 5.07 | [4] | |
| asr(zh) | WenetSpeech-test-net | cer⬇️ | 5.07 | [5] | |
| asr(zh) | WenetSpeech-test-meeting | cer⬇️ | 5.93 | [6] | |
| asr(zh) | asr_lianghui | cer⬇️ | 5.40 | [7] | fail_rate: 0.04% |
| asr(zh) | asr_qunliao_eyi | cer⬇️ | 7.91 | [8] | |
| asr(zh) | asr_shipinhao_long | cer⬇️ | 3.85 | [9] | |
| asr(zh) | asr_shipinhao_fangyan | cer⬇️ | 27.66 | [10] | dialect |
| asr(zh) | asr_badcase | cer⬇️ | 15.96 | [11] | hard cases |

---

## Evaluation Commands

[1] `python audio_evals/main.py --dataset tedlium-release1 --model qwen3-asr-en`
[2] `python audio_evals/main.py --dataset fleurs-en_us --model qwen3-asr-en`
[3] `python audio_evals/main.py --dataset fleurs-zh --model qwen3-asr-zh`
[4] `python audio_evals/main.py --dataset KeSpeech --model qwen3-asr-zh`
[5] `python audio_evals/main.py --dataset WenetSpeech-test-net --model qwen3-asr-zh`
[6] `python audio_evals/main.py --dataset WenetSpeech-test-meeting --model qwen3-asr-zh`
[7] `python audio_evals/main.py --dataset asr_lianghui --model qwen3-asr-zh`
[8] `python audio_evals/main.py --dataset asr_qunliao_eyi --model qwen3-asr-zh`
[9] `python audio_evals/main.py --dataset asr_shipinhao_long --model qwen3-asr-zh`
[10] `python audio_evals/main.py --dataset asr_shipinhao_fangyan --model qwen3-asr-zh`
[11] `python audio_evals/main.py --dataset asr_badcase --model qwen3-asr-zh`
