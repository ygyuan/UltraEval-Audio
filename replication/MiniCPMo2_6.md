# MiniCPM-o 2.6 Evaluation Results

**Model**: [MiniCPM-o-2_6](../registry/model/minicpmo.yaml)
**Evaluation Date**: 2026/03

**Metrics Legend**:
- **WER⬇️**: Word Error Rate (lower is better)
- **CER⬇️**: Character Error Rate (lower is better)
- **BLEU⬆️**: BLEU Score (higher is better)
- **ACC⬆️**: Accuracy (higher is better)

---

## ASR (English)

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| asr(en) | tedlium-release1 | wer⬇️ | 3.18 | [1] | |
| asr(en) | fleurs-en_us | wer⬇️ | 5.77 | [2] | |

## ASR (Chinese)

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| asr(zh) | fleurs-zh | cer⬇️ | 2.59 | [3] | |
| asr(zh) | WenetSpeech-test-net | cer⬇️ | 7.27 | [4] | |
| asr(zh) | WenetSpeech-test-meeting | cer⬇️ | 15.74 | [5] | |
| asr(zh) | KeSpeech | cer⬇️ | 23.28 | [6] | |

## Audio Speech Translation

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| ast | covost2-zh-en | bleu⬆️ | 25.63 | [7] | |
| ast | covost2-en-zh | bleu⬆️ | 32.21 | [8] | |

## Emotion Recognition

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| emo | meld-emo | acc⬆️ | 51.42 | [9] | |

## Audio Generation (Speech → Speech)

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| speech-qa | Speech Web Questions | acc⬆️ | 40.55 | [10] | |
| speech-qa | Speech TriviaQA | acc⬆️ | 41.21 | [11] | |
| speech-qa | Speech CMMLU | acc⬆️ | 50.28 | [12] | |

---

## Evaluation Commands

[1] `python audio_evals/main.py --dataset tedlium-release1 --model MiniCPMo2_6-audio --prompt mini-cpm-omni-asr-en`
[2] `python audio_evals/main.py --dataset fleurs-en_us --model MiniCPMo2_6-audio --prompt mini-cpm-omni-asr-en`
[3] `python audio_evals/main.py --dataset fleurs-zh --model MiniCPMo2_6-audio --prompt mini-cpm-omni-asr-zh`
[4] `python audio_evals/main.py --dataset WenetSpeech-test-net --model MiniCPMo2_6-audio --prompt mini-cpm-omni-asr-zh`
[5] `python audio_evals/main.py --dataset WenetSpeech-test-meeting --model MiniCPMo2_6-audio --prompt mini-cpm-omni-asr-zh`
[6] `python audio_evals/main.py --dataset KeSpeech --model MiniCPMo2_6-audio --prompt mini-cpm-omni-asr-zh`
[7] `python audio_evals/main.py --dataset covost2-zh-en --model MiniCPMo2_6-audio --prompt mini-cpm-omni-s2tt-zh2en`
[8] `python audio_evals/main.py --dataset covost2-en-zh --model MiniCPMo2_6-audio --prompt mini-cpm-omni-s2tt-en2zh`
[9] `python audio_evals/main.py --dataset meld-emo --model MiniCPMo2_6-audio --prompt mini-cpm-omni-emotion_analysis`

[10] `python audio_evals/main.py --dataset speech-web-questions --model MiniCPMo2_6-speech`
[11] `python audio_evals/main.py --dataset speech-triviaqa --model MiniCPMo2_6-speech`
[12] `python audio_evals/main.py --dataset speech-cmmlu --model MiniCPMo2_6-speech`
