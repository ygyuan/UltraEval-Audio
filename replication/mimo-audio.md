# MiMo-Audio Evaluation Results

**Model**: `mimo-audio` ([config](../registry/model/mimo.yaml))
**Evaluation Date**: 2026/04

**Metrics Legend**:
- **WER⬇️**: Word Error Rate (lower is better)
- **CER⬇️**: Character Error Rate (lower is better)
- **BLEU⬆️**: BLEU Score (higher is better)
- **ACC⬆️**: Accuracy (higher is better)

---

## ASR (English)

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| asr(en) | tedlium-release1 | wer⬇️ | 16.36 | [1] | |
| asr(en) | fleurs-en_us | wer⬇️ | 21.51 | [2] | |

## ASR (Chinese)

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| asr(zh) | fleurs-zh | cer⬇️ | 7.37 | [3] | |
| asr(zh) | KeSpeech | cer⬇️ | 21.84 | [4] | |
| asr(zh) | WenetSpeech-test-net | cer⬇️ | 11.94 | [5] | |
| asr(zh) | WenetSpeech-test-meeting | cer⬇️ | 9.84 | [6] | |

## Audio Speech Translation

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| ast | covost2-zh-en | bleu⬆️ | 14.23 | [7] | |
| ast | covost2-en-zh | bleu⬆️ | 26.34 | [8] | |

## Emotion Recognition

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| emo | meld-emo | acc⬆️ | 44.67 | [9] | |

## Audio Generation (Speech → Speech)

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| speech-qa | Speech Web Questions | acc⬆️ | 37.80 | [10] | |
| speech-qa | Speech TriviaQA | acc⬆️ | 33.89 | [11] | |
| speech-qa | Speech CMMLU | acc⬆️ | 54.83 | [12] | |

---

## Evaluation Commands

[1] `python audio_evals/main.py --dataset tedlium-release1 --model mimo-audio`
[2] `python audio_evals/main.py --dataset fleurs-en_us --model mimo-audio`
[3] `python audio_evals/main.py --dataset fleurs-zh --model mimo-audio`
[4] `python audio_evals/main.py --dataset KeSpeech --model mimo-audio`
[5] `python audio_evals/main.py --dataset WenetSpeech-test-net --model mimo-audio`
[6] `python audio_evals/main.py --dataset WenetSpeech-test-meeting --model mimo-audio`
[7] `python audio_evals/main.py --dataset covost2-zh-en --model mimo-audio`
[8] `python audio_evals/main.py --dataset covost2-en-zh --model mimo-audio`
[9] `python audio_evals/main.py --dataset meld-emo --model mimo-audio`
[10] `python audio_evals/main.py --dataset speech-web-questions --model mimo-audio`
[11] `python audio_evals/main.py --dataset speech-triviaqa --model mimo-audio`
[12] `python audio_evals/main.py --dataset speech-cmmlu --model mimo-audio`
