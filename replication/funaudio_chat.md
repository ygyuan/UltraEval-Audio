# FunAudio-Chat Evaluation Results

**Model**: `funaudio_chat` ([config](../registry/model/funaudio.yaml))
**Evaluation Date**: 2026/05

**Metrics Legend**:
- **WER⬇️**: Word Error Rate (lower is better)
- **CER⬇️**: Character Error Rate (lower is better)
- **BLEU⬆️**: BLEU Score (higher is better)
- **ACC⬆️**: Accuracy (higher is better)

---

## ASR (English)

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| asr(en) | tedlium-release1 | wer⬇️ | 4.53 | [1] | |
| asr(en) | fleurs-en_us | wer⬇️ | 7.33 | [2] | |

## ASR (Chinese)

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| asr(zh) | fleurs-zh | cer⬇️ | 12.95 | [3] | |
| asr(zh) | KeSpeech | cer⬇️ | 12.68 | [4] | |
| asr(zh) | WenetSpeech-test-net | cer⬇️ | 21.77 | [5] | |
| asr(zh) | WenetSpeech-test-meeting | cer⬇️ | 12.61 | [6] | |

## Audio Speech Translation

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| ast | covost2-zh-en | bleu⬆️ | 10.00 | [7] | |
| ast | covost2-en-zh | bleu⬆️ | 24.09 | [8] | |

## Emotion Recognition

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| emo | meld-emo | acc⬆️ | 39.66 | [9] | |

## Audio Generation (Speech → Speech)

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| speech-qa | Speech Web Questions | acc⬆️ | 42.77 | [10] | |
| speech-qa | Speech TriviaQA | acc⬆️ | 43.85 | [11] | |
| speech-qa | Speech CMMLU | acc⬆️ | 71.10 | [12] | |

---

## Evaluation Commands

[1] `python audio_evals/main.py --dataset tedlium-release1 --model funaudio_chat`
[2] `python audio_evals/main.py --dataset fleurs-en_us --model funaudio_chat`
[3] `python audio_evals/main.py --dataset fleurs-zh --model funaudio_chat`
[4] `python audio_evals/main.py --dataset KeSpeech --model funaudio_chat`
[5] `python audio_evals/main.py --dataset WenetSpeech-test-net --model funaudio_chat`
[6] `python audio_evals/main.py --dataset WenetSpeech-test-meeting --model funaudio_chat`
[7] `python audio_evals/main.py --dataset covost2-zh-en --model funaudio_chat`
[8] `python audio_evals/main.py --dataset covost2-en-zh --model funaudio_chat`
[9] `python audio_evals/main.py --dataset meld-emo --model funaudio_chat`
[10] `python audio_evals/main.py --dataset speech-web-questions --model funaudio_chat`
[11] `python audio_evals/main.py --dataset speech-triviaqa --model funaudio_chat`
[12] `python audio_evals/main.py --dataset speech-cmmlu --model funaudio_chat`
