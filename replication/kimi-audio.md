# Kimi-Audio Evaluation Results

**Model**: [Kimi-Audio-7B-Instruct](../registry/model/moonshot.yaml)
**Evaluation Date**: 2026/03
**Paper**: [Kimi-Audio Technical Report](https://arxiv.org/abs/2504.18425)

**Metrics Legend**:
- **WER⬇️**: Word Error Rate (lower is better)
- **CER⬇️**: Character Error Rate (lower is better)
- **BLEU⬆️**: BLEU Score (higher is better)
- **ACC⬆️**: Accuracy (higher is better)

---

## ASR (English)

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| asr(en) | tedlium-release1 | wer⬇️ | 3.09 | [1] | |
| asr(en) | fleurs-en_us | wer⬇️ | 5.21 | [2] | |

## ASR (Chinese)

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| asr(zh) | fleurs-zh | cer⬇️ | 2.54 | [3] | |
| asr(zh) | KeSpeech | cer⬇️ | 5.02 | [4] | |
| asr(zh) | WenetSpeech-test-net | cer⬇️ | 5.47 | [5] | |
| asr(zh) | WenetSpeech-test-meeting | cer⬇️ | 6.37 | [6] | |

## Audio Speech Translation

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| ast | covost2-zh-en | bleu⬆️ | 14.60 | [7] | |
| ast | covost2-en-zh | bleu⬆️ | 35.07 | [8] | |

## Emotion Recognition

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| emo | meld-emo | acc⬆️ | 45.29 | [9] | |

## Audio Generation (Speech → Speech)

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| speech-qa | Speech CMMLU | acc⬆️ | 67.77 | [10] | |


## Evaluation Commands

[1] `python audio_evals/main.py --dataset tedlium-release1 --model kimiaudio --prompt kimi-audio-asr-en`
[2] `python audio_evals/main.py --dataset fleurs-en_us --model kimiaudio --prompt kimi-audio-asr-en`
[3] `python audio_evals/main.py --dataset fleurs-zh --model kimiaudio --prompt kimi-audio-asr-zh`
[4] `python audio_evals/main.py --dataset KeSpeech --model kimiaudio --prompt kimi-audio-asr-zh`
[5] `python audio_evals/main.py --dataset WenetSpeech-test-net --model kimiaudio --prompt kimi-audio-asr-zh`
[6] `python audio_evals/main.py --dataset WenetSpeech-test-meeting --model kimiaudio --prompt kimi-audio-asr-zh`
[7] `python audio_evals/main.py --dataset covost2-zh-en --model kimiaudio`
[8] `python audio_evals/main.py --dataset covost2-en-zh --model kimiaudio`
[9] `python audio_evals/main.py --dataset meld-emo --model kimiaudio`

[10] `python audio_evals/main.py --dataset speech-cmmlu --model kimiaudio-speech`
