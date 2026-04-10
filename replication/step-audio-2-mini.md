# Step-Audio-2-mini Evaluation Results

**Model**: [Step-Audio-2-mini](../registry/model/step.yaml)
**Evaluation Date**: 2026/04
**Paper**: [Step-Audio-2-mini Technical Report](https://arxiv.org/pdf/2506.09333)

**Metrics Legend**:
- **WER⬇️**: Word Error Rate (lower is better)
- **CER⬇️**: Character Error Rate (lower is better)
- **BLEU⬆️**: BLEU Score (higher is better)
- **ACC⬆️**: Accuracy (higher is better)

---

## ASR (English)

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| asr(en) | tedlium-release1 | wer⬇️ | 3.45 | [1] | |
| asr(en) | fleurs-en_us | wer⬇️ | 5.94 | [2] | |

## ASR (Chinese)

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| asr(zh) | WenetSpeech-test-net | cer⬇️ | 6.28 | [3] | |
| asr(zh) | WenetSpeech-test-meeting | cer⬇️ | 5.52 | [4] | |
| asr(zh) | KeSpeech | cer⬇️ | 4.04 | [5] | |
| asr(zh) | fleurs-zh | cer⬇️ | 8.04 | [6] | |

## Audio Speech Translation

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| ast | covost2-zh-en | bleu⬆️ | 24.46 | [7] | |
| ast | covost2-en-zh | bleu⬆️ | 49.20 | [8] | |

## Emotion Recognition

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| emo | meld-emo | acc⬆️ | 55.48 | [9] | |

## Audio Generation (Speech → Speech)

| task | dataset | measure | performance | eval_cli | note |
|------|---------|---------|-------------|----------|------|
| speech-qa | Speech CMMLU | acc⬆️ | 72.08 | [10] | |
| speech-qa | Speech Web Questions | acc⬆️ | 41.78 | [11] | |
| speech-qa | Speech TriviaQA | acc⬆️ | 41.11 | [12] | |

---

## Evaluation Commands

[1] `python -m audio_evals.main --dataset tedlium-release1 --model Step-Audio-2-mini --prompt step_audio_2_mini_asr_en --use_model_pool --workers 8`
[2] `python -m audio_evals.main --dataset fleurs-en_us --model Step-Audio-2-mini --prompt step_audio_2_mini_asr_en --use_model_pool --workers 8`
[3] `python -m audio_evals.main --dataset WenetSpeech-test-net --model Step-Audio-2-mini --prompt step_audio_2_mini_asr_zh --use_model_pool --workers 8`
[4] `python -m audio_evals.main --dataset WenetSpeech-test-meeting --model Step-Audio-2-mini --prompt step_audio_2_mini_asr_zh --use_model_pool --workers 8`
[5] `python -m audio_evals.main --dataset KeSpeech --model Step-Audio-2-mini --prompt step_audio_2_mini_asr_zh --use_model_pool --workers 8`
[6] `python -m audio_evals.main --dataset fleurs-zh --model Step-Audio-2-mini --prompt step_audio_2_mini_asr_zh --use_model_pool --workers 8`
[7] `python -m audio_evals.main --dataset covost2-zh-en --model Step-Audio-2-mini --prompt step_audio_2_mini_s2tt_en --use_model_pool --workers 8`
[8] `python -m audio_evals.main --dataset covost2-en-zh --model Step-Audio-2-mini --prompt step_audio_2_mini_s2tt_zh --use_model_pool --workers 8`
[9] `python -m audio_evals.main --dataset meld-emo --model Step-Audio-2-mini --use_model_pool --workers 8`
[10] `python -m audio_evals.main --dataset speech-cmmlu --model Step-Audio-2-mini --use_model_pool --workers 8`
[11] `python -m audio_evals.main --dataset speech-web-questions --model Step-Audio-2-mini --use_model_pool --workers 8`
[12] `python -m audio_evals.main --dataset speech-triviaqa --model Step-Audio-2-mini --use_model_pool --workers 8`
