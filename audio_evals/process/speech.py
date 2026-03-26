import json
import logging
import os.path
from audio_evals.process.base import Process

logger = logging.getLogger(__name__)


class Speech2text(Process):

    def __init__(self, model_name: str = "whisper", prompt_name: str = "whisper-asr"):
        from audio_evals.registry import registry

        self.model = registry.get_model(model_name)
        self.prompt = registry.get_prompt(prompt_name)

    def __call__(self, answer: str) -> str:
        # Try to extract text from JSON string (e.g. '{"text": "C"}')
        if isinstance(answer, str):
            try:
                d = json.loads(answer.strip())
                if isinstance(d, dict) and "text" in d:
                    if "audio" not in d:
                        # No audio field, return text directly (skip ASR)
                        logger.info("Speech2text: no audio in JSON, returning text field: %s", d["text"][:100])
                        return d["text"]
                    else:
                        answer = d["audio"]
            except (json.JSONDecodeError, TypeError):
                pass

        if not os.path.exists(answer):
            logger.warning(
                "Speech2text: expected a valid audio file path, but got text output. "
                "Returning as-is: %s", answer[:100]
            )
            return answer
        real_prompt = self.prompt.load(WavPath=answer)
        return self.model.inference(real_prompt)
