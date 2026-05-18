import json
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Process(ABC):
    @abstractmethod
    def __call__(self, answer: str) -> str:
        raise NotImplementedError()


class ContentExtract(Process):

    def __call__(self, answer: str) -> str:
        try:
            answer = answer.strip()
            if answer.startswith("```json"):
                answer = answer[7:-3].strip()
            elif answer.startswith("```"):
                answer = answer[3:-3].strip()
            d = json.loads(answer)
            if isinstance(d, dict):
                if "content" in d:
                    return d["content"]
                if "text" in d:
                    return d["text"]
            return d
        except (json.JSONDecodeError, TypeError):
            pass
        except Exception as e:
            logger.warning("ContentExtract json.loads failed for: %.100s, error: %s", answer, e)
        # Fallback: return original text as-is (skip ast.literal_eval for non-Python expressions)
        return answer


class JsonExtract(Process):
    """
    Extract a specific key from a json string.
    the key is specified by the `extract_key` parameter.
    if the key is not found, return the `default_value` if specified,
    otherwise raise a KeyError.
    """

    def __init__(self, extract_key: str = None, default_value: str = None):
        """
        Initialize the JsonExtract process.
        Args:
            extract_key: required, the key to extract from the json string.
            default_value: optional, the default value to return if the key is not found.

        Returns: JsonExtract object.

        """
        self.extract_key = extract_key
        self.default_value = default_value

    def __call__(self, answer: str) -> any:
        """
        Extract the value of the `extract_key` from the json string `answer`.
        Args:
            answer: required, the json string to extract the value from.

        Returns: any, the value of the `extract_key` in the json string `answer`.

        """
        if isinstance(answer, str):
            try:
                d = json.loads(answer.strip())
            except Exception as e:
                logger.debug(f"load json `{answer}` fail: {str(e)}")
                return answer
        elif isinstance(answer, dict):
            d = answer
        else:
            raise ValueError(f"Unsupported answer type: {type(answer)}")
        if self.extract_key is None:
            return d

        if self.extract_key in d:
            return d[self.extract_key]
        if self.default_value is not None:
            return self.default_value
        logger.warning(
            "JsonExtract: key '%s' not found in parsed JSON (keys: %s), returning raw input",
            self.extract_key, list(d.keys())
        )
        return answer


class BracketedTagStrip(Process):
    """
    Strip bracketed structural tags from a string.

    Some ASR models (e.g. VibeVoice-ASR) may emit non-speech / structural
    tags such as ``[Music]``, ``[Lyric]``, ``[Vocal]``, ``[Singing]``,
    ``[Background]``, ``[Speech]``, ``[Noise]``, ``[Silence]``,
    ``[Applause]``, ``[Laughter]`` ...  These tags are not present in
    the reference transcriptions of standard ASR benchmarks and inflate
    CER artificially.

    This process removes any ``[Xxx]`` / ``[Xxx Yyy]`` ASCII tag and
    collapses the resulting whitespace.  Non-ASCII (e.g. Chinese) brackets
    or content are left untouched.
    """

    import re as _re
    _BRACKET_TAG_RE = _re.compile(r"\[\s*[A-Za-z][A-Za-z0-9 _\-/]*\s*\]")
    _WS_RE = _re.compile(r"\s+")

    def __call__(self, answer: str) -> str:
        if not isinstance(answer, str) or not answer:
            return answer
        cleaned = self._BRACKET_TAG_RE.sub(" ", answer)
        cleaned = self._WS_RE.sub(" ", cleaned).strip()
        return cleaned
