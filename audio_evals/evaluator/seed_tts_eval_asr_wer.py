from typing import Dict
import os
import re
from audio_evals.evaluator.base import Evaluator
import zhconv
try:
    import jiwer
    _jiwer_version = tuple(int(x) for x in jiwer.__version__.split(".")[:2])
except Exception:
    _jiwer_version = (0, 0)
try:
    from num2words import num2words
    _has_num2words = True
except ImportError:
    _has_num2words = False

def compute_measures(truth, hypo):
    """Wrapper for jiwer WER computation, compatible with multiple versions."""
    try:
        # Try new API first (jiwer >= 3.0)
        from jiwer import wer as jiwer_wer
        return {"wer": jiwer_wer(truth, hypo)}
    except Exception:
        from jiwer import compute_measures as _compute_measures
        return _compute_measures(truth, hypo)
from zhon.hanzi import punctuation
import string
import logging

logger = logging.getLogger(__name__)

punctuation_all = punctuation + string.punctuation

# Common English contractions mapping for expansion
_CONTRACTIONS = {
    "aren't": "are not",
    "can't": "can not",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}


def _normalize_apostrophes(text):
    """Normalize all Unicode apostrophe-like characters to ASCII apostrophe.
    e.g. \u2019 (\u2018) -> '
    """
    return re.sub(r"[\u2018\u2019\u0060\u00B4\u2032]", "'", text)


def _expand_contractions(text):
    """Expand English contractions. e.g. "aren't" -> "are not"."""
    for contraction, expansion in _CONTRACTIONS.items():
        text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
    return text


def _normalize_numbers(text, lang):
    """Normalize digits to words for fair WER comparison.
    e.g. '50' -> 'fifty', '3' -> 'three'
    """
    if not _has_num2words:
        return text
    try:
        n2w_lang = 'zh' if lang in ['zh', 'yue'] else 'en'
        text = re.sub(
            r"\d+",
            lambda m: f" {num2words(int(m.group(0)), lang=n2w_lang)} ",
            text,
        )
    except Exception:
        pass
    return text


def _normalize_en(text):
    """Comprehensive English text normalization for fair WER comparison.
    Handles: apostrophe variants, contractions, numbers, punctuation, casing.
    """
    text = text.strip()
    # 1. Normalize all apostrophe-like chars to ASCII apostrophe
    text = _normalize_apostrophes(text)
    # 2. Lowercase
    text = text.lower()
    # 3. Expand contractions (must be after lowercase + apostrophe normalization)
    text = _expand_contractions(text)
    # 4. Normalize numbers to words
    text = _normalize_numbers(text, "en")
    # 5. Remove hyphens between words by joining them (e.g. "south-west" -> "southwest")
    #    This ensures hyphenated and non-hyphenated forms match: "south-west" == "southwest"
    text = re.sub(r"(\w)-(\w)", r"\1\2", text)
    # 6. Remove all punctuation (including apostrophes)
    text = re.sub(r"[^\w\s]", " ", text)
    # 6. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def process_one(hypo, truth, lang):
    raw_truth = truth
    raw_hypo = hypo

    if lang == "en":
        # Use comprehensive English normalization to handle:
        # - Unicode apostrophe variants (U+2019 vs U+0027)
        # - contractions: "I'm" / "I am", "aren't" / "are not"
        # - numbers: "50" -> "fifty"
        truth = _normalize_en(truth)
        hypo = _normalize_en(hypo)
    else:
        # Normalize apostrophes for all languages
        truth = _normalize_apostrophes(truth)
        hypo = _normalize_apostrophes(hypo)

        # Normalize numbers to words before comparison
        truth = _normalize_numbers(truth, lang)
        hypo = _normalize_numbers(hypo, lang)

        for x in punctuation_all:
            if x == "'":
                continue
            truth = truth.replace(x, "")
            hypo = hypo.replace(x, "")

        truth = truth.replace("  ", " ")
        hypo = hypo.replace("  ", " ")

        # yue: cantonese, th: thai
        if lang in ["zh", "ja", "yue", "th", "ko"]:
            truth = " ".join([x for x in truth if x.strip()])
            hypo = " ".join([x for x in hypo if x.strip()])
        else:
            truth = truth.lower()
            hypo = hypo.lower()

    try:
        measures = compute_measures(truth, hypo)
        wer = measures["wer"]
    except Exception as e:
        logger.error(f"Error computing measures: {e}. truth: '{truth}', hypo: '{hypo}'")
        raise e

    return wer


class SeedTTSEvalASRWER(Evaluator):
    def __init__(self, model_name, prompt_name, lang):
        from audio_evals.registry import registry

        self.prompt = registry.get_prompt(prompt_name)
        self.model = registry.get_model(model_name)
        self.lang = lang

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        pred = str(pred)
        label_text = kwargs["text"]
        assert os.path.exists(pred), "must be a valid audio file, but got {}".format(
            pred
        )

        real_prompt = self.prompt.load(WavPath=pred)

        # Pass language to model for non-Chinese languages or if specified
        # Whisper model expects language in generate_kwargs
        inf_kwargs = {}
        if self.lang != "zh":
            inf_kwargs["generate_kwargs"] = {
                "language": kwargs.get("language", self.lang)
            }

        transcription = self.model.inference(real_prompt, **inf_kwargs)

        if self.lang == "zh" or kwargs.get("language") == "chinese":
            transcription = zhconv.convert(transcription, "zh-cn")

        res = {"wer%": process_one(transcription, label_text, self.lang) * 100}
        res.update(
            {
                "transcription": transcription,
                "label_text": label_text,
                "pred": pred,
                "ref": label,
            }
        )
        return res
