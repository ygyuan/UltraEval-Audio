from audio_evals.evaluator.base import Evaluator
from audio_evals.lib.wer import compute_wer


def _is_blank_ref(label: str) -> bool:
    """Return True if the reference is blank / whitespace only.

    WER/CER is undefined when the reference token list is empty.  We
    detect the obvious blank case here so we can raise a descriptive
    error before ``compute_wer`` divides by zero; ``eval_task._run``
    will catch the exception and skip the sample.  Edge cases where the
    text is non-empty but the language-specific normalizer strips it to
    empty are still handled defensively inside ``compute_wer``.
    """
    return label is None or not str(label).strip()


class WER(Evaluator):
    def __init__(self, ignore_case: bool = False, lang="en"):
        self.ignore_case = ignore_case
        self.lang = lang

    def _eval(self, pred: str, label: str, **kwargs):
        pred, label = str(pred), str(label)
        if self.ignore_case:
            pred, label = pred.lower(), label.lower()
        if _is_blank_ref(label):
            raise ValueError("skip sample: empty reference for WER")
        return {
            "wer%": compute_wer([label], [pred], language=self.lang) * 100,
        }


class CER(Evaluator):
    def __init__(self, ignore_case: bool = False):
        self.ignore_case = ignore_case

    def _eval(self, pred: str, label: str, **kwargs):
        pred, label = str(pred), str(label)
        if self.ignore_case:
            pred, label = pred.lower(), label.lower()
        if _is_blank_ref(label):
            raise ValueError("skip sample: empty reference for CER")
        return {"cer%": compute_wer([label], [pred], language="zh") * 100}


from jiwer import process_words
from zhon.hanzi import punctuation
import string


punctuation_all = punctuation + string.punctuation


def process_one(hypo, truth, lang):
    raw_truth = truth
    raw_hypo = hypo

    for x in punctuation_all:
        if x == "'":
            continue
        truth = truth.replace(x, "")
        hypo = hypo.replace(x, "")

    truth = truth.replace("  ", " ")
    hypo = hypo.replace("  ", " ")

    if lang == "zh":
        truth = " ".join([x for x in truth])
        hypo = " ".join([x for x in hypo])
    elif lang == "en":
        truth = truth.lower()
        hypo = hypo.lower()
    else:
        raise NotImplementedError

    measures = process_words(truth, hypo)
    wer = measures.wer
    return wer


class NaiveWER(Evaluator):
    def __init__(self, lang="en"):
        assert lang in ["en", "zh"], "Unsupported language"
        self.lang = lang

    def _eval(self, pred: str, label: str, **kwargs):
        pred, label = str(pred), str(label)
        return {"wer%": process_one(pred, label, self.lang) * 100}
