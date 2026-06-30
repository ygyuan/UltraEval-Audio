import logging
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.WARNING)
import os
import socket
from contextlib import contextmanager
os.environ["HF_HUB_MAX_RETRY"] = "1"  # 只重试1次
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "10"  # 超时10秒
from typing import Dict, List, Optional

import soundfile as sf
from datasets import load_dataset, DatasetDict, Dataset, load_from_disk

from audio_evals.constants import DEFAULT_MODEL_PATH
from audio_evals.dataset.dataset import Dataset as BaseDataset

logger = logging.getLogger(__name__)


# Cache the network reachability check result so that the (already slow)
# probe is only paid once per process — subsequent dataset loads reuse it.
_HF_HUB_REACHABLE: Optional[bool] = None


def _hf_hub_reachable(timeout: float = 2.0) -> bool:
    """Probe whether ``huggingface.co`` is reachable on this host.

    A quick TCP connect on port 443 is far cheaper than letting
    ``huggingface_hub`` go through its 5-step exponential backoff
    (~23s wall time) for every single ``load_dataset`` call.
    """
    global _HF_HUB_REACHABLE
    if _HF_HUB_REACHABLE is not None:
        return _HF_HUB_REACHABLE

    # Respect explicit user override: if any of the standard offline env
    # vars is set we should skip the probe and treat the hub as down.
    if (
        os.environ.get("HF_HUB_OFFLINE") == "1"
        or os.environ.get("HF_DATASETS_OFFLINE") == "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    ):
        _HF_HUB_REACHABLE = False
        return False

    try:
        with socket.create_connection(("huggingface.co", 443), timeout=timeout):
            _HF_HUB_REACHABLE = True
    except (OSError, socket.timeout):
        _HF_HUB_REACHABLE = False
        logger.info(
            "huggingface.co is not reachable (offline environment detected); "
            "future dataset loads will skip Hub lookups and use local "
            "fallbacks under %s.",
            DEFAULT_MODEL_PATH,
        )
    return _HF_HUB_REACHABLE


@contextmanager
def _force_hf_offline():
    """Temporarily set the HF offline environment flags.

    ``load_dataset(path=<local-dir>)`` still tries to HEAD the README on
    the hub by default, which costs ~23s of retries on offline machines.
    Setting these flags short-circuits that behaviour for the duration
    of the ``with`` block.
    """
    keys = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
    saved = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ[k] = "1"
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def save_audio_to_local(ds: Dataset, save_path: str):
    """
    save audio file to local.
    :param ds:
    :param save_path:
    :return:
    """

    # Sentinel WavPath used to mark broken / undecodable rows so that we
    # can filter them out after ``ds.map``.  We can not simply ``return
    # None`` from ``save_audio`` because ``datasets`` requires the mapped
    # function to return a dict with the same schema.
    _BROKEN_SENTINEL = ""

    bad_counter = {"n": 0}

    def save_audio(example, index):
        if "audio" not in example:
            logger.error(f"audio not in example: {example}, skip this example")
            example["WavPath"] = _BROKEN_SENTINEL
            bad_counter["n"] += 1
            return example
        # Some HF parquet shards contain rows whose ``audio.bytes`` is
        # ``None`` (upload pipeline lost the audio blob) or whose blob
        # is corrupted.  Accessing ``example['audio']['array']`` triggers
        # datasets' Audio decoder which calls ``soundfile.read`` and
        # raises ``LibsndfileError: Format not recognised`` on such
        # rows, killing the whole ``ds.map`` worker.  Catch decode
        # failures and mark the row as broken so the pipeline can skip
        # it instead of aborting the entire eval run.
        try:
            audio = example["audio"]
            if audio is None:
                raise ValueError("audio field is None")
            audio_array = audio["array"]
            sampling_rate = audio["sampling_rate"]
            if audio_array is None or len(audio_array) == 0:
                raise ValueError("empty audio array")
        except Exception as e:
            audio_path_hint = ""
            try:
                audio_path_hint = (example.get("audio") or {}).get("path") or ""
            except Exception:
                pass
            logger.warning(
                "skip broken audio row index=%d path=%r: %s: %s",
                index,
                audio_path_hint,
                type(e).__name__,
                str(e)[:200],
            )
            example["WavPath"] = _BROKEN_SENTINEL
            bad_counter["n"] += 1
            return example

        output_path = os.path.join(save_path, f"{index}.wav")
        example["WavPath"] = output_path
        d = os.path.dirname(output_path)
        os.makedirs(d, exist_ok=True)
        if not os.path.exists(output_path):
            try:
                sf.write(output_path, audio_array, sampling_rate)
            except Exception as e:
                logger.warning(
                    "skip row index=%d, sf.write failed: %s: %s",
                    index,
                    type(e).__name__,
                    str(e)[:200],
                )
                example["WavPath"] = _BROKEN_SENTINEL
                bad_counter["n"] += 1
        return example

    # Drop the ``audio`` column from the mapped output schema.  Reasons:
    #  1. We have already written every decodable sample to disk as a
    #     ``WavPath`` wav file, so downstream code never reads ``audio``
    #     again.
    #  2. Keeping the ``audio`` column would force ``datasets``' Arrow
    #     writer (in ``writer.finalize`` -> ``Audio.encode_example``) to
    #     re-encode every sample with ``sf.write``.  For broken rows
    #     whose ``array`` is a 0-d / empty numpy scalar this raises
    #     ``IndexError: tuple index out of range`` at
    #     ``data.shape[1]`` and kills the whole map at the very end,
    #     even though we already marked those rows as broken.
    remove_cols = ["audio"] if "audio" in ds.column_names else None
    ds = ds.map(
        save_audio,
        with_indices=True,
        load_from_cache_file=False,
        remove_columns=remove_cols,
    )
    if bad_counter["n"] > 0:
        logger.warning(
            "save_audio_to_local: skipped %d broken row(s) out of %d total",
            bad_counter["n"],
            len(ds),
        )
        ds = ds.filter(
            lambda ex: bool(ex.get("WavPath")) and ex["WavPath"] != _BROKEN_SENTINEL,
            load_from_cache_file=False,
        )
    return ds


def load_audio_hf_dataset(name, subset=None, split="", local_path="", col_aliases=None):
    if col_aliases is None:
        col_aliases = {}
    if local_path:
        ds = load_from_disk(local_path)
    else:
        load_args = {"path": name}
        if subset:
            load_args["name"] = subset
        if split:
            load_args["split"] = split

        # ----------------------------------------------------------------
        # Fast-path for offline / air-gapped environments.
        #
        # Previously this function always tried ``load_dataset(path=<repo>)``
        # first, which on an offline host wastes ~23 seconds inside
        # ``huggingface_hub``'s urllib3 retry loop before raising
        # ``LocalEntryNotFoundError`` and falling back. Then the fallback
        # call ``load_dataset(path=<local-dir>)`` ALSO tries to HEAD the
        # README on the Hub, costing another ~23 seconds.
        #
        # We now:
        #   1. Probe ``huggingface.co`` once with a 2 s TCP connect.
        #   2. If unreachable AND a local fallback exists under
        #      ``init_model/<repo>``, skip the Hub call entirely and load
        #      the local copy with HF_HUB_OFFLINE=1 set so datasets does
        #      not retry the README HEAD either.
        # ----------------------------------------------------------------
        local_fallback = os.path.join(DEFAULT_MODEL_PATH, name)
        local_fallback_exists = os.path.exists(local_fallback)

        if local_fallback_exists and not _hf_hub_reachable():
            logger.info(
                "Hub unreachable, loading dataset directly from local path: %s",
                local_fallback,
            )
            fallback_args = {**load_args, "path": local_fallback}
            with _force_hf_offline():
                ds = load_dataset(**fallback_args, trust_remote_code=True)
        else:
            try:
                ds = load_dataset(**load_args, trust_remote_code=True)
            except Exception as e:
                logger.warning(f"load args is {load_args} load dataset from Hub failed: {e}")
                # Fallback: try loading from local init_model directory
                if local_fallback_exists:
                    logger.info(f"Falling back to local dataset path: {local_fallback}")
                    fallback_args = {**load_args, "path": local_fallback}
                    try:
                        # Force offline mode so the fallback does not
                        # repeat the 5-step retry storm.
                        with _force_hf_offline():
                            ds = load_dataset(
                                **fallback_args, trust_remote_code=True
                            )
                    except Exception as e2:
                        logger.error(f"Local fallback also failed: {e2}")
                        raise e2
                else:
                    logger.error(f"No local fallback found at {local_fallback}")
                    raise e

    for k, v in col_aliases.items():
        if v in ds.column_names:
            raise ValueError(f"col_aliases conflict with existing column name: {v}")
        ds = ds.rename_column(k, v)

    def conv2ds(ds):
        save_path = f"raw/{name}/"
        if subset:
            save_path += f"{subset}/"
        if split:
            save_path += f"{split}/"

        os.makedirs(save_path, exist_ok=True)
        return list(save_audio_to_local(ds, save_path))

    if isinstance(ds, DatasetDict):
        result = []
        for k in ds:
            reload_ds = {
                "name": name,
                "subset": subset,
                "split": k,
                "local_path": local_path,
                "col_aliases": col_aliases,
            }
            result.extend(load_audio_hf_dataset(**reload_ds))
        return result
    return conv2ds(ds)


class Huggingface(BaseDataset):
    def __init__(
        self,
        name: str,
        default_task: str,
        ref_col: str,
        subset: Optional[str] = None,
        split: str = "",
        local_path: str = "",
        col_aliases: Dict[str, str] = None,
    ):
        super().__init__(default_task, ref_col, col_aliases)
        self.name = name
        self.subset = subset
        self.split = split
        self.local_path = local_path

    def load(self, limit=0) -> List[Dict[str, any]]:
        logger.info(
            "start load data, it will take a while for download dataset when first load dataset"
        )
        res = load_audio_hf_dataset(
            self.name, self.subset, self.split, self.local_path, self.col_aliases
        )
        return res[:limit] if limit > 0 else res
