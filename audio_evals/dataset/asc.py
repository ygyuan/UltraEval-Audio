import glob
import logging
import os
from typing import Dict, List

from audio_evals.constants import DEFAULT_MODEL_PATH
from audio_evals.dataset.huggingface import (
    Huggingface,
    load_audio_hf_dataset,
    save_audio_to_local,
    _force_hf_offline,
)

logger = logging.getLogger(__name__)


class ASCDataset(Huggingface):
    """Audio Scene Classification dataset (Yougen/asc_testset).

    The dataset is packed in WebDataset format. After loading via
    ``datasets.load_dataset`` the columns are ``wav`` (decoded audio dict),
    ``json`` (per-sample metadata dict including ``label_str`` / ``label`` /
    ``id`` / ``rel_path`` / ``duration`` / ``wav_format``), ``__key__`` and
    ``__url__``.

    The base ``Huggingface`` pipeline expects an ``audio`` column for audio
    decoding and a flat ``ref_col`` (e.g. ``label_str``) at the top level.
    Here we:

    1. Force ``col_aliases`` to rename ``wav`` -> ``audio`` so that
       ``save_audio_to_local`` can write each sample to disk.
    2. Flatten the ``json`` dict into top level columns so that the
       evaluator can read ``ref_col`` (e.g. ``label_str``) directly.
    """

    def __init__(self, **kwargs):
        col_aliases = kwargs.get("col_aliases") or {}
        # ensure wav is exposed as audio for save_audio_to_local
        if "wav" not in col_aliases and "audio" not in col_aliases.values():
            col_aliases["wav"] = "audio"
        kwargs["col_aliases"] = col_aliases
        super().__init__(**kwargs)

    # Keys from the nested ``json`` field that would collide with parameters
    # of ``Evaluator._eval(self, pred, label, **kwargs)`` and friends if
    # passed through ``**doc`` to the evaluator.  We rename them on
    # promotion so the eval task can still reach the original values
    # (e.g. ``label_id``) without breaking the call signature.
    _RESERVED_KWARG_NAMES = {"pred", "label", "ref", "self"}

    def _try_load_single_split_local(self):
        """Try to load a single split directly from the local fallback dir.

        The published ``Yougen/asc_testset`` repo's ``README.md`` declares
        7 splits (``test_a1..a5``, ``test_p1``, ``test_p2``).  Even when
        ``load_dataset(..., split="test_p1")`` is used, ``datasets`` still
        resolves and *generates* all 7 splits' arrow caches on first load
        (we observed ``Generating test_a1 split: 3848 examples`` and
        ``Generating test_a2 split: 7889 examples`` even though only
        ``test_p1`` is needed), which wastes a lot of time and disk.

        To avoid this, when a single ``split`` is requested and the local
        webdataset shards exist under
        ``init_model/<repo>/data/<split>/audio/*.tar`` we bypass the
        README config entirely and load that split directly via the
        generic ``webdataset`` builder with explicit ``data_files``.

        Returns ``None`` if this fast path is not applicable so the
        caller falls back to the default loader.
        """
        if not self.split or self.local_path:
            return None

        local_root = os.path.join(DEFAULT_MODEL_PATH, self.name)
        shard_glob = os.path.join(
            local_root, "data", self.split, "audio", "*.tar"
        )
        shards = sorted(glob.glob(shard_glob))
        if not shards:
            return None

        from datasets import load_dataset

        logger.info(
            "ASCDataset fast-path: loading only split=%s from %d local shard(s) "
            "(bypassing README's multi-split config)",
            self.split,
            len(shards),
        )

        with _force_hf_offline():
            ds = load_dataset(
                "webdataset",
                data_files={self.split: shards},
                split=self.split,
                trust_remote_code=True,
            )

        col_aliases = self.col_aliases or {}
        for k, v in col_aliases.items():
            if k in ds.column_names:
                if v in ds.column_names:
                    raise ValueError(
                        f"col_aliases conflict with existing column name: {v}"
                    )
                ds = ds.rename_column(k, v)

        save_path = f"raw/{self.name}/"
        if self.subset:
            save_path += f"{self.subset}/"
        save_path += f"{self.split}/"
        os.makedirs(save_path, exist_ok=True)
        return list(save_audio_to_local(ds, save_path))

    def load(self, limit=0) -> List[Dict[str, any]]:
        logger.info(
            "start load data, it will take a while for download dataset when first load dataset"
        )
        raw = self._try_load_single_split_local()
        if raw is None:
            raw = load_audio_hf_dataset(
                self.name, self.subset, self.split, self.local_path, self.col_aliases
            )
        res = []
        for item in raw:
            meta = item.pop("json", None)
            if isinstance(meta, dict):
                for k, v in meta.items():
                    # rename keys that would collide with Evaluator._eval
                    # positional parameters (e.g. ``label``) when the doc is
                    # forwarded as ``**kwargs``.
                    out_key = f"{k}_id" if k in self._RESERVED_KWARG_NAMES else k
                    # do not overwrite existing top level keys (e.g. WavPath)
                    if out_key not in item:
                        item[out_key] = v
            res.append(item)
        if limit > 0:
            res = res[:limit]
        return res
