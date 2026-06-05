"""Re-aggregate an existing eval jsonl with the asr-mix policy.

Usage:
    python project/scripts/reagg_asr_mix.py <path-to-jsonl> [<more-jsonl> ...]

For each jsonl this script collects all ``"type": "eval"`` records, runs the
``MixedWER`` aggregator (zh-by-character / en-by-word) and prints the result
to stdout. It also writes the result alongside the input file as
``<basename>-overall-asr-mix.json`` so we don't overwrite the original
overall json produced by the run.
"""
from __future__ import annotations

import json
import os
import sys
from typing import List, Dict, Any

from audio_evals.agg.base import MixedWER


def reagg(path: str) -> Dict[str, Any]:
    score_detail: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") != "eval":
                continue
            d = obj.get("data", {})
            if "pred" in d and "ref" in d:
                score_detail.append(d)
    agg = MixedWER(ignore_case=True)
    return agg(score_detail)


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        sys.exit(1)
    for path in argv[1:]:
        if not os.path.exists(path):
            print(f"[skip] not found: {path}")
            continue
        result = reagg(path)
        print(f"==> {path}")
        print(f"    {result}")
        out = path.replace(".jsonl", "-overall-asr-mix.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"    written: {out}")


if __name__ == "__main__":
    main(sys.argv)
