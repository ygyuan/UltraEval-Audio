"""
Re-evaluate an existing VibeVoice-ASR inference jsonl without rerunning the
model.

Reads an evaluation jsonl produced by ``audio_evals/main.py`` (e.g.
``res/vibevoice-asr/WenetSpeech-test-net/2026-05-12_10-47-04.jsonl``),
applies the ``BracketedTagStrip`` post-processor to the ``inference`` /
``post_process`` outputs (removing ``[Music]`` / ``[Lyric]`` / ``[Vocal]``
... structural tags), and recomputes CER (zh) or WER (en) using the
project's standard evaluators.  The cleaned jsonl is written next to the
input file with a ``.recleaned.jsonl`` suffix and an ``-overall.json``
summary is also produced.

Usage
-----
    python project/scripts/recompute_vibevoice_asr_cer.py \\
        --input  res/vibevoice-asr/WenetSpeech-test-net/2026-05-12_10-47-04.jsonl \\
        --metric cer

    python project/scripts/recompute_vibevoice_asr_cer.py \\
        --input  res/vibevoice-asr/tedlium-release1/<file>.jsonl \\
        --metric wer
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

# Make sure the project root is on sys.path so that ``audio_evals`` imports
# work regardless of where this script is invoked from.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from audio_evals.process.base import BracketedTagStrip  # noqa: E402
from audio_evals.evaluator.wer import CER, WER  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the existing inference jsonl to re-evaluate.",
    )
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Output jsonl path (default: <input>.recleaned.jsonl).  An "
            "<output>-overall.json summary is also written."
        ),
    )
    parser.add_argument(
        "--metric",
        default="cer",
        choices=["cer", "wer"],
        help="Which evaluator to use (cer for zh, wer for en). Default: cer.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    if args.output:
        output_path = args.output
    else:
        if input_path.endswith(".jsonl"):
            output_path = input_path[: -len(".jsonl")] + ".recleaned.jsonl"
        else:
            output_path = input_path + ".recleaned.jsonl"
    overall_path = output_path.replace(".jsonl", "-overall.json")

    cleaner = BracketedTagStrip()
    if args.metric == "cer":
        evaluator = CER(ignore_case=True)
        score_key = "cer%"
    else:
        evaluator = WER(ignore_case=True, lang="en")
        score_key = "wer%"

    # First pass: collect prompts (audio path / ref) and inferences per id.
    docs_by_id: Dict[int, Dict[str, dict]] = defaultdict(dict)
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" not in rec or "type" not in rec:
                continue
            docs_by_id[int(rec["id"])][rec["type"]] = rec.get("data", {})

    out_lines: List[str] = []
    scores: List[float] = []
    skipped = 0
    cleaned_count = 0

    for idx in sorted(docs_by_id.keys()):
        bundle = docs_by_id[idx]

        prompt_data = bundle.get("prompt") or {}
        inference_data = bundle.get("inference") or {}
        old_eval_data = bundle.get("eval") or {}

        if "content" not in inference_data:
            skipped += 1
            continue

        old_pred = str(inference_data.get("content", ""))
        new_pred = cleaner(old_pred)
        if new_pred != old_pred:
            cleaned_count += 1

        ref = str(old_eval_data.get("ref", ""))
        if not ref:
            skipped += 1
            continue

        try:
            score = evaluator._eval(new_pred, ref)[score_key]
        except Exception as e:
            print(f"[WARN] eval failed at id={idx}: {e}", file=sys.stderr)
            skipped += 1
            continue

        scores.append(score)

        # Re-emit the four standard records for this id.
        out_lines.append(json.dumps(
            {"type": "prompt", "id": idx, "data": prompt_data},
            ensure_ascii=False,
        ))
        out_lines.append(json.dumps(
            {"type": "inference", "id": idx, "data": {"content": old_pred}},
            ensure_ascii=False,
        ))
        out_lines.append(json.dumps(
            {"type": "post_process", "id": idx, "data": {"content": new_pred}},
            ensure_ascii=False,
        ))
        out_lines.append(json.dumps(
            {"type": "eval", "id": idx,
             "data": {"pred": new_pred, "ref": ref, score_key: score}},
            ensure_ascii=False,
        ))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")

    if scores:
        avg = sum(scores) / len(scores)
    else:
        avg = float("nan")

    summary = {
        "input": input_path,
        "output": output_path,
        "metric": args.metric,
        "n_total": len(docs_by_id),
        "n_evaluated": len(scores),
        "n_skipped": skipped,
        "n_cleaned": cleaned_count,
        f"avg_{score_key}": avg,
    }
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] cleaned jsonl -> {output_path}")
    print(f"[OK] summary       -> {overall_path}")


if __name__ == "__main__":
    main()
