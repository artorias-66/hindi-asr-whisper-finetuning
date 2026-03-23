import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from scripts.asr_eval_utils import WERBreakdown, _levenshtein_alignment, compute_wer, normalize_text_for_wer, write_csv, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ASR prediction errors and generate taxonomy artifacts.")
    parser.add_argument(
        "--predictions_jsonl",
        type=str,
        default="project/results/eval_finetuned_v2/predictions_fleurs_hi_test.jsonl",
    )
    parser.add_argument("--output_dir", type=str, default="project/results/error_analysis_v2")
    parser.add_argument("--sample_size", type=int, default=30)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def has_consecutive_repeat(words: List[str]) -> bool:
    if len(words) < 2:
        return False
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            return True
    return False


def devanagari_ratio(text: str) -> float:
    if not text:
        return 0.0
    chars = [ch for ch in text if ch.strip()]
    if not chars:
        return 0.0
    dev = sum(1 for ch in chars if "\u0900" <= ch <= "\u097F")
    return dev / len(chars)


def categorize_error(ref: str, pred: str, b: WERBreakdown) -> str:
    ref_words = normalize_text_for_wer(ref).split()
    pred_words = normalize_text_for_wer(pred).split()

    if not pred_words and ref_words:
        return "empty_or_near_empty_prediction"

    if has_consecutive_repeat(pred_words):
        return "repetition_hallucination"

    if ref_words and devanagari_ratio(ref) > 0.6 and devanagari_ratio(pred) < 0.3:
        return "script_or_language_mismatch"

    if b.insertions > b.deletions and b.insertions > b.substitutions:
        return "insertion_dominant"

    if b.deletions > b.insertions and b.deletions > b.substitutions:
        return "deletion_dominant"

    if b.substitutions >= b.insertions and b.substitutions >= b.deletions:
        return "substitution_dominant"

    return "mixed_errors"


def dedupe_consecutive_words(text: str) -> str:
    words = normalize_text_for_wer(text).split()
    if not words:
        return ""

    cleaned = [words[0]]
    for w in words[1:]:
        if w != cleaned[-1]:
            cleaned.append(w)
    return " ".join(cleaned)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(Path(args.predictions_jsonl))

    analyzed: List[Dict] = []
    type_counter: Counter = Counter()

    for row in rows:
        ref = str(row.get("reference", ""))
        pred = str(row.get("prediction", ""))
        ref_words = normalize_text_for_wer(ref).split()
        pred_words = normalize_text_for_wer(pred).split()
        b = _levenshtein_alignment(ref_words, pred_words)
        denom = max(1, b.ref_words)
        wer = (b.substitutions + b.deletions + b.insertions) / denom
        err_type = categorize_error(ref, pred, b)
        type_counter[err_type] += 1
        analyzed.append(
            {
                "id": row.get("id", ""),
                "reference": ref,
                "prediction": pred,
                "wer": round(wer, 6),
                "substitutions": b.substitutions,
                "deletions": b.deletions,
                "insertions": b.insertions,
                "ref_words": b.ref_words,
                "error_type": err_type,
            }
        )

    error_rows = [r for r in analyzed if r["wer"] > 0]
    error_rows_sorted = sorted(error_rows, key=lambda x: x["wer"], reverse=True)
    sample_rows = error_rows_sorted[: max(25, args.sample_size)]

    write_jsonl(sample_rows, output_dir / "sample_30_errors.jsonl")
    write_csv(sample_rows, output_dir / "sample_30_errors.csv")

    taxonomy_rows = [
        {"error_type": k, "count": v, "percent": round(100.0 * v / max(1, len(error_rows)), 2)}
        for k, v in type_counter.most_common()
    ]
    write_csv(taxonomy_rows, output_dir / "error_taxonomy_counts.csv")

    top3 = [x["error_type"] for x in taxonomy_rows[:3]]

    target = [r for r in sample_rows if r["error_type"] == "repetition_hallucination"]
    if not target:
        target = sample_rows[:25]

    refs_before = [r["reference"] for r in target]
    preds_before = [r["prediction"] for r in target]
    before_wer, _ = compute_wer(refs_before, preds_before)

    fixed_preds = [dedupe_consecutive_words(p) for p in preds_before]
    after_wer, _ = compute_wer(refs_before, fixed_preds)

    fix_rows = []
    for src, fixed in zip(target, fixed_preds):
        fix_rows.append(
            {
                "id": src["id"],
                "reference": src["reference"],
                "prediction_before": src["prediction"],
                "prediction_after": fixed,
                "error_type": src["error_type"],
            }
        )

    write_jsonl(fix_rows, output_dir / "fix_repetition_target_subset.jsonl")

    summary = {
        "total_predictions": len(rows),
        "total_error_samples": len(error_rows),
        "sample_size": len(sample_rows),
        "top_3_error_types": top3,
        "target_subset_size": len(target),
        "target_subset_wer_before": round(before_wer, 6),
        "target_subset_wer_after": round(after_wer, 6),
        "target_subset_wer_delta": round(after_wer - before_wer, 6),
    }

    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
