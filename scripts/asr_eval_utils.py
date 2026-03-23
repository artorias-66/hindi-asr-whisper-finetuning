import csv
import json
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass
class WERBreakdown:
    substitutions: int
    deletions: int
    insertions: int
    ref_words: int


def set_seed(seed: int) -> None:
    random.seed(seed)


def normalize_text_for_wer(text: str) -> str:
    text = text or ""
    text = text.replace("\ufeff", " ")

    chars: List[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in {"Cc", "Cf"} and ch not in {"\t", "\n", "\r"}:
            continue
        chars.append(ch)

    cleaned = "".join(chars)
    cleaned = re.sub(r"[\t\n\r]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Remove punctuation while preserving letters/digits in all scripts.
    cleaned = " ".join("".join(ch for ch in token if ch.isalnum()) for token in cleaned.split())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _levenshtein_alignment(ref_words: Sequence[str], hyp_words: Sequence[str]) -> WERBreakdown:
    n = len(ref_words)
    m = len(hyp_words)

    # dp[i][j] = (cost, s, d, i)
    dp: List[List[Tuple[int, int, int, int]]] = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        prev = dp[i - 1][0]
        dp[i][0] = (prev[0] + 1, prev[1], prev[2] + 1, prev[3])
    for j in range(1, m + 1):
        prev = dp[0][j - 1]
        dp[0][j] = (prev[0] + 1, prev[1], prev[2], prev[3] + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue

            sub = dp[i - 1][j - 1]
            delete = dp[i - 1][j]
            insert = dp[i][j - 1]

            candidates = [
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                (insert[0] + 1, insert[1], insert[2], insert[3] + 1),
            ]
            dp[i][j] = min(candidates, key=lambda x: (x[0], x[1] + x[2] + x[3]))

    _, s, d, ins = dp[n][m]
    return WERBreakdown(substitutions=s, deletions=d, insertions=ins, ref_words=n)


def compute_wer(references: Sequence[str], predictions: Sequence[str]) -> Tuple[float, WERBreakdown]:
    total = WERBreakdown(substitutions=0, deletions=0, insertions=0, ref_words=0)

    for ref, hyp in zip(references, predictions):
        ref_words = normalize_text_for_wer(ref).split()
        hyp_words = normalize_text_for_wer(hyp).split()
        b = _levenshtein_alignment(ref_words, hyp_words)
        total.substitutions += b.substitutions
        total.deletions += b.deletions
        total.insertions += b.insertions
        total.ref_words += b.ref_words

    if total.ref_words == 0:
        return 0.0, total

    wer = (total.substitutions + total.deletions + total.insertions) / total.ref_words
    return wer, total


def write_jsonl(records: Iterable[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(rows: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    keys = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
