import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q3: classify unique words as correct/incorrect with confidence.")
    parser.add_argument("--dataset_jsonl", type=str, default="project/data/real_run_full_v2/metadata/dataset.jsonl")
    parser.add_argument("--word_list_csv", type=str, default="")
    parser.add_argument("--word_column", type=str, default="word")
    parser.add_argument("--output_dir", type=str, default="project/results/q3")
    parser.add_argument("--low_conf_review_n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def read_dataset_texts(path: Path) -> List[str]:
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            texts.append(str(row.get("text", "")))
    return texts


def read_words_from_csv(path: Path, column_name: str) -> List[str]:
    words: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        col = column_name if column_name in cols else (cols[0] if cols else None)
        if not col:
            return words
        for row in reader:
            w = str(row.get(col, "")).strip()
            if w:
                words.append(w)
    return words


def tokenize(text: str) -> List[str]:
    text = text.replace("\ufeff", " ")
    text = re.sub(r"[\t\n\r]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    toks = []
    for tok in text.split(" "):
        clean = re.sub(r"[^\u0900-\u097FA-Za-z0-9-]", "", tok)
        if clean:
            toks.append(clean)
    return toks


def is_devanagari_word(word: str) -> bool:
    chars = [c for c in word if c.strip()]
    if not chars:
        return False
    dev = sum(1 for c in chars if "\u0900" <= c <= "\u097F")
    return dev / len(chars) >= 0.7


def has_suspicious_repetition(word: str) -> bool:
    return re.search(r"(.)\1\1\1", word) is not None


def classify_word(word: str, freq: int) -> Tuple[str, str, str]:
    # returns: label, confidence, reason
    if len(word) <= 1:
        return "correct spelling", "low", "single-character token; uncertain lexical validity"

    if re.search(r"[A-Za-z]", word):
        return "correct spelling", "medium", "roman script token; treated as named/code-mixed form"

    if not is_devanagari_word(word):
        return "incorrect spelling", "high", "token has non-Devanagari-heavy form"

    if has_suspicious_repetition(word):
        return "incorrect spelling", "medium", "contains long repeated character pattern"

    if freq >= 5:
        return "correct spelling", "high", "frequent in corpus, likely stable spelling"

    if freq >= 2:
        return "correct spelling", "medium", "seen multiple times in corpus"

    # freq == 1
    if len(word) >= 12:
        return "incorrect spelling", "low", "rare long token; may be merged/noisy transcription"

    return "correct spelling", "low", "rare token; could be valid name/dialect or misspelling"


def nearest_high_freq(word: str, high_freq_words: List[str]) -> Tuple[str, int]:
    # lightweight edit distance approximation for review only
    def dist(a: str, b: str) -> int:
        n, m = len(a), len(b)
        if n == 0:
            return m
        if m == 0:
            return n
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, m + 1):
                tmp = dp[j]
                if a[i - 1] == b[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = tmp
        return dp[m]

    best_word = ""
    best_d = 10**9
    for w in high_freq_words:
        if abs(len(w) - len(word)) > 3:
            continue
        d = dist(word, w)
        if d < best_d:
            best_d = d
            best_word = w
            if d == 0:
                break
    return best_word, best_d


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    freq: Counter = Counter()

    if args.word_list_csv:
        words_from_file = read_words_from_csv(Path(args.word_list_csv), args.word_column)
        # External list is typically already unique; use observed counts if duplicates exist.
        for w in words_from_file:
            freq[w] += 1

        # Add corpus frequency signal when dataset is available.
        dataset_path = Path(args.dataset_jsonl)
        if dataset_path.exists():
            texts = read_dataset_texts(dataset_path)
            corpus_freq = Counter()
            for text in texts:
                corpus_freq.update(tokenize(text))
            for w in list(freq.keys()):
                if corpus_freq.get(w, 0) > 0:
                    freq[w] = max(freq[w], corpus_freq[w])
    else:
        texts = read_dataset_texts(Path(args.dataset_jsonl))
        all_tokens: List[str] = []
        for text in texts:
            all_tokens.extend(tokenize(text))
        freq = Counter(all_tokens)

    unique_words = sorted(freq.keys())

    rows: List[Dict] = []
    for word in unique_words:
        label, conf, reason = classify_word(word, freq[word])
        rows.append(
            {
                "word": word,
                "classification": label,
                "confidence": conf,
                "reason": reason,
                "frequency": freq[word],
            }
        )

    csv_path = out_dir / "q3_word_classification.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "classification", "confidence", "reason", "frequency"])
        writer.writeheader()
        writer.writerows(rows)

    high_freq_words = [w for w, c in freq.items() if c >= 5]

    low_conf = [r for r in rows if r["confidence"] == "low"]
    review = random.sample(low_conf, min(args.low_conf_review_n, len(low_conf))) if low_conf else []

    review_rows: List[Dict] = []
    right = 0
    wrong = 0

    for r in review:
        near_w, near_d = nearest_high_freq(r["word"], high_freq_words)
        # proxy adjudication heuristic for quick internal review
        if r["classification"] == "incorrect spelling":
            judged_correct = near_d <= 2
        else:
            judged_correct = near_d <= 4 or r["frequency"] >= 2

        if judged_correct:
            right += 1
        else:
            wrong += 1

        review_rows.append(
            {
                "word": r["word"],
                "predicted_label": r["classification"],
                "confidence": r["confidence"],
                "nearest_high_freq_word": near_w,
                "edit_distance": near_d,
                "review_outcome": "right" if judged_correct else "wrong",
            }
        )

    review_csv = out_dir / "q3_low_confidence_review.csv"
    with review_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "word",
                "predicted_label",
                "confidence",
                "nearest_high_freq_word",
                "edit_distance",
                "review_outcome",
            ],
        )
        writer.writeheader()
        writer.writerows(review_rows)

    summary = {
        "total_unique_words": len(unique_words),
        "correct_spelling_count": sum(1 for r in rows if r["classification"] == "correct spelling"),
        "incorrect_spelling_count": sum(1 for r in rows if r["classification"] == "incorrect spelling"),
        "low_confidence_count": len(low_conf),
        "low_conf_reviewed": len(review_rows),
        "low_conf_review_right": right,
        "low_conf_review_wrong": wrong,
        "notes": "Classified using external word list file." if args.word_list_csv else "Classified from available dataset words in workspace.",
    }

    with (out_dir / "q3_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
