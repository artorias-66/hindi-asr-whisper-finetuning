import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


NUM_DIRECT = {
    "शून्य": 0,
    "एक": 1,
    "दो": 2,
    "तीन": 3,
    "चार": 4,
    "पांच": 5,
    "पाँच": 5,
    "छह": 6,
    "सात": 7,
    "आठ": 8,
    "नौ": 9,
    "दस": 10,
    "ग्यारह": 11,
    "बारह": 12,
    "तेरह": 13,
    "चौदह": 14,
    "पंद्रह": 15,
    "पन्द्रह": 15,
    "सोलह": 16,
    "सत्रह": 17,
    "अठारह": 18,
    "उन्नीस": 19,
    "बीस": 20,
    "इक्कीस": 21,
    "बाईस": 22,
    "तेइस": 23,
    "तेईस": 23,
    "चौबीस": 24,
    "पच्चीस": 25,
    "छब्बीस": 26,
    "सत्ताईस": 27,
    "अट्ठाईस": 28,
    "उनतीस": 29,
    "तीस": 30,
    "इकतीस": 31,
    "बत्तीस": 32,
    "तैंतीस": 33,
    "तैंतीस": 33,
    "चौंतीस": 34,
    "पैंतीस": 35,
    "छत्तीस": 36,
    "सैंतीस": 37,
    "अड़तीस": 38,
    "उनतालीस": 39,
    "चालीस": 40,
    "इकतालीस": 41,
    "बयालीस": 42,
    "तैंतालीस": 43,
    "चवालीस": 44,
    "पैंतालीस": 45,
    "छियालीस": 46,
    "सैंतालीस": 47,
    "अड़तालीस": 48,
    "उनचास": 49,
    "पचास": 50,
    "इक्यावन": 51,
    "बावन": 52,
    "तिरेपन": 53,
    "चौवन": 54,
    "पचपन": 55,
    "छप्पन": 56,
    "सत्तावन": 57,
    "अट्ठावन": 58,
    "उनसठ": 59,
    "साठ": 60,
    "इकसठ": 61,
    "बासठ": 62,
    "तिरसठ": 63,
    "चौंसठ": 64,
    "पैंसठ": 65,
    "छियासठ": 66,
    "सड़सठ": 67,
    "अड़सठ": 68,
    "उनहत्तर": 69,
    "सत्तर": 70,
    "इकहत्तर": 71,
    "बहत्तर": 72,
    "तिहत्तर": 73,
    "चौहत्तर": 74,
    "पचहत्तर": 75,
    "छिहत्तर": 76,
    "सतहत्तर": 77,
    "अठहत्तर": 78,
    "उन्यासी": 79,
    "अस्सी": 80,
    "इक्यासी": 81,
    "बयासी": 82,
    "तिरासी": 83,
    "चौरासी": 84,
    "पचासी": 85,
    "छियासी": 86,
    "सत्तासी": 87,
    "अट्ठासी": 88,
    "नवासी": 89,
    "नब्बे": 90,
    "इक्यानवे": 91,
    "बानवे": 92,
    "तिरानवे": 93,
    "चौरानवे": 94,
    "पचानवे": 95,
    "छियानवे": 96,
    "सत्तानवे": 97,
    "अट्ठानवे": 98,
    "निन्यानवे": 99,
}

MULTIPLIERS = {
    "सौ": 100,
    "हज़ार": 1000,
    "हजार": 1000,
    "लाख": 100000,
    "करोड़": 10000000,
}

IDIOM_PATTERNS = [
    re.compile(r"^दो[- ]चार$"),
    re.compile(r"^चार[- ]छह$"),
    re.compile(r"^एक[- ]दो$"),
]

EN_HINTS = {
    "इंटरव्यू",
    "जॉब",
    "कंप्यूटर",
    "कम्प्यूटर",
    "प्रॉब्लम",
    "प्रोब्लम",
    "टेस्ट",
    "प्लान",
    "प्रोजेक्ट",
    "टीम",
    "कोड",
    "सॉफ्टवेयर",
    "सॉफ़्टवेयर",
    "हार्डवेयर",
    "फाइल",
    "फाइलें",
    "रिपोर्ट",
    "डेटा",
    "मॉडल",
    "मोडल",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q2 cleanup pipeline: number normalization + EN tagging")
    parser.add_argument("--raw_jsonl", type=str, default="project/results/q2/raw_asr_pretrained.jsonl")
    parser.add_argument("--output_dir", type=str, default="project/results/q2")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_number_word(token: str) -> bool:
    t = token.strip(".,!?;:।")
    return t in NUM_DIRECT or t in MULTIPLIERS


def parse_number_tokens(tokens: List[str]) -> Tuple[bool, int]:
    total = 0
    current = 0
    used = False

    for tok in tokens:
        t = tok.strip(".,!?;:।")
        if t in NUM_DIRECT:
            current += NUM_DIRECT[t]
            used = True
        elif t in MULTIPLIERS:
            used = True
            mult = MULTIPLIERS[t]
            if mult == 100:
                if current == 0:
                    current = 1
                current *= 100
            else:
                if current == 0:
                    current = 1
                total += current * mult
                current = 0
        else:
            return False, 0

    if not used:
        return False, 0
    return True, total + current


def normalize_numbers(text: str) -> Tuple[str, List[Dict], List[Dict]]:
    tokens = text.split()
    out: List[str] = []
    changes: List[Dict] = []
    edges: List[Dict] = []

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        # idiom guard
        local = tok.strip(".,!?;:।")
        if any(p.match(local) for p in IDIOM_PATTERNS):
            edges.append({"span": local, "decision": "kept_as_idiom"})
            out.append(tok)
            i += 1
            continue

        if not is_number_word(tok):
            out.append(tok)
            i += 1
            continue

        j = i
        span_tokens: List[str] = []
        while j < len(tokens) and is_number_word(tokens[j]):
            span_tokens.append(tokens[j])
            j += 1

        ok, value = parse_number_tokens(span_tokens)
        span_text = " ".join(span_tokens)
        if ok and span_text:
            out.append(str(value))
            changes.append({"before": span_text, "after": str(value)})
            i = j
        else:
            out.extend(span_tokens)
            i = j

    return " ".join(out), changes, edges


def is_probably_english_word(token: str) -> bool:
    t = token.strip(".,!?;:।")
    if not t:
        return False
    if re.search(r"[A-Za-z]", t):
        return True
    return t in EN_HINTS


def tag_english_words(text: str) -> Tuple[str, List[str]]:
    tagged: List[str] = []
    hits: List[str] = []
    for token in text.split():
        if is_probably_english_word(token):
            tagged.append(f"[EN]{token}[/EN]")
            hits.append(token)
        else:
            tagged.append(token)
    return " ".join(tagged), hits


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.raw_jsonl))

    out_rows: List[Dict] = []
    number_examples: List[Dict] = []
    edge_examples: List[Dict] = []

    for row in rows:
        raw = str(row.get("raw_prediction", ""))
        num_text, num_changes, edges = normalize_numbers(raw)
        tagged_text, en_hits = tag_english_words(num_text)

        if num_changes and len(number_examples) < 8:
            number_examples.append(
                {
                    "id": row.get("id"),
                    "raw": raw,
                    "after_number_norm": num_text,
                    "changes": num_changes,
                }
            )

        if edges and len(edge_examples) < 5:
            edge_examples.append(
                {
                    "id": row.get("id"),
                    "raw": raw,
                    "edge_decisions": edges,
                }
            )

        out_rows.append(
            {
                "id": row.get("id"),
                "audio": row.get("audio"),
                "reference": row.get("reference"),
                "raw_prediction": raw,
                "after_number_norm": num_text,
                "english_tagged": tagged_text,
                "english_hits": en_hits,
                "number_changes": num_changes,
                "edge_decisions": edges,
            }
        )

    output_dir = Path(args.output_dir)
    write_jsonl(out_rows, output_dir / "q2_cleaned_outputs.jsonl")
    write_jsonl(number_examples, output_dir / "q2_number_examples.jsonl")
    write_jsonl(edge_examples, output_dir / "q2_number_edge_cases.jsonl")

    summary = {
        "samples": len(out_rows),
        "with_number_changes": sum(1 for r in out_rows if r["number_changes"]),
        "with_english_hits": sum(1 for r in out_rows if r["english_hits"]),
        "output": str(output_dir / "q2_cleaned_outputs.jsonl"),
    }

    with (output_dir / "q2_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
