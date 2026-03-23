import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from scripts.asr_eval_utils import compute_wer, normalize_text_for_wer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q4 lattice-based evaluation across multiple ASR model outputs.")
    parser.add_argument(
        "--model_prediction_files",
        nargs="+",
        default=[
            "project/results/eval_baseline_full/predictions_fleurs_hi_test.jsonl",
            "project/results/eval_finetuned_short/predictions_fleurs_hi_test.jsonl",
            "project/results/eval_finetuned_v2/predictions_fleurs_hi_test.jsonl",
            "project/results/eval_finetuned_v2_dedup/predictions_fleurs_hi_test.jsonl",
            "project/results/eval_baseline_dedup/predictions_fleurs_hi_test.jsonl",
        ],
    )
    parser.add_argument("--output_dir", type=str, default="project/results/q4")
    parser.add_argument("--agreement_threshold", type=int, default=3)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def align_ref_hyp(ref: Sequence[str], hyp: Sequence[str]) -> List[Tuple[str, int, int]]:
    # returns operations with indices: (op, i_ref, i_hyp)
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

    ops: List[Tuple[str, int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(("eq", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", i - 1, j))
            i -= 1
        else:
            ops.append(("ins", i, j - 1))
            j -= 1

    ops.reverse()
    return ops


def lattice_cost(hyp: Sequence[str], bins: List[Dict]) -> Tuple[int, int, int, int]:
    # DP against bins with optional epsilon bins represented by "" in alternatives
    n, m = len(bins), len(hyp)
    dp = [[(10**9, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]
    dp[0][0] = (0, 0, 0, 0)

    for i in range(1, n + 1):
        prev = dp[i - 1][0]
        del_cost = 0 if "" in bins[i - 1]["alternatives"] else 1
        dp[i][0] = (prev[0] + del_cost, prev[1], prev[2] + del_cost, prev[3])

    for j in range(1, m + 1):
        prev = dp[0][j - 1]
        dp[0][j] = (prev[0] + 1, prev[1], prev[2], prev[3] + 1)

    for i in range(1, n + 1):
        alts = bins[i - 1]["alternatives"]
        for j in range(1, m + 1):
            tok = hyp[j - 1]

            # substitution / match
            prev = dp[i - 1][j - 1]
            sub_penalty = 0 if tok in alts else 1
            cand_sub = (prev[0] + sub_penalty, prev[1] + sub_penalty, prev[2], prev[3])

            # deletion of bin
            prev_del = dp[i - 1][j]
            del_penalty = 0 if "" in alts else 1
            cand_del = (prev_del[0] + del_penalty, prev_del[1], prev_del[2] + del_penalty, prev_del[3])

            # insertion in hyp
            prev_ins = dp[i][j - 1]
            cand_ins = (prev_ins[0] + 1, prev_ins[1], prev_ins[2], prev_ins[3] + 1)

            dp[i][j] = min([cand_sub, cand_del, cand_ins], key=lambda x: (x[0], x[1] + x[2] + x[3]))

    total, s, d, ins = dp[n][m]
    ref_len = max(1, sum(1 for b in bins if "" not in b["alternatives"]))
    return s, d, ins, ref_len


def build_lattice_for_sample(reference: str, model_preds: List[str], agreement_threshold: int) -> List[Dict]:
    ref_tokens = normalize_text_for_wer(reference).split()
    bins = [{"alternatives": {tok}} for tok in ref_tokens]

    # Collect substitutions/equalities and insertion candidates from each model alignment.
    ins_by_gap: Dict[int, Counter] = defaultdict(Counter)
    aligned_tokens_by_pos: Dict[int, Counter] = defaultdict(Counter)

    for pred in model_preds:
        hyp = normalize_text_for_wer(pred).split()
        ops = align_ref_hyp(ref_tokens, hyp)
        for op, i_ref, i_hyp in ops:
            if op in {"eq", "sub"}:
                if 0 <= i_ref < len(ref_tokens) and 0 <= i_hyp < len(hyp):
                    aligned_tokens_by_pos[i_ref][hyp[i_hyp]] += 1
            elif op == "ins":
                gap = i_ref  # insertion before ref index i_ref
                if 0 <= i_hyp < len(hyp):
                    ins_by_gap[gap][hyp[i_hyp]] += 1

    # Add trusted alternatives from model agreement at each reference position.
    for pos, c in aligned_tokens_by_pos.items():
        for tok, cnt in c.items():
            if cnt >= agreement_threshold:
                bins[pos]["alternatives"].add(tok)

    # Insert optional bins for agreed insertions.
    new_bins: List[Dict] = []
    for gap in range(len(ref_tokens) + 1):
        if gap in ins_by_gap:
            trusted_ins = [tok for tok, cnt in ins_by_gap[gap].items() if cnt >= agreement_threshold]
            if trusted_ins:
                alt = set(trusted_ins)
                alt.add("")
                new_bins.append({"alternatives": alt})

        if gap < len(ref_tokens):
            new_bins.append(bins[gap])

    return new_bins


def main() -> None:
    args = parse_args()

    model_files = [Path(p) for p in args.model_prediction_files]
    for p in model_files:
        if not p.exists():
            raise FileNotFoundError(f"Missing model predictions file: {p}")

    model_names = [p.parent.name for p in model_files]
    model_rows = [read_jsonl(p) for p in model_files]

    # Build id-index maps and shared ids.
    maps: List[Dict[str, Dict]] = []
    shared_ids = None
    for rows in model_rows:
        d = {str(r["id"]): r for r in rows}
        maps.append(d)
        ids = set(d.keys())
        shared_ids = ids if shared_ids is None else (shared_ids & ids)

    if not shared_ids:
        raise RuntimeError("No shared sample ids across provided model files.")

    shared_ids_sorted = sorted(shared_ids)

    standard_refs_by_model: Dict[str, List[str]] = {name: [] for name in model_names}
    standard_preds_by_model: Dict[str, List[str]] = {name: [] for name in model_names}

    lattice_stats: Dict[str, Dict[str, int]] = {
        name: {"s": 0, "d": 0, "i": 0, "n": 0} for name in model_names
    }

    lattices_preview: List[Dict] = []

    for idx, sid in enumerate(shared_ids_sorted):
        sample_rows = [m[sid] for m in maps]
        reference = str(sample_rows[0]["reference"])
        preds = [str(r["prediction"]) for r in sample_rows]

        lattice = build_lattice_for_sample(reference, preds, args.agreement_threshold)

        if len(lattices_preview) < 5:
            lattices_preview.append(
                {
                    "id": sid,
                    "reference": reference,
                    "lattice_bins": [sorted(list(b["alternatives"])) for b in lattice],
                }
            )

        for name, row in zip(model_names, sample_rows):
            ref = str(row["reference"])
            pred = str(row["prediction"])

            standard_refs_by_model[name].append(ref)
            standard_preds_by_model[name].append(pred)

            hyp_tokens = normalize_text_for_wer(pred).split()
            s, d, ins, n = lattice_cost(hyp_tokens, lattice)
            lattice_stats[name]["s"] += s
            lattice_stats[name]["d"] += d
            lattice_stats[name]["i"] += ins
            lattice_stats[name]["n"] += n

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict] = []
    for name in model_names:
        standard_wer, _ = compute_wer(standard_refs_by_model[name], standard_preds_by_model[name])
        s = lattice_stats[name]["s"]
        d = lattice_stats[name]["d"]
        ins = lattice_stats[name]["i"]
        n = max(1, lattice_stats[name]["n"])
        lattice_wer = (s + d + ins) / n

        summary_rows.append(
            {
                "model": name,
                "samples": len(shared_ids_sorted),
                "standard_wer": round(standard_wer, 6),
                "lattice_wer": round(lattice_wer, 6),
                "absolute_delta": round(lattice_wer - standard_wer, 6),
                "agreement_threshold": args.agreement_threshold,
            }
        )

    with (out_dir / "q4_lattice_wer_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    with (out_dir / "q4_lattice_preview.json").open("w", encoding="utf-8") as f:
        json.dump(lattices_preview, f, ensure_ascii=False, indent=2)

    # CSV
    import csv

    with (out_dir / "q4_lattice_wer_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "samples", "standard_wer", "lattice_wer", "absolute_delta", "agreement_threshold"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(json.dumps({"samples": len(shared_ids_sorted), "models": model_names, "output_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
