import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from scripts.asr_eval_utils import compute_wer, normalize_text_for_wer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q4 lattice-based evaluation from Question 4 task CSV.")
    parser.add_argument("--task_csv", type=str, default="project/data/Question 4 - Task.csv")
    parser.add_argument("--output_dir", type=str, default="project/results/q4_task")
    parser.add_argument("--agreement_threshold", type=int, default=3)
    return parser.parse_args()


def read_task_rows(path: Path) -> Tuple[List[Dict], List[str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = reader.fieldnames or []

    model_cols = [h for h in headers if h and h.strip().lower().startswith("model")]
    return rows, model_cols


def align_ref_hyp(ref: Sequence[str], hyp: Sequence[str]) -> List[Tuple[str, int, int]]:
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


def build_lattice(reference: str, hyps: List[str], agreement_threshold: int) -> List[Dict]:
    ref = normalize_text_for_wer(reference).split()
    bins = [{"alts": {tok}} for tok in ref]

    pos_votes: Dict[int, Dict[str, int]] = {i: {} for i in range(len(ref))}
    ins_votes: Dict[int, Dict[str, int]] = {}

    for hyp_text in hyps:
        hyp = normalize_text_for_wer(hyp_text).split()
        ops = align_ref_hyp(ref, hyp)
        for op, i_ref, i_hyp in ops:
            if op in {"eq", "sub"} and 0 <= i_ref < len(ref) and 0 <= i_hyp < len(hyp):
                tok = hyp[i_hyp]
                pos_votes[i_ref][tok] = pos_votes[i_ref].get(tok, 0) + 1
            elif op == "ins" and 0 <= i_hyp < len(hyp):
                gap = i_ref
                if gap not in ins_votes:
                    ins_votes[gap] = {}
                tok = hyp[i_hyp]
                ins_votes[gap][tok] = ins_votes[gap].get(tok, 0) + 1

    for i in range(len(ref)):
        for tok, cnt in pos_votes[i].items():
            if cnt >= agreement_threshold:
                bins[i]["alts"].add(tok)

    out: List[Dict] = []
    for gap in range(len(ref) + 1):
        if gap in ins_votes:
            trusted = [tok for tok, cnt in ins_votes[gap].items() if cnt >= agreement_threshold]
            if trusted:
                s = set(trusted)
                s.add("")
                out.append({"alts": s})
        if gap < len(ref):
            out.append(bins[gap])

    return out


def lattice_wer_for_hyp(lattice: List[Dict], hyp_text: str) -> Tuple[int, int, int, int]:
    hyp = normalize_text_for_wer(hyp_text).split()
    n, m = len(lattice), len(hyp)
    inf = 10**9
    dp = [[(inf, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]
    dp[0][0] = (0, 0, 0, 0)

    for i in range(1, n + 1):
        prev = dp[i - 1][0]
        del_cost = 0 if "" in lattice[i - 1]["alts"] else 1
        dp[i][0] = (prev[0] + del_cost, prev[1], prev[2] + del_cost, prev[3])

    for j in range(1, m + 1):
        prev = dp[0][j - 1]
        dp[0][j] = (prev[0] + 1, prev[1], prev[2], prev[3] + 1)

    for i in range(1, n + 1):
        alts = lattice[i - 1]["alts"]
        for j in range(1, m + 1):
            tok = hyp[j - 1]

            a = dp[i - 1][j - 1]
            sub = 0 if tok in alts else 1
            c1 = (a[0] + sub, a[1] + sub, a[2], a[3])

            b = dp[i - 1][j]
            dcost = 0 if "" in alts else 1
            c2 = (b[0] + dcost, b[1], b[2] + dcost, b[3])

            c = dp[i][j - 1]
            c3 = (c[0] + 1, c[1], c[2], c[3] + 1)

            dp[i][j] = min([c1, c2, c3], key=lambda x: (x[0], x[1] + x[2] + x[3]))

    _, s, d, ins = dp[n][m]
    ref_words = max(1, sum(1 for b in lattice if "" not in b["alts"]))
    return s, d, ins, ref_words


def main() -> None:
    args = parse_args()
    rows, model_cols = read_task_rows(Path(args.task_csv))
    if not rows:
        raise RuntimeError("No rows found in task CSV.")
    if not model_cols:
        raise RuntimeError("No model columns found in task CSV.")

    refs_by_model = {m: [] for m in model_cols}
    preds_by_model = {m: [] for m in model_cols}
    lattice_acc = {m: {"s": 0, "d": 0, "i": 0, "n": 0} for m in model_cols}
    lattice_preview: List[Dict] = []

    for idx, row in enumerate(rows):
        ref = str(row.get("Human", "")).strip()
        if not ref:
            continue

        hyps = [str(row.get(m, "")).strip() for m in model_cols]
        lattice = build_lattice(ref, hyps, args.agreement_threshold)

        if len(lattice_preview) < 5:
            lattice_preview.append(
                {
                    "row_index": idx,
                    "reference": ref,
                    "bins": [sorted(list(b["alts"])) for b in lattice],
                }
            )

        for m in model_cols:
            pred = str(row.get(m, "")).strip()
            refs_by_model[m].append(ref)
            preds_by_model[m].append(pred)

            s, d, ins, n = lattice_wer_for_hyp(lattice, pred)
            lattice_acc[m]["s"] += s
            lattice_acc[m]["d"] += d
            lattice_acc[m]["i"] += ins
            lattice_acc[m]["n"] += n

    summary: List[Dict] = []
    for m in model_cols:
        std_wer, _ = compute_wer(refs_by_model[m], preds_by_model[m])
        s = lattice_acc[m]["s"]
        d = lattice_acc[m]["d"]
        ins = lattice_acc[m]["i"]
        n = max(1, lattice_acc[m]["n"])
        lat_wer = (s + d + ins) / n
        summary.append(
            {
                "model": m,
                "samples": len(refs_by_model[m]),
                "standard_wer": round(std_wer, 6),
                "lattice_wer": round(lat_wer, 6),
                "absolute_delta": round(lat_wer - std_wer, 6),
                "agreement_threshold": args.agreement_threshold,
            }
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "q4_task_lattice_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (out / "q4_task_lattice_preview.json").open("w", encoding="utf-8") as f:
        json.dump(lattice_preview, f, ensure_ascii=False, indent=2)

    with (out / "q4_task_lattice_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "samples", "standard_wer", "lattice_wer", "absolute_delta", "agreement_threshold"],
        )
        writer.writeheader()
        writer.writerows(summary)

    print(json.dumps({"samples": len(rows), "models": model_cols, "output_dir": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
