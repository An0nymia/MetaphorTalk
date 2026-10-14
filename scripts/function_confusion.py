#!/usr/bin/env python3
"""
Extract Function Confusion Matrix and Qualitative Case Studies.

This script:
1. Aligns creative metaphor spans across human annotator pairs (using Hungarian matching with Coverage >= 0.4/0.5).
2. Computes the symmetric 8x8 communicative function confusion/co-occurrence matrix.
3. Automatically identifies representative agreement and disagreement cases for each
   of the 8 communicative functions.
4. Generates publication-ready outputs: CSV, LaTeX table, and Markdown qualitative summary.

Usage:
    python scripts/function_confusion.py \\
        --ann1 data/gold/main_annotator1.json \\
        --ann2 data/gold/main_annotator2.json \\
        --ann3 data/gold/main_annotator3.json \\
        --gold data/gold/gold_standard_annotations.json \\
        --out-dir data/gold
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from scipy.optimize import linear_sum_assignment

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

VALID_INTENTIONS = [
    "Artistic metaphor",
    "Visualization",
    "Persuasiveness",
    "Explanation",
    "Argumentative metaphor",
    "Social interaction",
    "Humour",
    "Heuristic reasoning",
]

INTENTION_SHORT_LABELS = {
    "Artistic metaphor": "Art",
    "Visualization": "Vis",
    "Persuasiveness": "Per",
    "Explanation": "Exp",
    "Argumentative metaphor": "Arg",
    "Social interaction": "Soc",
    "Humour": "Hum",
    "Heuristic reasoning": "Heu",
}


def load_annotator_records(file_path: Path) -> Dict[str, Dict[str, Any]]:
    """Loads utterances and annotations from MetAnno export JSON."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    utts = data.get("utterances", {})
    anns = data.get("annotations", {})

    records = {}
    for ann_id, ann in anns.items():
        uid = ann.get("utterance_id")
        u = utts.get(uid, {})
        text = u.get("text", "")
        spans = []
        for m in ann.get("metaphors", []):
            start = int(m["start"])
            end = int(m["end"])
            span_text = m.get("text", text[start:end]) if text else m.get("text", "")
            raw_funcs = m.get("intentions", [])
            if isinstance(raw_funcs, str):
                raw_funcs = [x.strip() for x in raw_funcs.split(";") if x.strip()]
            norm_intents = [cat for cat in raw_funcs if cat in VALID_INTENTIONS]
            spans.append({
                "start": start,
                "end": end,
                "text": span_text,
                "lexicalized": bool(m.get("lexicalized", False)),
                "intentions": norm_intents,
            })
        records[uid] = {
            "text": text,
            "speaker": u.get("speaker", "Unknown"),
            "has_metaphor": bool(ann.get("metaphor_present", len(spans) > 0)),
            "spans": spans,
            "is_completed": bool(ann.get("identification_completed", True)),
        }
    return records


def span_coverage(gold_span: Tuple[int, int], pred_span: Tuple[int, int]) -> float:
    s_g, e_g = gold_span
    s_p, e_p = pred_span
    intersection = max(0, min(e_p, e_g) - max(s_p, s_g))
    g_len = max(1, e_g - s_g)
    return intersection / g_len


def align_spans_one_to_one(
    spans_a: List[Dict[str, Any]],
    spans_b: List[Dict[str, Any]],
    threshold: float = 0.4,
) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
    """Hungarian optimal bipartite matching between two annotator span lists."""
    if not spans_a or not spans_b:
        return []

    n_a = len(spans_a)
    n_b = len(spans_b)
    score_matrix = np.zeros((n_a, n_b), dtype=float)

    for i, sa in enumerate(spans_a):
        span_a_tup = (sa["start"], sa["end"])
        for j, sb in enumerate(spans_b):
            span_b_tup = (sb["start"], sb["end"])
            cov_a = span_coverage(span_a_tup, span_b_tup)
            cov_b = span_coverage(span_b_tup, span_a_tup)
            score_matrix[i, j] = max(cov_a, cov_b)

    cost_matrix = -score_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched = []
    for r, c in zip(row_ind, col_ind):
        score = score_matrix[r, c]
        if score >= threshold:
            matched.append((spans_a[r], spans_b[c], score))

    return matched


def compute_function_confusion_and_extract_cases(
    ann_records: List[Dict[str, Dict[str, Any]]],
    labels: List[str],
    shared_turn_ids: List[str],
    gold_records: Optional[Dict[str, Dict[str, Any]]] = None,
    threshold: float = 0.4,
) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """Computes symmetric confusion matrix and collects top agreement / disagreement cases."""
    conf_matrix = pd.DataFrame(0, index=VALID_INTENTIONS, columns=VALID_INTENTIONS, dtype=int)
    agreements_by_cat = defaultdict(list)
    disagreements_by_cat = defaultdict(list)

    n_raters = len(ann_records)
    for i in range(n_raters):
        for j in range(i + 1, n_raters):
            h1, h2 = labels[i], labels[j]
            rec1, rec2 = ann_records[i], ann_records[j]

            for uid in shared_turn_ids:
                s1_list = rec1[uid]["spans"]
                s2_list = rec2[uid]["spans"]
                text = rec1[uid]["text"]
                speaker = rec1[uid].get("speaker", "Unknown")
                if gold_records and uid in gold_records:
                    speaker = gold_records[uid].get("speaker", speaker)

                matched = align_spans_one_to_one(s1_list, s2_list, threshold=threshold)
                for s1, s2, score in matched:
                    int1 = set(s1.get("intentions", []))
                    int2 = set(s2.get("intentions", []))

                    # Update symmetric confusion matrix
                    for c1 in int1:
                        for c2 in int2:
                            conf_matrix.loc[c1, c2] += 1
                            conf_matrix.loc[c2, c1] += 1

                    # Track Agreements
                    common = int1 & int2
                    for c in common:
                        agreements_by_cat[c].append({
                            "turn_id": uid,
                            "speaker": speaker,
                            "span": s1["text"],
                            "text": text,
                            "r1": h1,
                            "r2": h2,
                            "int1": list(int1),
                            "int2": list(int2),
                        })

                    # Track Disagreements (Rater 1 marked c, but Rater 2 did not)
                    for c in int1 - int2:
                        disagreements_by_cat[c].append({
                            "turn_id": uid,
                            "speaker": speaker,
                            "span1": s1["text"],
                            "span2": s2["text"],
                            "text": text,
                            "rater_pos": h1,
                            "rater_neg": h2,
                            "pos_int": list(int1),
                            "neg_int": list(int2) if int2 else ["(Conventional / Lexicalized)"],
                        })

                    # Track reverse Disagreements (Rater 2 marked c, but Rater 1 did not)
                    for c in int2 - int1:
                        disagreements_by_cat[c].append({
                            "turn_id": uid,
                            "speaker": speaker,
                            "span1": s2["text"],
                            "span2": s1["text"],
                            "text": text,
                            "rater_pos": h2,
                            "rater_neg": h1,
                            "pos_int": list(int2),
                            "neg_int": list(int1) if int1 else ["(Conventional / Lexicalized)"],
                        })

    return conf_matrix, agreements_by_cat, disagreements_by_cat


def generate_latex_table(conf_matrix: pd.DataFrame) -> str:
    """Formats single-column and double-column LaTeX tables."""
    cols_short = [INTENTION_SHORT_LABELS[c] for c in VALID_INTENTIONS]
    
    lines = [
        "% --- SINGLE-COLUMN CONFUSION MATRIX ---",
        "\\begin{table}[th]",
        "\\centering",
        "\\footnotesize",
        "\\caption{Human inter-annotator co-occurrence and confusion matrix across 1-to-1 aligned creative spans. Diagonal entries (bold) denote category agreement; off-diagonal entries reflect multi-label co-assignment or boundary confusion.}",
        "\\label{tab:func_confusion_single_col}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        "\\textbf{Category} & " + " & ".join([f"\\textbf{{{s}}}" for s in cols_short]) + " \\\\",
        "\\midrule",
    ]

    for cat in VALID_INTENTIONS:
        short_row = INTENTION_SHORT_LABELS[cat]
        row_vals = []
        for col_cat in VALID_INTENTIONS:
            val = conf_matrix.loc[cat, col_cat]
            if cat == col_cat:
                row_vals.append(f"\\textbf{{{val}}}")
            else:
                row_vals.append(str(val))
        lines.append(f"\\textbf{{{cat} ({short_row})}} & " + " & ".join(row_vals) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Extract Communicative Function Confusion Matrix and Qualitative Examples.")
    parser.add_argument("--ann1", type=str, required=True, help="Path to Annotator 1 JSON.")
    parser.add_argument("--ann2", type=str, required=True, help="Path to Annotator 2 JSON.")
    parser.add_argument("--ann3", type=str, required=True, help="Path to Annotator 3 JSON.")
    parser.add_argument("--gold", type=str, default=None, help="Optional path to Gold Standard JSON.")
    parser.add_argument("--threshold", type=float, default=0.4, help="Hungarian coverage threshold (default: 0.4).")
    parser.add_argument("--out-dir", type=str, default=".", help="Directory to save generated artifacts.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading human annotation files...")
    a1_rec = load_annotator_records(Path(args.ann1))
    a2_rec = load_annotator_records(Path(args.ann2))
    a3_rec = load_annotator_records(Path(args.ann3))
    gold_rec = load_annotator_records(Path(args.gold)) if args.gold and Path(args.gold).exists() else None

    # Shared completed turns
    shared_turns = sorted(list(set(a1_rec.keys()) & set(a2_rec.keys()) & set(a3_rec.keys())))
    if gold_rec:
        shared_turns = [uid for uid in shared_turns if uid in gold_rec]
    logger.info(f"Identified {len(shared_turns)} shared turns across annotators.")

    conf_matrix, agreements, disagreements = compute_function_confusion_and_extract_cases(
        ann_records=[a1_rec, a2_rec, a3_rec],
        labels=["A1", "A2", "A3"],
        shared_turn_ids=shared_turns,
        gold_records=gold_rec,
        threshold=args.threshold,
    )

    # 1. Save CSV Matrix
    csv_path = out_dir / "function_confusion_matrix.csv"
    conf_matrix.to_csv(csv_path)
    logger.info(f"Saved confusion matrix CSV to: {csv_path}")

    # 2. Save LaTeX Table
    latex_str = generate_latex_table(conf_matrix)
    latex_path = out_dir / "function_confusion_matrix.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    logger.info(f"Saved LaTeX table to: {latex_path}")

    # 3. Print Matrix Summary
    print("\n" + "=" * 90)
    print("HUMAN INTER-ANNOTATOR CONFUSION MATRIX (8x8 SYMMETRIC)")
    print("=" * 90)
    print(conf_matrix)
    print("=" * 90 + "\n")

    # 4. Save JSON of extracted case studies
    cases_summary = {}
    for cat in VALID_INTENTIONS:
        cases_summary[cat] = {
            "agreements_count": len(agreements[cat]),
            "disagreements_count": len(disagreements[cat]),
            "top_agreement": agreements[cat][0] if agreements[cat] else None,
            "top_disagreement": disagreements[cat][0] if disagreements[cat] else None,
        }

    cases_path = out_dir / "function_case_studies.json"
    with open(cases_path, "w", encoding="utf-8") as f:
        json.dump(cases_summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved case studies JSON to: {cases_path}")


if __name__ == "__main__":
    main()
