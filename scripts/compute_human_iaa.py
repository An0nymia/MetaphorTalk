#!/usr/bin/env python3
"""
Human Inter-Annotator Agreement (IAA) & Reliability Analysis (Table 2).

Evaluates the three human annotators across the reported paper metrics:
1. Turn-Level Metaphor Detection: Pairwise Cohen's Kappa & Multi-Rater Fleiss' Kappa.
2. Span Localization: Continuous Character-Level F1 Overlap (Kasner et al.).
3. Lexicalization Agreement: Pairwise Cohen's Kappa & Multi-Rater Fleiss' Kappa on
   1-to-1 Hungarian Bipartite Matched Spans (Coverage >= 0.5).
4. Communicative Functions: Multi-Label Micro and Macro F1 on aligned creative spans.

Usage:
    python scripts/compute_human_iaa.py \
      --files data/annotator1.json data/annotator2.json data/annotator3.json \
      --labels A1 A2 A3 \
      --out-report outputs/human_iaa_report.txt \
      --out-json outputs/human_iaa_metrics.json
"""

import argparse
import itertools
import json
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    cohen_kappa_score,
    precision_recall_fscore_support,
)

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


# ==============================================================================
# 1. Loading & Health Checking MetAnno Exports
# ==============================================================================

def load_metanno_export(file_path: Path) -> Dict[str, Any]:
    """Loads raw MetAnno JSON export."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_and_validate_annotator(file_path: Path, label: str) -> Dict[str, Any]:
    """Loads, validates, and standardizes a single annotator export."""
    data = load_metanno_export(file_path)
    utterances = data.get("utterances", {})
    annotations = data.get("annotations", {})

    parsed_records = {}
    total_spans = 0

    for utt_id, utt_dict in utterances.items():
        text = utt_dict.get("text", "")
        speaker = utt_dict.get("speaker", "unknown")
        conv_id = utt_dict.get("conversation_id", "unknown")

        ann_list = annotations.get(utt_id, [])
        has_metaphor = len(ann_list) > 0
        total_spans += len(ann_list)

        clean_spans = []
        for m in ann_list:
            start = int(m.get("start", -1))
            end = int(m.get("end", -1))
            span_text = str(m.get("text", "")).strip()

            label_val = str(m.get("label", "")).strip().upper()
            is_lex = "CREATIVE" not in label_val and label_val not in ["NON_LEXICALIZED", "NOVEL", "FALSE"]

            raw_intentions = m.get("attributes", {}).get("functions", [])
            if isinstance(raw_intentions, str):
                raw_intentions = [i.strip() for i in raw_intentions.split(";") if i.strip()]

            norm_intents = [str(cat).strip() for cat in raw_intentions if str(cat).strip() in VALID_INTENTIONS]

            clean_spans.append({
                "start": start,
                "end": end,
                "text": span_text,
                "lexicalized": is_lex,
                "intentions": norm_intents,
            })

        parsed_records[utt_id] = {
            "text": text,
            "speaker": speaker,
            "conversation_id": conv_id,
            "metaphor_present": has_metaphor,
            "spans": clean_spans,
        }

    return {
        "label": label,
        "file_name": file_path.name,
        "records": parsed_records,
        "total_utterances": len(utterances),
        "total_spans": total_spans,
    }


# ==============================================================================
# 2. Continuous Character-Level Overlap & Alignment Functions
# ==============================================================================

def char_overlap(span_a: Tuple[int, int], span_b: Tuple[int, int]) -> int:
    s1, e1 = span_a
    s2, e2 = span_b
    return max(0, min(e1, e2) - max(s1, s2))


def span_coverage(gold_span: Tuple[int, int], pred_span: Tuple[int, int]) -> float:
    s_g, e_g = gold_span
    s_p, e_p = pred_span
    len_g = e_g - s_g
    len_p = e_p - s_p
    min_len = min(len_g, len_p)
    if min_len <= 0:
        return 0.0
    return char_overlap(gold_span, pred_span) / min_len


def compute_character_overlap_f1(spans_a: List[Dict[str, Any]], spans_b: List[Dict[str, Any]]) -> Tuple[float, float, float]:
    """Computes continuous character-level precision, recall, and F1 across a turn."""
    if not spans_a and not spans_b:
        return 1.0, 1.0, 1.0
    if not spans_a or not spans_b:
        return 0.0, 0.0, 0.0

    chars_a = set()
    for s in spans_a:
        chars_a.update(range(s["start"], s["end"]))

    chars_b = set()
    for s in spans_b:
        chars_b.update(range(s["start"], s["end"]))

    intersection = len(chars_a & chars_b)
    p = intersection / len(chars_a) if chars_a else 0.0
    r = intersection / len(chars_b) if chars_b else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def align_spans_one_to_one(
    spans_a: List[Dict[str, Any]],
    spans_b: List[Dict[str, Any]],
    threshold: float = 0.5,
) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
    """Hungarian bipartite 1-to-1 matching requiring coverage >= threshold."""
    if not spans_a or not spans_b:
        return []

    n_a, n_b = len(spans_a), len(spans_b)
    cost_matrix = np.ones((n_a, n_b), dtype=float)

    for i, s1 in enumerate(spans_a):
        for j, s2 in enumerate(spans_b):
            cov = span_coverage((s1["start"], s1["end"]), (s2["start"], s2["end"]))
            if cov >= threshold:
                cost_matrix[i, j] = 1.0 - cov

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    for r, c in zip(row_ind, col_ind):
        score = 1.0 - cost_matrix[r, c]
        if score >= threshold:
            matches.append((spans_a[r], spans_b[c], score))
    return matches


# ==============================================================================
# 3. Pairwise IAA Calculation
# ==============================================================================

def compute_pairwise_agreement(
    ann_a: Dict[str, Any],
    ann_b: Dict[str, Any],
) -> Dict[str, Any]:
    recs_a, recs_b = ann_a["records"], ann_b["records"]
    common_utts = sorted(list(set(recs_a.keys()) & set(recs_b.keys())))

    # 1. Turn-Level Metaphor Detection
    y_true_pres = [recs_a[u]["metaphor_present"] for u in common_utts]
    y_pred_pres = [recs_b[u]["metaphor_present"] for u in common_utts]
    kappa_pres = float(cohen_kappa_score(y_true_pres, y_pred_pres))

    # 2. Continuous Character Overlap F1
    total_char_overlap = 0
    total_char_a = 0
    total_char_b = 0

    for u in common_utts:
        s_a = recs_a[u]["spans"]
        s_b = recs_b[u]["spans"]
        chars_a = set()
        for s in s_a:
            chars_a.update(range(s["start"], s["end"]))
        chars_b = set()
        for s in s_b:
            chars_b.update(range(s["start"], s["end"]))

        total_char_overlap += len(chars_a & chars_b)
        total_char_a += len(chars_a)
        total_char_b += len(chars_b)

    micro_prec = (total_char_overlap / total_char_a) if total_char_a > 0 else 0.0
    micro_rec = (total_char_overlap / total_char_b) if total_char_b > 0 else 0.0
    char_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) > 0 else 0.0

    # 3. Hungarian 1-to-1 Bipartite Alignments (Coverage >= 0.5)
    all_matched_05 = []
    for u in common_utts:
        all_matched_05.extend(align_spans_one_to_one(recs_a[u]["spans"], recs_b[u]["spans"], threshold=0.5))

    # 4. Lexicalization Agreement on Aligned Spans
    matched_lex_a = [m[0]["lexicalized"] for m in all_matched_05]
    matched_lex_b = [m[1]["lexicalized"] for m in all_matched_05]

    if matched_lex_a:
        try:
            kappa_lex = float(cohen_kappa_score(matched_lex_a, matched_lex_b, labels=[False, True]))
            if np.isnan(kappa_lex):
                kappa_lex = 0.0
        except Exception:
            kappa_lex = 0.0
    else:
        kappa_lex = 0.0

    # 5. Communicative Functions Multi-Label Micro & Macro F1
    intentions_bin_a = []
    intentions_bin_b = []
    for s_a, s_b, _ in all_matched_05:
        set_a = set(s_a.get("intentions", []))
        set_b = set(s_b.get("intentions", []))
        intentions_bin_a.append([1 if cat in set_a else 0 for cat in VALID_INTENTIONS])
        intentions_bin_b.append([1 if cat in set_b else 0 for cat in VALID_INTENTIONS])

    if intentions_bin_a:
        mat_a = np.array(intentions_bin_a)
        mat_b = np.array(intentions_bin_b)
        _, _, f1_func_micro, _ = precision_recall_fscore_support(mat_a, mat_b, average="micro", zero_division=0)
        _, _, f1_func_macro, _ = precision_recall_fscore_support(mat_a, mat_b, average="macro", zero_division=0)
    else:
        f1_func_micro = f1_func_macro = 0.0

    return {
        "pair_label": f"{ann_a['label']} vs {ann_b['label']}",
        "turn_kappa": kappa_pres,
        "span_char_f1": char_f1,
        "lex_kappa": kappa_lex,
        "func_f1_micro": float(f1_func_micro),
        "func_f1_macro": float(f1_func_macro),
        "aligned_spans_count": len(all_matched_05),
    }


def compute_turn_fleiss_kappa(all_annotators: List[Dict[str, Any]]) -> float:
    """Multi-Rater Fleiss' Kappa for utterance metaphor presence."""
    common_keys = set(all_annotators[0]["records"].keys())
    for a in all_annotators[1:]:
        common_keys &= set(a["records"].keys())

    if not common_keys:
        return 0.0

    n_items = len(common_keys)
    n_raters = len(all_annotators)
    table = np.zeros((n_items, 2), dtype=int)

    for i, k in enumerate(sorted(list(common_keys))):
        for a in all_annotators:
            val = 1 if a["records"][k]["metaphor_present"] else 0
            table[i, val] += 1

    p_j = np.sum(table, axis=0) / (n_items * n_raters)
    p_e_bar = np.sum(p_j ** 2)
    p_i = (np.sum(table ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = np.mean(p_i)
    if p_e_bar == 1.0:
        return 1.0
    return float((p_bar - p_e_bar) / (1.0 - p_e_bar))


def compute_lexicalization_fleiss_kappa(all_annotators: List[Dict[str, Any]], coverage_thresh: float = 0.5) -> float:
    """Multi-Rater Fleiss' Kappa on mutually aligned spans (Hungarian Coverage >= 0.5)."""
    common_keys = set(all_annotators[0]["records"].keys())
    for a in all_annotators[1:]:
        common_keys &= set(a["records"].keys())

    if len(all_annotators) < 3 or not common_keys:
        return 0.0

    triplets = []
    for uid in sorted(list(common_keys)):
        spans_0 = all_annotators[0]["records"][uid]["spans"]
        spans_1 = all_annotators[1]["records"][uid]["spans"]
        spans_2 = all_annotators[2]["records"][uid]["spans"]

        for m0 in spans_0:
            match1 = next((m1 for m1 in spans_1 if span_coverage((m0["start"], m0["end"]), (m1["start"], m1["end"])) >= coverage_thresh), None)
            if not match1:
                continue
            match2 = next((m2 for m2 in spans_2 if span_coverage((m0["start"], m0["end"]), (m2["start"], m2["end"])) >= coverage_thresh), None)
            if match2:
                triplets.append((m0["lexicalized"], match1["lexicalized"], match2["lexicalized"]))

    if not triplets:
        return 0.0

    table = np.zeros((len(triplets), 2), dtype=int)
    for idx, trip in enumerate(triplets):
        for val in trip:
            table[idx, 1 if val else 0] += 1

    n_items = len(triplets)
    n_raters = len(all_annotators)
    p_j = np.sum(table, axis=0) / (n_items * n_raters)
    p_e_bar = np.sum(p_j ** 2)
    p_i = (np.sum(table ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = np.mean(p_i)
    if p_e_bar == 1.0:
        return 1.0
    return float((p_bar - p_e_bar) / (1.0 - p_e_bar))


# ==============================================================================
# 4. Main Runner
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compute Human Inter-Annotator Agreement (Table 2).")
    parser.add_argument("--files", nargs="+", required=True, help="List of MetAnno JSON export files (e.g. A1, A2, A3)")
    parser.add_argument("--labels", nargs="+", default=None, help="Labels for annotators (e.g. A1 A2 A3)")
    parser.add_argument("--out-report", type=str, default=None, help="Output path for summary report")
    parser.add_argument("--out-json", type=str, default=None, help="Output path for JSON metrics")
    args = parser.parse_args()

    file_paths = [Path(p) for p in args.files]
    labels = args.labels or [f"A{i+1}" for i in range(len(file_paths))]

    annotators = [load_and_validate_annotator(fp, lbl) for fp, lbl in zip(file_paths, labels)]

    # Pairwise IAA
    pairwise_results = []
    for ann_a, ann_b in itertools.combinations(annotators, 2):
        res = compute_pairwise_agreement(ann_a, ann_b)
        pairwise_results.append(res)

    # Multi-Rater Fleiss' Kappa
    turn_fleiss = compute_turn_fleiss_kappa(annotators)
    lex_fleiss = compute_lexicalization_fleiss_kappa(annotators, coverage_thresh=0.5)

    # Print Table 2 Format
    print("\n" + "=" * 80)
    print("HUMAN INTER-ANNOTATOR AGREEMENT (TABLE 2)")
    print("=" * 80)
    print(f"{'Pair':<15} | {'Turn Kappa':<12} | {'Span F1':<10} | {'Lex Kappa':<10} | {'Func F1 (Micro / Macro)':<22}")
    print("-" * 80)

    mean_turn_k = np.mean([p["turn_kappa"] for p in pairwise_results])
    mean_span_f1 = np.mean([p["span_char_f1"] for p in pairwise_results])
    mean_lex_k = np.mean([p["lex_kappa"] for p in pairwise_results])
    mean_func_micro = np.mean([p["func_f1_micro"] for p in pairwise_results])
    mean_func_macro = np.mean([p["func_f1_macro"] for p in pairwise_results])

    for p in pairwise_results:
        func_str = f"{p['func_f1_micro']:.3f} / {p['func_f1_macro']:.3f}"
        print(f"{p['pair_label']:<15} | {p['turn_kappa']:<12.3f} | {p['span_char_f1']:<10.3f} | {p['lex_kappa']:<10.3f} | {func_str:<22}")

    print("-" * 80)
    mean_func_str = f"{mean_func_micro:.3f} / {mean_func_macro:.3f}"
    print(f"{'Pairwise Mean':<15} | {mean_turn_k:<12.3f} | {mean_span_f1:<10.3f} | {mean_lex_k:<10.3f} | {mean_func_str:<22}")
    print("=" * 80)
    print(f"Multi-Rater Fleiss' Kappa (Turn Presence): \t{turn_fleiss:.3f}")
    print(f"Multi-Rater Fleiss' Kappa (Lexicalization): \t{lex_fleiss:.3f}")
    print("=" * 80 + "\n")

    summary_data = {
        "pairwise": pairwise_results,
        "pairwise_mean": {
            "turn_kappa": float(mean_turn_k),
            "span_char_f1": float(mean_span_f1),
            "lex_kappa": float(mean_lex_k),
            "func_f1_micro": float(mean_func_micro),
            "func_f1_macro": float(mean_func_macro),
        },
        "multi_rater": {
            "turn_fleiss_kappa": float(turn_fleiss),
            "lexicalization_fleiss_kappa": float(lex_fleiss),
        }
    }

    if args.out_json:
        out_p = Path(args.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        print(f"Saved metrics to: {out_p}")


if __name__ == "__main__":
    main()
