#!/usr/bin/env python3
"""
Benchmark LLM Annotators Against the Adjudicated Human Gold Standard (Table 3).

Evaluates three open-weight LLMs (GPT-OSS 120B, Qwen 397B, Mistral 128B) against the Human Gold Standard
across the four dimensions reported in Table 3 of the paper:
1. Turn-Level Metaphor Detection: Turn F1 and Cohen's Kappa.
2. Span Localization: Continuous Character-Level Overlap F1 (Kasner et al.).
3. Lexicalization Classification: Cohen's Kappa on 1-to-1 Hungarian Matched Spans (Coverage >= 0.5).
4. Communicative Functions: Multi-Label Micro and Macro F1 on aligned creative spans.

Usage:
    python scripts/benchmark_llms.py \
      --gold data/gold_standard_annotations.json \
      --main-anno data/main_anno.csv \
      --llm-gpt data/results_main_gpt_oss_120b.json \
      --llm-qwen data/results_main_qwen_397b.json \
      --llm-mistral data/results_main_mistral_128b.json
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd
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
# 1. Span Matching & Continuous Overlap Functions
# ==============================================================================

def char_overlap(span_a: Tuple[int, int], span_b: Tuple[int, int]) -> int:
    s1, e1 = span_a
    s2, e2 = span_b
    return max(0, min(e1, e2) - max(s1, s2))


def span_coverage(gold_span: Tuple[int, int], pred_span: Tuple[int, int]) -> float:
    s_g, e_g = gold_span
    s_p, e_p = pred_span
    intersection = max(0, min(e_p, e_g) - max(s_p, s_g))
    g_len = max(1, e_g - s_g)
    return intersection / g_len


def align_spans_one_to_one(
    spans_gold: List[Dict[str, Any]],
    spans_pred: List[Dict[str, Any]],
    threshold: float = 0.5,
) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
    """Hungarian optimal bipartite matching between Gold and Predicted spans."""
    if not spans_gold or not spans_pred:
        return []

    n_g = len(spans_gold)
    n_p = len(spans_pred)
    score_matrix = np.zeros((n_g, n_p), dtype=float)

    for i, sg in enumerate(spans_gold):
        span_g_tup = (sg["start"], sg["end"])
        for j, sp in enumerate(spans_pred):
            span_p_tup = (sp["start"], sp["end"])
            score_matrix[i, j] = span_coverage(span_g_tup, span_p_tup)

    cost_matrix = -score_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched = []
    for r, c in zip(row_ind, col_ind):
        score = score_matrix[r, c]
        if score >= threshold:
            matched.append((spans_gold[r], spans_pred[c], score))

    return matched


# ==============================================================================
# 2. Evaluation Against Gold Standard (Table 3 Metrics)
# ==============================================================================

def evaluate_model_against_gold(
    gold_records: Dict[str, Dict[str, Any]],
    pred_records: Dict[str, Dict[str, Any]],
    common_utts: List[str],
) -> Dict[str, float]:
    """Computes Table 3 metrics for a single model against the human gold standard."""
    # 1. Turn-Level Presence
    y_gold_turn = [gold_records[u]["has_metaphor"] for u in common_utts]
    y_pred_turn = [pred_records[u]["has_metaphor"] for u in common_utts]

    _, _, f1_turn, _ = precision_recall_fscore_support(y_gold_turn, y_pred_turn, average="binary", zero_division=0)
    try:
        kappa_turn = float(cohen_kappa_score(y_gold_turn, y_pred_turn))
        if np.isnan(kappa_turn):
            kappa_turn = 0.0
    except Exception:
        kappa_turn = 0.0

    # 2. Continuous Character Overlap F1
    total_char_overlap = 0
    total_char_gold = 0
    total_char_pred = 0

    for u in common_utts:
        g_spans = gold_records[u]["spans"]
        p_spans = pred_records[u]["spans"]

        chars_g = set()
        for s in g_spans:
            chars_g.update(range(s["start"], s["end"]))
        chars_p = set()
        for s in p_spans:
            chars_p.update(range(s["start"], s["end"]))

        total_char_overlap += len(chars_g & chars_p)
        total_char_gold += len(chars_g)
        total_char_pred += len(chars_p)

    char_p = (total_char_overlap / total_char_pred) if total_char_pred > 0 else 0.0
    char_r = (total_char_overlap / total_char_gold) if total_char_gold > 0 else 0.0
    char_f1 = (2 * char_p * char_r / (char_p + char_r)) if (char_p + char_r) > 0 else 0.0

    # 3. Hungarian 1-to-1 Alignments (Coverage >= 0.5)
    all_matched = []
    for u in common_utts:
        all_matched.extend(align_spans_one_to_one(gold_records[u]["spans"], pred_records[u]["spans"], threshold=0.5))

    # 4. Lexicalization Agreement (Cohen's Kappa)
    matched_lex_gold = [m[0]["lexicalized"] for m in all_matched]
    matched_lex_pred = [m[1]["lexicalized"] for m in all_matched]

    if matched_lex_gold:
        try:
            kappa_lex = float(cohen_kappa_score(matched_lex_gold, matched_lex_pred, labels=[False, True]))
            if np.isnan(kappa_lex):
                kappa_lex = 0.0
        except Exception:
            kappa_lex = 0.0
    else:
        kappa_lex = 0.0

    # 5. Communicative Functions Multi-Label Micro & Macro F1
    intentions_bin_gold = []
    intentions_bin_pred = []

    for s_g, s_p, _ in all_matched:
        set_g = set(s_g.get("intentions", []))
        set_p = set(s_p.get("intentions", []))
        intentions_bin_gold.append([1 if cat in set_g else 0 for cat in VALID_INTENTIONS])
        intentions_bin_pred.append([1 if cat in set_p else 0 for cat in VALID_INTENTIONS])

    if intentions_bin_gold:
        mat_g = np.array(intentions_bin_gold)
        mat_p = np.array(intentions_bin_pred)
        _, _, fn_micro_f1, _ = precision_recall_fscore_support(mat_g, mat_p, average="micro", zero_division=0)
        _, _, fn_macro_f1, _ = precision_recall_fscore_support(mat_g, mat_p, average="macro", zero_division=0)
    else:
        fn_micro_f1 = fn_macro_f1 = 0.0

    return {
        "turn_f1": float(f1_turn),
        "turn_kappa": float(kappa_turn),
        "span_f1": float(char_f1),
        "lex_kappa": float(kappa_lex),
        "func_micro_f1": float(fn_micro_f1),
        "func_macro_f1": float(fn_macro_f1),
    }


# ==============================================================================
# 3. Data Parsers
# ==============================================================================

def load_gold_records(gold_path: Path) -> Dict[str, Dict[str, Any]]:
    with gold_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    utterances = data.get("utterances", {})
    annotations = data.get("annotations", {})

    gold_records = {}
    for uid, u in utterances.items():
        text = u.get("text", "")
        ann_list = annotations.get(uid, [])
        spans = []
        for a in ann_list:
            lbl = str(a.get("label", "")).upper()
            is_lex = "CREATIVE" not in lbl and lbl not in ["NON_LEXICALIZED", "FALSE"]
            raw_funcs = a.get("attributes", {}).get("functions", [])
            if isinstance(raw_funcs, str):
                raw_funcs = [x.strip() for x in raw_funcs.split(";") if x.strip()]
            norm_funcs = [cat for cat in raw_funcs if cat in VALID_INTENTIONS]
            spans.append({
                "start": int(a["start"]),
                "end": int(a["end"]),
                "text": a.get("text", text[int(a["start"]):int(a["end"])]),
                "lexicalized": is_lex,
                "intentions": norm_funcs,
            })
        gold_records[uid] = {
            "text": text,
            "has_metaphor": len(spans) > 0,
            "spans": spans,
        }
    return gold_records


def load_llm_records(llm_path: Path, row_to_utt: Optional[Dict[int, str]] = None) -> Dict[str, Dict[str, Any]]:
    with llm_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    pred_records = {}
    for r in data:
        uid = None
        if row_to_utt:
            row_idx = r.get("row_index")
            if row_idx is not None:
                uid = row_to_utt.get(row_idx)
        if not uid:
            uid = r.get("turn_id") or r.get("utterance_id") or r.get("id")
        if not uid:
            continue
        text = r.get("target_text", r.get("text", ""))
        spans = []
        for a in r.get("annotations", []):
            span_text = a.get("span", a.get("text", ""))
            start = int(a.get("start_char", a.get("start", -1)))
            end = int(a.get("end_char", a.get("end", -1)))
            if start == -1 and span_text and text:
                start = text.find(span_text)
                if start != -1:
                    end = start + len(span_text)

            raw_funcs = a.get("communicative_intentions", a.get("intentions", []))
            if isinstance(raw_funcs, str):
                raw_funcs = [x.strip() for x in raw_funcs.split(";") if x.strip()]
            norm_funcs = [cat for cat in raw_funcs if cat in VALID_INTENTIONS]

            is_lex = bool(a.get("is_lexicalized", a.get("lexicalized", False)))
            spans.append({
                "start": start,
                "end": end,
                "text": span_text,
                "lexicalized": is_lex,
                "intentions": norm_funcs,
            })
        pred_records[uid] = {
            "text": text,
            "has_metaphor": len(spans) > 0,
            "spans": spans,
        }
    return pred_records


# ==============================================================================
# 4. Main Runner
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark LLMs against Human Gold Standard (Table 3).")
    parser.add_argument("--gold", type=str, default="data/gold_standard_release.json", help="Path to gold standard JSON")
    parser.add_argument("--predictions", type=str, default=None, help="Path to model predictions JSON to benchmark against gold")
    parser.add_argument("--model-name", type=str, default="Evaluated Model", help="Display name for the evaluated model")
    parser.add_argument("--main-anno", type=str, default=None, help="Optional legacy metadata CSV (for row_index mapping)")
    parser.add_argument("--llm-gpt", type=str, default=None, help="Optional path to GPT-OSS 120B predictions")
    parser.add_argument("--llm-qwen", type=str, default=None, help="Optional path to Qwen 397B predictions")
    parser.add_argument("--llm-mistral", type=str, default=None, help="Optional path to Mistral 128B predictions")
    parser.add_argument("--out-csv", type=str, default=None, help="Optional output CSV path for benchmark summary")
    args = parser.parse_args()

    gold_path = Path(args.gold)

    if not gold_path.exists():
        logger.warning(f"Gold standard file not found at {gold_path}.")
        print("\nPass valid --gold path to execute benchmark.")
        return

    # Optional mapping from legacy row index to Turn ID
    row_to_utt = {}
    if args.main_anno and Path(args.main_anno).exists():
        df_meta = pd.read_csv(args.main_anno)
        for idx, row in df_meta.iterrows():
            row_to_utt[idx] = f"utt_main_{row['conversation_id']}_{row['message_index']}"

    gold_records = load_gold_records(gold_path)

    models = []
    if args.predictions:
        models.append((args.model_name, Path(args.predictions)))
    if args.llm_qwen:
        models.append(("Qwen 397B", Path(args.llm_qwen)))
    if args.llm_gpt:
        models.append(("GPT-OSS 120B", Path(args.llm_gpt)))
    if args.llm_mistral:
        models.append(("Mistral 128B", Path(args.llm_mistral)))

    if not models:
        # Default placeholder demo
        models = [("Evaluated Model", Path("data/predictions.json"))]

    results = []
    for model_name, model_path in models:
        if not model_path.exists():
            logger.warning(f"Prediction file for {model_name} not found at {model_path}. Skipping.")
            continue
        pred_records = load_llm_records(model_path, row_to_utt)
        common_utts = sorted(list(set(gold_records.keys()) & set(pred_records.keys())))
        metrics = evaluate_model_against_gold(gold_records, pred_records, common_utts)
        metrics["model"] = model_name
        results.append(metrics)

    if not results:
        print("No model prediction files could be evaluated.")
        return

    df_res = pd.DataFrame(results)

    # Print in Table 3 exact layout
    print("\n" + "=" * 80)
    print("LLM BENCHMARK PERFORMANCE AGAINST HUMAN GOLD STANDARD (TABLE 3)")
    print("=" * 80)
    print(f"{'Model':<15} | {'Turn F1':<10} | {'Turn Kappa':<12} | {'Span F1':<10} | {'Lex Kappa':<10} | {'Func F1 (Micro / Macro)':<22}")
    print("-" * 80)
    for r in results:
        func_str = f"{r['func_micro_f1']:.3f} / {r['func_macro_f1']:.3f}"
        print(f"{r['model']:<15} | {r['turn_f1']:<10.3f} | {r['turn_kappa']:<12.3f} | {r['span_f1']:<10.3f} | {r['lex_kappa']:<10.3f} | {func_str:<22}")
    print("=" * 80 + "\n")

    if args.out_csv:
        df_res.to_csv(args.out_csv, index=False)
        print(f"Saved benchmark summary to: {args.out_csv}")


if __name__ == "__main__":
    main()
