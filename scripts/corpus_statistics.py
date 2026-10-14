#!/usr/bin/env python3
"""
Corpus Statistics & Metaphor Distribution Breakdown (Table 1 & § 5.2).

Calculates:
- Turn counts, word lengths, metaphor counts, and word-normalized density (spans/100 words).
- Conventionality distribution (Lexicalized vs. Non-lexicalized / Creative).
- Breakdown across speaker roles (User vs. LLM) and overall corpus totals.

Usage:
    # On the released Gold dataset CSVs:
    python scripts/corpus_statistics.py \
      --turns data/gold_standard_release_turns.csv \
      --spans data/gold_standard_release_spans.csv \
      --out outputs/corpus_statistics_summary.csv

    # Or on MetAnno JSON export format:
    python scripts/corpus_statistics.py \
      --gold-json data/gold_standard_release.json \
      --out outputs/corpus_statistics_summary.csv
"""

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd

# Allow large text fields
csv.field_size_limit(sys.maxsize)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Gold Standard Corpus Summary Statistics (Table 1).")
    parser.add_argument("--turns", type=str, default="data/gold_standard_release_turns.csv", help="Path to released gold turns CSV")
    parser.add_argument("--spans", type=str, default="data/gold_standard_release_spans.csv", help="Path to released gold spans CSV")
    parser.add_argument("--gold-json", type=str, default=None, help="Optional path to gold standard annotations JSON")
    parser.add_argument("--out", type=str, default="outputs/corpus_statistics_summary.csv", help="Output summary CSV path")
    return parser.parse_args()


def load_from_csvs(turns_path: Path, spans_path: Path) -> pd.DataFrame:
    """Loads and aggregates corpus records from released turns and spans CSV files."""
    df_turns = pd.read_csv(turns_path)
    df_spans = pd.read_csv(spans_path) if spans_path.exists() else pd.DataFrame()

    spans_by_turn: Dict[str, List[dict]] = {}
    if not df_spans.empty:
        for _, row in df_spans.iterrows():
            tid = str(row["Turn_ID"])
            if tid not in spans_by_turn:
                spans_by_turn[tid] = []
            is_lex = bool(row["Lexicalized"]) if "Lexicalized" in row else True
            spans_by_turn[tid].append({"lexicalized": is_lex})

    records = []
    for _, row in df_turns.iterrows():
        tid = str(row["Turn_ID"])
        speaker_raw = str(row["Speaker"]).strip()
        speaker = "LLM" if speaker_raw.lower() in ["llm", "assistant", "model", "bot"] else "User"
        text = str(row["Text"])
        word_count = len(text.split())

        turn_spans = spans_by_turn.get(tid, [])
        conv_spans = sum(1 for s in turn_spans if s["lexicalized"])
        creat_spans = sum(1 for s in turn_spans if not s["lexicalized"])
        total_spans = conv_spans + creat_spans

        has_met = str(row.get("Has_Metaphor", "")).upper() in ["YES", "TRUE", "1"] or total_spans > 0

        records.append({
            "turn_id": tid,
            "speaker": speaker,
            "word_count": word_count,
            "has_metaphor": 1 if has_met else 0,
            "conv_spans": conv_spans,
            "creat_spans": creat_spans,
            "total_spans": total_spans,
        })

    return pd.DataFrame(records)


def load_from_json(gold_json_path: Path) -> pd.DataFrame:
    """Loads and aggregates corpus records from gold standard JSON export."""
    with gold_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    utterances = data.get("utterances", {})
    annotations = data.get("annotations", {})

    records = []
    for uid, utt in utterances.items():
        speaker_raw = str(utt.get("speaker", "User")).strip()
        speaker = "LLM" if speaker_raw.lower() in ["llm", "assistant", "model", "bot"] else "User"
        text = str(utt.get("text", ""))
        word_count = len(text.split())

        # Handle both list annotations and dictionary with metaphors
        ann_entry = annotations.get(uid, [])
        if isinstance(ann_entry, dict):
            spans = ann_entry.get("metaphors", [])
        else:
            spans = ann_entry

        conv_spans = 0
        creat_spans = 0
        for s in spans:
            lbl = str(s.get("label", "")).upper()
            is_lex = s.get("lexicalized", True) if "lexicalized" in s else ("CREATIVE" not in lbl)
            if is_lex:
                conv_spans += 1
            else:
                creat_spans += 1

        total_spans = conv_spans + creat_spans
        has_met = 1 if total_spans > 0 else 0

        records.append({
            "turn_id": uid,
            "speaker": speaker,
            "word_count": word_count,
            "has_metaphor": has_met,
            "conv_spans": conv_spans,
            "creat_spans": creat_spans,
            "total_spans": total_spans,
        })

    return pd.DataFrame(records)


def calc_summary(df_sub: pd.DataFrame) -> pd.Series:
    """Computes Table 1 summary metrics for a corpus partition."""
    turns = len(df_sub)
    words = df_sub["word_count"].sum()
    met_turns = df_sub["has_metaphor"].sum()
    conv = df_sub["conv_spans"].sum()
    creat = df_sub["creat_spans"].sum()
    total = conv + creat

    return pd.Series({
        "Total Turns": turns,
        "Turns w/ Metaphor (%)": f"{100 * met_turns / turns:.1f}%" if turns > 0 else "0.0%",
        "Conventional Spans": conv,
        "Creative Spans": creat,
        "Total Spans": total,
        "Creative Ratio (%)": f"{100 * creat / total:.1f}%" if total > 0 else "—",
        "Words/Turn (Mean)": f"{words / turns:.1f}" if turns > 0 else "0.0",
        "Density (Spans/100 Words)": f"{100 * total / words:.2f}" if words > 0 else "0.00"
    })


def main():
    args = parse_args()

    turns_path = Path(args.turns)
    spans_path = Path(args.spans)
    gold_json_path = Path(args.gold_json) if args.gold_json else None

    if gold_json_path and gold_json_path.exists():
        print(f"Loading corpus from JSON: {gold_json_path}")
        df_corpus = load_from_json(gold_json_path)
    elif turns_path.exists():
        print(f"Loading corpus from CSVs: {turns_path} & {spans_path}")
        df_corpus = load_from_csvs(turns_path, spans_path)
    else:
        print(f"Error: Neither {turns_path} nor {gold_json_path} could be found.")
        print("Please provide valid --turns and --spans or --gold-json.")
        sys.exit(1)

    tbl_speaker = df_corpus.groupby("speaker").apply(calc_summary)
    tbl_total = pd.DataFrame([calc_summary(df_corpus)], index=["ALL GOLD TURNS"])

    summary_table = pd.concat([
        pd.DataFrame(index=["[ BY SPEAKER ]"]),
        tbl_speaker,
        pd.DataFrame(index=["[ OVERALL ]"]),
        tbl_total
    ])

    print("\n" + "=" * 80)
    print("GOLD STANDARD CORPUS SUMMARY STATISTICS (TABLE 1)")
    print("=" * 80)
    print(summary_table.to_string())
    print("=" * 80 + "\n")

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    summary_table.to_csv(out_p)
    print(f"[SAVED] Summary table saved to: {out_p}")


if __name__ == "__main__":
    main()
