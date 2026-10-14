#!/usr/bin/env python3
"""
Gold Standard MetAnno Exporter & Dataset Compiler (§ 5.1).

Takes the completed adjudication sheet (CSV) from the human adjudication meeting
and compiles the final curated gold standard dataset into:
1. `gold_standard_annotations.json` (Valid MetAnno project export format, loadable in MetAnno/benchmarks)
2. `gold_standard_turns.csv` (Turn-level corpus view)
3. `gold_standard_spans.csv` (Span-level corpus view with offsets, conventionality, and functions)

Usage:
    python3 compile_gold_standard.py \
      --sheet data/adjudication_sheet_completed.csv \
      --base-export data/base_metanno_export.json \
      --out-json data/gold_standard_annotations.json \
      --out-csv data/gold_standard_spans.csv
"""

import argparse
import re
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Optional

# The 8 canonical communicative functions defined in the paper (§ 3.1 & Table 5)
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


def compile_gold_standard(
    sheet_path: Path,
    base_export_path: Path,
    out_json_path: Path,
    out_csv_path: Optional[Path] = None,
):
    """
    Compiles adjudicated decisions into a cohesive gold standard dataset.
    """
    df = pd.read_csv(sheet_path).fillna("")

    with base_export_path.open("r", encoding="utf-8") as f:
        base_data = json.load(f)

    utterances = base_data.get("utterances", {})
    gold_annotations = {}

    turns_by_id = defaultdict(list)
    for _, row in df.iterrows():
        turns_by_id[row["Turn_ID"]].append(row)

    total_turns = len(turns_by_id)
    metaphor_turns = 0
    total_spans = 0
    lexicalized_spans = 0
    creative_spans = 0
    span_records = []

    for uid, rows in turns_by_id.items():
        row0 = rows[0]
        text = str(row0["Text"])
        speaker = str(row0["Speaker"])
        conv_id = str(row0["Conv_ID"])

        # Determine Final Metaphor Presence:
        # Check explicit adjudicator resolution, fallback to Step 1 majority decision
        final_m_str = str(row0.get("Adjudicator_Final_Metaphor", "")).strip().upper()
        if not final_m_str:
            final_m_str = str(row0.get("Step1_Metaphor_Decision", "")).strip().upper()

        has_metaphor = final_m_str in ["YES", "TRUE", "1", "Y"]

        gold_spans = []
        if has_metaphor:
            for r in rows:
                if r.get("Cluster_ID") == "—":
                    continue

                span_text = str(r.get("Adjudicator_Final_Span", "")).strip()
                if not span_text:
                    span_text = str(r.get("Step2_Suggested_Span", "")).strip()

                if not span_text or span_text == "—":
                    continue

                # Locate character offsets in the turn text
                start = text.find(span_text)
                if start == -1:
                    clean_span = span_text.strip().strip('"').strip("'")
                    start = text.find(clean_span)
                    if start == -1:
                        start = text.lower().find(clean_span.lower())
                end = start + len(span_text) if start != -1 else -1

                # Conventionality status (Lexicalized vs Creative / Novel)
                lex_str = str(r.get("Adjudicator_Final_Lexicalized", "")).strip().upper()
                if not lex_str:
                    lex_str = str(r.get("Step3_Suggested_Status", "")).strip().upper()
                is_lex = "CREATIVE" not in lex_str

                # Communicative functions (strictly for novel / creative metaphors)
                funcs_str = str(r.get("Adjudicator_Final_Functions", "")).strip()
                if not funcs_str or funcs_str in ["—", "nan"]:
                    funcs_str = str(r.get("Step4_Suggested_Functions", "")).strip()

                functions = []
                if not is_lex and funcs_str and funcs_str not in ["—", "nan", "NEEDS_ADJUDICATION"]:
                    # Split comma- or semicolon-separated functions and strip vote tallies
                    raw_fn_items = [f.strip() for f in re.split(r"[;,]", funcs_str) if f.strip()]
                    for fn in raw_fn_items:
                        clean_fn = fn.split("(")[0].strip()
                        if clean_fn in VALID_INTENTIONS:
                            functions.append(clean_fn)

                if is_lex:
                    lexicalized_spans += 1
                else:
                    creative_spans += 1
                total_spans += 1

                span_dict = {
                    "start": start,
                    "end": end,
                    "text": span_text,
                    "lexicalized": is_lex,
                    "intentions": functions,
                    "comment": str(r.get("Adjudicator_Notes", "")).strip(),
                }
                gold_spans.append(span_dict)

                span_records.append({
                    "Turn_ID": uid,
                    "Conv_ID": conv_id,
                    "Speaker": speaker,
                    "Turn_Text": text,
                    "Span_Text": span_text,
                    "Start": start,
                    "End": end,
                    "Lexicalized": is_lex,
                    "Communicative_Functions": "; ".join(functions) if functions else "—",
                    "Adjudicator_Notes": r.get("Adjudicator_Notes", ""),
                })

        if has_metaphor and len(gold_spans) > 0:
            metaphor_turns += 1

        ann_id = f"ann_gold_{uid}"
        gold_annotations[ann_id] = {
            "id": ann_id,
            "utterance_id": uid,
            "metaphor_present": has_metaphor,
            "identification_completed": True,
            "classification_completed": True,
            "metaphors": gold_spans,
        }

    gold_export = {
        "projects": base_data.get("projects", {}),
        "schemas": base_data.get("schemas", {}),
        "utterances": utterances,
        "annotations": gold_annotations,
    }

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(gold_export, f, indent=2, ensure_ascii=False)
    print(f"\n[GOLD STANDARD JSON SAVED] {out_json_path}")

    if out_csv_path and span_records:
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(span_records).to_csv(out_csv_path, index=False)
        print(f"[GOLD STANDARD SPANS CSV SAVED] {out_csv_path}")

    print("\n" + "=" * 60)
    print("GOLD STANDARD DATASET SUMMARY:")
    print("=" * 60)
    print(f"  • Total Turns:            {total_turns}")
    print(f"  • Metaphor Turns:         {metaphor_turns} ({metaphor_turns / max(1, total_turns) * 100:.1f}%)")
    print(f"  • Total Metaphor Spans:   {total_spans}")
    print(f"  • Conventional / Lexical: {lexicalized_spans} ({lexicalized_spans / max(1, total_spans) * 100:.1f}%)")
    print(f"  • Creative / Novel:       {creative_spans} ({creative_spans / max(1, total_spans) * 100:.1f}%)")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compile Final Gold Standard from Adjudication Sheet (§ 5.1).")
    parser.add_argument(
        "--sheet",
        type=str,
        required=True,
        help="Path to completed/reviewed adjudication CSV.",
    )
    parser.add_argument(
        "--base-export",
        type=str,
        required=True,
        help="Path to base MetAnno JSON export (providing schema and utterance metadata).",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="data/gold_standard_annotations.json",
        help="Path to output gold standard MetAnno JSON.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="data/gold_standard_spans.csv",
        help="Path to output gold standard spans CSV.",
    )
    args = parser.parse_args()

    compile_gold_standard(
        sheet_path=Path(args.sheet),
        base_export_path=Path(args.base_export),
        out_json_path=Path(args.out_json),
        out_csv_path=Path(args.out_csv) if args.out_csv else None,
    )


if __name__ == "__main__":
    main()
