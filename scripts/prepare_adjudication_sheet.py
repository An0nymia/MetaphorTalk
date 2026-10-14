#!/usr/bin/env python3
"""
Metaphor Annotation Adjudication Preparation Pipeline (§ 5.1).

This script implements the adjudication preparation pipeline:
1. Turn-Level Metaphor Identification:
   - Computes majority votes across the independent human annotators.
2. Span Boundary Clustering:
   - Identifies overlapping or nested span annotations using character overlap
     and substring matching, clustering them into candidate metaphor sites.
3. Conventionality (Lexicalized vs. Non-Lexicalized):
   - Tallies conventionality votes and determines majority decisions or ties.
4. Communicative Functions:
   - For non-lexicalized (novel/creative) metaphors, aggregates the communicative
     functions selected by annotators.
5. Priority Ranking & Output Generation:
   - Categorizes cases into review priorities (e.g. lexicalization disputes,
     span boundary differences, communicative function reviews).
   - Generates an Excel/Sheets-compatible conflict resolution table (`adjudication_sheet.csv`)
     and a formatted meeting agenda dossier (`adjudication_dossier.txt`) for human experts.

Usage:
    python3 prepare_adjudication_sheet.py \
      --files data/annotator1.json data/annotator2.json data/annotator3.json \
      --labels Annotator1 Annotator2 Annotator3 \
      --out-csv outputs/adjudication_sheet.csv \
      --out-dossier outputs/adjudication_dossier.txt
"""

import argparse
import json
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# The 8 canonical communicative functions defined in the paper
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


def load_metanno_export(path: Path) -> Dict[str, Any]:
    """Loads a MetAnno JSON project export file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_annotator_export(raw_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Parses a MetAnno JSON export into standardized per-turn records.
    Filters communicative functions strictly for non-lexicalized spans against VALID_INTENTIONS.
    """
    utterances = raw_data.get("utterances", {})
    annotations = raw_data.get("annotations", {})

    parsed = {}
    for ann_id, ann in annotations.items():
        utt_id = ann.get("utterance_id")
        utt = utterances.get(utt_id, {})
        text = str(utt.get("text", ""))
        speaker = str(utt.get("speaker", "Unknown"))
        conv_id = str(utt.get("conversation_id", ""))
        index = utt.get("index", 0)

        is_id_comp = ann.get("identification_completed", False)
        is_class_comp = ann.get("classification_completed", False)
        has_metaphor = bool(ann.get("metaphor_present", False))
        raw_metaphors = ann.get("metaphors", [])

        clean_spans = []
        for m in raw_metaphors:
            start = int(m.get("start", -1))
            end = int(m.get("end", -1))
            span_text = str(m.get("text", ""))
            is_lex = bool(m.get("lexicalized", False))
            raw_intents = m.get("intentions", [])

            # Recover offsets if slicing mismatch occurred in export
            if text and (start < 0 or end > len(text) or text[start:end].strip() != span_text.strip()):
                idx = text.find(span_text)
                if idx != -1:
                    start = idx
                    end = idx + len(span_text)

            # Keep only canonical communicative functions for non-lexicalized metaphors
            clean_intents = []
            if not is_lex and raw_intents:
                for cat in raw_intents:
                    cat_clean = str(cat).strip()
                    if cat_clean in VALID_INTENTIONS:
                        clean_intents.append(cat_clean)

            clean_spans.append({
                "start": start,
                "end": end,
                "text": span_text,
                "lexicalized": is_lex,
                "intentions": clean_intents,
                "comment": m.get("comment", ""),
            })

        parsed[utt_id] = {
            "utt_id": utt_id,
            "conversation_id": conv_id,
            "index": index,
            "speaker": speaker,
            "text": text,
            "is_completed": is_id_comp or is_class_comp or (has_metaphor and len(clean_spans) > 0),
            "metaphor_present": has_metaphor,
            "spans": clean_spans,
        }
    return parsed


def compute_span_char_overlap(s1: Dict[str, Any], s2: Dict[str, Any]) -> float:
    """Computes character-level Intersection-over-Union (IoU) between two spans."""
    start = max(s1["start"], s2["start"])
    end = min(s1["end"], s2["end"])
    intersection = max(0, end - start)
    union = max(s1["end"], s2["end"]) - min(s1["start"], s2["start"])
    if union == 0:
        return 0.0
    return intersection / union


def cluster_spans_for_turn(
    annotator_records: List[Tuple[str, Dict[str, Any]]],
    target_text: str,
) -> List[Dict[str, Any]]:
    """
    Clusters overlapping spans across annotators into distinct metaphor candidate sites
    based on character IoU overlap (> 0.2) or substring containment.
    """
    all_spans = []
    for lbl, rec in annotator_records:
        for s in rec.get("spans", []):
            all_spans.append({"annotator": lbl, **s})

    if not all_spans:
        return []

    clusters = []
    visited: Set[int] = set()

    for i, s1 in enumerate(all_spans):
        if i in visited:
            continue
        cluster = [s1]
        visited.add(i)

        for j, s2 in enumerate(all_spans):
            if j in visited:
                continue
            overlap = compute_span_char_overlap(s1, s2)
            is_sub = (s1["text"].strip() in s2["text"].strip()) or (s2["text"].strip() in s1["text"].strip())
            if overlap > 0.2 or is_sub:
                cluster.append(s2)
                visited.add(j)

        clusters.append(cluster)

    processed_clusters = []
    for c_idx, cl in enumerate(clusters, 1):
        span_texts = [s["text"].strip() for s in cl]
        text_counts = Counter(span_texts)
        most_common_text = text_counts.most_common(1)[0][0]
        contrib_annots = sorted(list(set(s["annotator"] for s in cl)))

        # Conventionality votes
        lex_votes = [s["lexicalized"] for s in cl]
        n_lex = sum(1 for v in lex_votes if v)
        n_creative = sum(1 for v in lex_votes if not v)

        n_raters = len(contrib_annots)
        if n_raters >= 2:
            if n_creative > n_lex:
                sug_lex = "CREATIVE (Majority)"
            elif n_lex > n_creative:
                sug_lex = "LEXICALIZED (Majority)"
            else:
                sug_lex = "DISPUTED (Tie)"
        else:
            sug_lex = "CREATIVE (Single Rater)" if n_creative > 0 else "LEXICALIZED (Single Rater)"

        # Functions votes (from creative/novel annotators only)
        creative_intents = []
        for s in cl:
            if not s["lexicalized"]:
                creative_intents.extend(s.get("intentions", []))
        intent_counts = Counter(creative_intents)

        sug_functions = []
        if "CREATIVE" in sug_lex and intent_counts:
            sug_functions = [f"{fn} ({cnt})" for fn, cnt in intent_counts.most_common()]

        annot_details = []
        for s in cl:
            tag = "LEX" if s["lexicalized"] else "CREATIVE"
            ints = ",".join(s["intentions"]) if s["intentions"] else "—"
            annot_details.append(f"{s['annotator']}: \"{s['text']}\" [{tag}, fn={ints}]")

        processed_clusters.append({
            "cluster_index": c_idx,
            "candidate_texts": list(text_counts.keys()),
            "most_common_span": most_common_text,
            "contributing_annotators": contrib_annots,
            "annot_details": " | ".join(annot_details),
            "lexicalized_votes": f"{n_lex} Lex vs {n_creative} Creative",
            "suggested_lexicalized": sug_lex,
            "suggested_functions": ", ".join(sug_functions) if sug_functions else ("—" if sug_lex == "LEXICALIZED" else "NEEDS_ADJUDICATION"),
            "raw_spans": cl,
        })

    return processed_clusters


def build_hierarchical_adjudication_table(
    parsed_annotators: List[Tuple[str, Dict[str, Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    """
    Builds the structured conflict resolution table for all dialogue turns,
    categorizing items by adjudication priority.
    """
    all_recs = [recs for _, recs in parsed_annotators]

    all_utt_ids = set()
    for recs in all_recs:
        all_utt_ids.update(recs.keys())

    sample_recs = {}
    for recs in all_recs:
        for uid, r in recs.items():
            if uid not in sample_recs:
                sample_recs[uid] = (r.get("conversation_id", ""), r.get("index", 0))

    sorted_utt_ids = sorted(
        list(all_utt_ids),
        key=lambda u: (sample_recs[u][0], sample_recs[u][1], u),
    )

    adjudication_rows = []

    for uid in sorted_utt_ids:
        target_text = ""
        speaker = "Unknown"
        conv_id = ""
        turn_idx = 0
        for recs in all_recs:
            if uid in recs:
                target_text = recs[uid]["text"]
                speaker = recs[uid]["speaker"]
                conv_id = recs[uid]["conversation_id"]
                turn_idx = recs[uid]["index"]
                break

        completed_annots = []
        yes_votes = []
        no_votes = []
        active_annots_data = []

        for lbl, recs in parsed_annotators:
            if uid in recs and recs[uid]["is_completed"]:
                completed_annots.append(lbl)
                active_annots_data.append((lbl, recs[uid]))
                if recs[uid]["metaphor_present"]:
                    yes_votes.append(lbl)
                else:
                    no_votes.append(lbl)

        n_completed = len(completed_annots)
        n_yes = len(yes_votes)
        n_no = len(no_votes)

        # Process turns annotated by the full annotator panel
        if n_completed < len(parsed_annotators):
            continue

        # Step 1: Turn-Level Metaphor Identification (Majority Vote)
        if n_yes >= 2:
            step1_decision = "YES"
            step1_status = f"MAJORITY ({n_yes}/3)"
        else:
            step1_decision = "NO"
            step1_status = f"MAJORITY ({n_no}/3 No)"

        votes_summary = f"Yes=[{','.join(yes_votes)}] | No=[{','.join(no_votes)}]"

        # Step 2: Span Boundary Clustering
        clusters = cluster_spans_for_turn(active_annots_data, target_text)

        if not clusters:
            row = {
                "Turn_ID": uid,
                "Conv_ID": conv_id,
                "Turn_Index": turn_idx,
                "Speaker": speaker,
                "Text": target_text,
                "Annotators_Completed": f"{n_completed}/3 ({','.join(completed_annots)})",
                "Step1_Metaphor_Decision": step1_decision,
                "Step1_Vote_Summary": votes_summary,
                "Step1_Vote_Status": step1_status,
                "Cluster_ID": "—",
                "Step2_Span_Candidates": "—",
                "Step2_Suggested_Span": "—",
                "Step3_Lexicalized_Votes": "—",
                "Step3_Suggested_Status": "—",
                "Step4_Suggested_Functions": "—",
                "Adjudication_Priority": "5_CLEAN_CONSENSUS_NO",
                "Adjudicator_Final_Metaphor": "NO" if step1_decision == "NO" else "",
                "Adjudicator_Final_Span": "",
                "Adjudicator_Final_Lexicalized": "",
                "Adjudicator_Final_Functions": "",
                "Adjudicator_Notes": "",
            }
            adjudication_rows.append(row)
        else:
            for cl in clusters:
                if step1_decision == "NO":
                    cl_priority = "5_CLEAN_CONSENSUS_NO"
                elif "DISPUTED" in cl["suggested_lexicalized"]:
                    cl_priority = "1_LEXICALIZATION_DISPUTED"
                elif "CREATIVE (Majority)" in cl["suggested_lexicalized"]:
                    cl_priority = "2_CREATIVE_FUNCTION_REVIEW"
                elif len(cl["candidate_texts"]) > 1:
                    cl_priority = "3_SPAN_BOUNDARY_REVIEW"
                else:
                    cl_priority = "4_CLEAN_CONSENSUS_LEX"

                row = {
                    "Turn_ID": uid,
                    "Conv_ID": conv_id,
                    "Turn_Index": turn_idx,
                    "Speaker": speaker,
                    "Text": target_text,
                    "Annotators_Completed": f"{n_completed}/3 ({','.join(completed_annots)})",
                    "Step1_Metaphor_Decision": step1_decision,
                    "Step1_Vote_Summary": votes_summary,
                    "Step1_Vote_Status": step1_status,
                    "Cluster_ID": f"Span_{cl['cluster_index']}",
                    "Step2_Span_Candidates": cl["annot_details"],
                    "Step2_Suggested_Span": cl["most_common_span"],
                    "Step3_Lexicalized_Votes": cl["lexicalized_votes"],
                    "Step3_Suggested_Status": cl["suggested_lexicalized"],
                    "Step4_Suggested_Functions": cl["suggested_functions"],
                    "Adjudication_Priority": cl_priority,
                    "Adjudicator_Final_Metaphor": "YES" if step1_decision == "YES" and "MAJORITY" in step1_status else "",
                    "Adjudicator_Final_Span": cl["most_common_span"] if len(cl["candidate_texts"]) == 1 else "",
                    "Adjudicator_Final_Lexicalized": cl["suggested_lexicalized"] if "MAJORITY" in cl["suggested_lexicalized"] else "",
                    "Adjudicator_Final_Functions": cl["suggested_functions"] if cl["suggested_lexicalized"] == "LEXICALIZED" else "",
                    "Adjudicator_Notes": "",
                }
                adjudication_rows.append(row)

    return adjudication_rows


def generate_meeting_dossier_text(
    rows: List[Dict[str, Any]],
    out_path: Path,
):
    """
    Generates a priority-ranked human-readable adjudication agenda dossier.
    """
    priority_order = [
        "1_LEXICALIZATION_DISPUTED",
        "2_CREATIVE_FUNCTION_REVIEW",
        "3_SPAN_BOUNDARY_REVIEW",
        "4_CLEAN_CONSENSUS_LEX",
        "5_CLEAN_CONSENSUS_NO",
    ]

    by_priority = defaultdict(list)
    for r in rows:
        by_priority[r["Adjudication_Priority"]].append(r)

    lines = []
    lines.append("=" * 100)
    lines.append("METAPHOR ANNOTATION ADJUDICATION DOSSIER (HUMAN MEETING WORKING GUIDE)")
    lines.append("=" * 100)
    lines.append("Workflow Order:")
    lines.append("  1. Metaphor? (Yes/No)  ->  2. Final Span  ->  3. Lexicalized? (Yes/No)  ->  4. Communicative Functions")
    lines.append("=" * 100 + "\n")

    lines.append("SUMMARY OF ITEMS BY ADJUDICATION PRIORITY:")
    for p in priority_order:
        items = by_priority.get(p, [])
        lines.append(f"  • {p:<45}: {len(items):>4} items")
    lines.append("")

    for p in priority_order:
        items = by_priority.get(p, [])
        if not items:
            continue

        lines.append("=" * 100)
        lines.append(f"SECTION: {p} (Total: {len(items)} cases)")
        lines.append("=" * 100 + "\n")

        for idx, r in enumerate(items, 1):
            lines.append(f"[{idx}] Turn: {r['Turn_ID']} | Conv: {r['Conv_ID']} | Speaker: {r['Speaker']} | Completed: {r['Annotators_Completed']}")
            lines.append(f"    Text: \"{r['Text']}\"")
            lines.append(f"    [Step 1 - Metaphor Y/N]:  {r['Step1_Metaphor_Decision']} ({r['Step1_Vote_Status']}) -> {r['Step1_Vote_Summary']}")
            if r['Cluster_ID'] != "—":
                lines.append(f"    [Step 2 - Candidates]:    {r['Step2_Span_Candidates']}")
                lines.append(f"    [Step 2 - Suggested Span]: \"{r['Step2_Suggested_Span']}\"")
                lines.append(f"    [Step 3 - Conventionality]: Suggested {r['Step3_Suggested_Status']} ({r['Step3_Lexicalized_Votes']})")
                lines.append(f"    [Step 4 - Functions]:      {r['Step4_Suggested_Functions']}")
            lines.append("    " + "-" * 80)
            lines.append("    >>> FINAL DECISION: Metaphor: [     ] | Span: [                                      ] | Lex: [     ] | Funcs: [                    ]")
            lines.append("    " + "-" * 80 + "\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[DOSSIER SAVED] Meeting Agenda Dossier: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Hierarchical Metaphor Adjudication Sheet & Dossier (§ 5.1).")
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="List of MetAnno JSON export files from independent annotators.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["Annotator1", "Annotator2", "Annotator3"],
        help="Labels for annotators corresponding to the files.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="outputs/adjudication_sheet.csv",
        help="Path to save output conflict resolution CSV.",
    )
    parser.add_argument(
        "--out-dossier",
        type=str,
        default="outputs/adjudication_dossier.txt",
        help="Path to save meeting agenda dossier.",
    )
    args = parser.parse_args()

    file_paths = [Path(f) for f in args.files]
    labels = args.labels[:len(file_paths)]

    parsed_annotators = []
    for p, lbl in zip(file_paths, labels):
        print(f"Loading {lbl} from {p}...")
        raw = load_metanno_export(p)
        recs = parse_annotator_export(raw)
        parsed_annotators.append((lbl, recs))

    print("Building hierarchical adjudication table and clustering spans...")
    rows = build_hierarchical_adjudication_table(parsed_annotators)

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n[CSV SAVED] Conflict Resolution Sheet: {out_csv} ({len(df)} rows)")

    out_dossier = Path(args.out_dossier)
    generate_meeting_dossier_text(rows, out_dossier)

    p_counts = df["Adjudication_Priority"].value_counts()
    print("\n" + "=" * 60)
    print("MEETING WORKLOAD & PRIORITY BREAKDOWN:")
    print("=" * 60)
    for p_name, cnt in p_counts.items():
        print(f"  • {p_name:<40}: {cnt:>4} items")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
