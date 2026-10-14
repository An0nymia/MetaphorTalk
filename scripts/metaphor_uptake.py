import re
import argparse
import spacy
import pandas as pd
from pathlib import Path


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Maximum number of turns after an introduction in which uptake is considered.
# 8 turns corresponds roughly to the local conversational context used in
# your annotation setup.
MAX_UPTAKE_DISTANCE = 8

# Minimum number of shared content lemmas for a strong lexical match.
MIN_SHARED_LEMMAS = 2

# Minimum proportion of the shorter span's content lemmas that must overlap.
MIN_COVERAGE = 0.50

# spaCy model
SPACY_MODEL = "en_core_web_sm"


# ==============================================================================
# 1. LOAD SPACY
# ==============================================================================

print(f"Loading spaCy model: {SPACY_MODEL}")

try:
    nlp = spacy.load(SPACY_MODEL)
except OSError:
    raise OSError(
        f"spaCy model '{SPACY_MODEL}' is not installed.\n"
        f"Install it with:\n\n"
        f"    python -m spacy download {SPACY_MODEL}\n"
    )



# ==============================================================================
# 2. LOAD GOLD STANDARD SPANS
# ==============================================================================

parser = argparse.ArgumentParser(description="Analyze metaphorical uptake in human-LLM dialog.")
parser.add_argument(
    "--input",
    type=str,
    default=None,
    help="Path to gold standard spans CSV (e.g. data/gold_standard_release_spans.csv or data/mock/mock_spans.csv)",
)
parser.add_argument(
    "--out-dir",
    type=str,
    default=".",
    help="Directory to save output tables and visualizations",
)
args, _ = parser.parse_known_args()

if args.input:
    csv_file = Path(args.input)
else:
    csv_paths = [
        Path("data/gold/release/gold_standard_release_spans.csv"),
        Path("data/gold_standard_release_spans.csv"),
        Path("data/mock/mock_spans.csv"),
        Path("LLM_Anno/data/gold/gold_standard_spans.csv"),
        Path("../LLM_Anno/data/gold/gold_standard_spans.csv"),
    ]
    csv_file = next((p for p in csv_paths if p.exists()), None)

if csv_file is None or not csv_file.exists():
    raise FileNotFoundError(
        "Could not find metaphor spans CSV. Please pass --input path/to/spans.csv"
    )

print(f"Loading metaphor spans from: {csv_file}")
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_file)


# ==============================================================================
# 3. BASIC VALIDATION
# ==============================================================================

required_columns = {
    "Turn_ID",
    "Conv_ID",
    "Speaker",
    "Span_Text",
    "Lexicalized",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )


# ==============================================================================
# 4. TURN INDEX
# ==============================================================================

def get_turn_index(turn_id):
    """
    Extract turn index from IDs such as:

        utt_main_1022_4 -> 4
    """

    parts = str(turn_id).split("_")

    if parts[-1].isdigit():
        return int(parts[-1])

    return None


df["Turn_Index"] = df["Turn_ID"].apply(get_turn_index)

if df["Turn_Index"].isna().any():
    print(
        "WARNING:",
        df["Turn_Index"].isna().sum(),
        "rows have no valid turn index."
    )


# ==============================================================================
# 5. METAPHOR TYPE
# ==============================================================================

df["Is_Lexicalized"] = (
    df["Lexicalized"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("true")
)

df["Type"] = df["Is_Lexicalized"].map({
    True: "Lexicalized",
    False: "Non-lexicalized",
})


# ==============================================================================
# 6. NORMALIZE SPEAKER LABELS
# ==============================================================================

def normalize_speaker(speaker):

    value = str(speaker).strip().lower()

    if value in {
        "user",
        "human",
        "human_user",
        "person",
    }:
        return "User"

    if value in {
        "assistant",
        "llm",
        "chatgpt",
        "gpt",
        "model",
    }:
        return "LLM"

    # Preserve unexpected labels so they can be inspected.
    return str(speaker).strip()


df["Speaker_Normalized"] = df["Speaker"].apply(normalize_speaker)

unexpected_speakers = sorted(
    set(df["Speaker_Normalized"])
    - {"User", "LLM"}
)

if unexpected_speakers:
    print(
        "\nWARNING: Unexpected speaker labels:",
        unexpected_speakers
    )


# ==============================================================================
# 7. TEXT NORMALIZATION
# ==============================================================================

def normalize_surface(text):
    """
    Normalize surface text for exact-ish comparison.
    """

    text = str(text).lower().strip()

    # Normalize typographic apostrophes
    text = text.replace("’", "'")
    text = text.replace("‘", "'")

    # Normalize quotation marks
    text = text.replace("“", '"')
    text = text.replace("”", '"')

    # Normalize dashes
    text = re.sub(r"[-–—]", "-", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text


# ==============================================================================
# 8. SPACY SPAN ANALYSIS
# ==============================================================================

span_cache = {}


def analyze_span(text):
    """
    Linguistically analyze a metaphorical span using spaCy.

    Returns:
        normalized surface form
        all lemmas
        content lemmas
        content lemma set
        POS information
    """

    text = str(text)

    if text in span_cache:
        return span_cache[text]

    doc = nlp(text)

    normalized_tokens = []
    content_lemmas = []
    pos_tags = []

    for token in doc:

        if token.is_space:
            continue

        if token.is_punct:
            continue

        lemma = token.lemma_.lower().strip()

        if not lemma:
            continue

        normalized_tokens.append(lemma)

        pos_tags.append(token.pos_)

        # Keep lexical content words.
        #
        # We deliberately exclude:
        # - stopwords
        # - punctuation
        # - very short tokens
        #
        # We retain:
        # NOUN, PROPN, VERB, ADJ, ADV
        if (
            token.is_alpha
            and not token.is_stop
            and len(lemma) > 2
            and token.pos_ in {
                "NOUN",
                "PROPN",
                "VERB",
                "ADJ",
                "ADV",
            }
        ):
            content_lemmas.append(lemma)

    result = {
        "normalized": " ".join(normalized_tokens),
        "content_lemmas": content_lemmas,
        "content_set": set(content_lemmas),
        "pos": pos_tags,
    }

    span_cache[text] = result

    return result


# ==============================================================================
# 9. SPAN MATCHING
# ==============================================================================

def match_span(span_a, span_b):
    """
    Determine whether two metaphorical spans may represent uptake.

    Returns one of:

        exact
        lexical
        possible
        none

    Definitions:

        exact
            Same normalized linguistic sequence.

        lexical
            Strong overlap in content lemmas.

        possible
            Weaker overlap that should be manually inspected.

        none
            No meaningful lexical overlap.
    """

    a = analyze_span(span_a)
    b = analyze_span(span_b)

    # --------------------------------------------------------------------------
    # Exact normalized match
    # --------------------------------------------------------------------------

    if a["normalized"] == b["normalized"]:
        return "exact"

    A = a["content_set"]
    B = b["content_set"]

    if not A or not B:
        return "none"

    shared = A & B

    if not shared:
        return "none"

    # --------------------------------------------------------------------------
    # Coverage of shorter expression
    # --------------------------------------------------------------------------

    shorter = min(len(A), len(B))

    coverage = len(shared) / shorter

    # --------------------------------------------------------------------------
    # Strong lexical reuse
    # --------------------------------------------------------------------------

    if (
        len(shared) >= MIN_SHARED_LEMMAS
        and coverage >= MIN_COVERAGE
    ):
        return "lexical"

    # --------------------------------------------------------------------------
    # One shared lemma with very high coverage.
    #
    # These are deliberately retained as candidates rather than treated
    # as confirmed uptake.
    # --------------------------------------------------------------------------

    if (
        len(shared) == 1
        and coverage >= 0.75
    ):
        return "possible"

    return "none"


# ==============================================================================
# 10. GROUP SPANS BY CONVERSATION
# ==============================================================================

print("\nAnalyzing conversations...")

adoption_events = []


for conv_id, group in df.groupby("Conv_ID"):

    group = (
        group
        .sort_values(
            ["Turn_Index"]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------------------------
    # Iterate over each metaphorical span as a potential introduction.
    # --------------------------------------------------------------------------

    for i, source in group.iterrows():

        source_turn = source["Turn_Index"]
        source_speaker = source["Speaker_Normalized"]
        source_span = source["Span_Text"]
        source_type = source["Type"]

        # Skip malformed rows
        if pd.isna(source_turn):
            continue

        # ----------------------------------------------------------------------
        # Determine whether the metaphorical expression/framing was already
        # used earlier in the conversation.
        #
        # IMPORTANT:
        # We check both speakers. If the metaphor was already introduced by
        # someone earlier, the current occurrence is not treated as a new
        # introduction.
        # ----------------------------------------------------------------------

        previously_introduced = False
        previous_turn = None
        previous_speaker = None
        previous_span = None

        for j in range(i):

            previous = group.iloc[j]

            if pd.isna(previous["Turn_Index"]):
                continue

            match_type = match_span(
                source_span,
                previous["Span_Text"]
            )

            if match_type != "none":

                previously_introduced = True
                previous_turn = previous["Turn_Index"]
                previous_speaker = previous["Speaker_Normalized"]
                previous_span = previous["Span_Text"]

                break

        # ----------------------------------------------------------------------
        # If this metaphor was already introduced, we do not count it as a
        # fresh introduction.
        # ----------------------------------------------------------------------

        if previously_introduced:

            continue

        # ----------------------------------------------------------------------
        # Search subsequent turns by the OTHER speaker.
        # ----------------------------------------------------------------------

        candidates = []

        for j in range(i + 1, len(group)):

            target = group.iloc[j]

            target_turn = target["Turn_Index"]
            target_speaker = target["Speaker_Normalized"]

            if pd.isna(target_turn):
                continue

            # Conversation window
            if target_turn - source_turn > MAX_UPTAKE_DISTANCE:
                break

            # Do not count self-repetition
            if target_speaker == source_speaker:
                continue

            match_type = match_span(
                source_span,
                target["Span_Text"]
            )

            if match_type == "none":
                continue

            candidates.append({
                "target": target,
                "match_type": match_type,
            })

        # ----------------------------------------------------------------------
        # Select earliest uptake candidate.
        # ----------------------------------------------------------------------

        if candidates:

            best = candidates[0]

            target = best["target"]

            adopted = True
            match_type = best["match_type"]

            adopting_turn = target["Turn_Index"]
            adopting_span = target["Span_Text"]
            adopting_type = target["Type"]

            turn_distance = (
                adopting_turn - source_turn
            )

        else:

            adopted = False
            match_type = "none"

            adopting_turn = None
            adopting_span = None
            adopting_type = None
            turn_distance = None

        # ----------------------------------------------------------------------
        # Direction
        # ----------------------------------------------------------------------

        if source_speaker == "User":
            direction = "User → LLM"

        elif source_speaker == "LLM":
            direction = "LLM → User"

        else:
            direction = f"{source_speaker} → Other"

        # ----------------------------------------------------------------------
        # Store event
        # ----------------------------------------------------------------------

        adoption_events.append({

            "Conv_ID": conv_id,

            "Direction": direction,

            "Speaker": source_speaker,

            "Type": source_type,

            "Span_Text": source_span,

            "Turn_Index": source_turn,

            "Previously_Introduced": previously_introduced,

            "Previous_Turn": previous_turn,

            "Previous_Speaker": previous_speaker,

            "Previous_Span": previous_span,

            "Adopted": adopted,

            "Match_Type": match_type,

            "Adopting_Turn": adopting_turn,

            "Adopting_Span": adopting_span,

            "Adopting_Type": adopting_type,

            "Turn_Distance": turn_distance,

        })


df_adopt = pd.DataFrame(adoption_events)


# ==============================================================================
# 11. BASIC COUNTS
# ==============================================================================

print("\n" + "=" * 80)
print("BASIC UPTAKE COUNTS")
print("=" * 80)

print(
    f"Potential metaphor introductions: {len(df_adopt)}"
)

print(
    f"Metaphors with subsequent uptake: "
    f"{df_adopt['Adopted'].sum()}"
)

if len(df_adopt) > 0:

    uptake_rate = (
        100
        * df_adopt["Adopted"].sum()
        / len(df_adopt)
    )

    print(
        f"Overall uptake rate: {uptake_rate:.1f}%"
    )


# ==============================================================================
# 12. SUMMARY FUNCTION
# ==============================================================================

def adoption_summary(group):

    introduced = len(group)

    adopted = int(
        group["Adopted"].sum()
    )

    rate = (
        100 * adopted / introduced
        if introduced > 0
        else 0.0
    )

    return pd.Series({

        "Introduced": introduced,

        "Adopted": adopted,

        "Adoption_Rate": rate,

    })


# ==============================================================================
# 13. SUMMARY BY SPEAKER DIRECTION × METAPHOR TYPE
# ==============================================================================

summary = (
    df_adopt
    .groupby(
        ["Direction", "Type"],
        dropna=False
    )
    .apply(
        adoption_summary,
        include_groups=False
    )
    .reset_index()
)


print("\n" + "=" * 80)
print("METAPHOR UPTAKE BY SPEAKER DIRECTION AND METAPHOR TYPE")
print("=" * 80)

print(
    summary[
        [
            "Direction",
            "Type",
            "Introduced",
            "Adopted",
            "Adoption_Rate",
        ]
    ].to_string(index=False)
)


# ==============================================================================
# 14. OVERALL SUMMARY BY METAPHOR TYPE
# ==============================================================================

type_summary = (
    df_adopt
    .groupby(
        "Type",
        dropna=False
    )
    .apply(
        adoption_summary,
        include_groups=False
    )
    .reset_index()
)


print("\n" + "=" * 80)
print("OVERALL UPTAKE BY METAPHOR TYPE")
print("=" * 80)

print(
    type_summary[
        [
            "Type",
            "Introduced",
            "Adopted",
            "Adoption_Rate",
        ]
    ].to_string(index=False)
)


# ==============================================================================
# 15. UPTAKE TYPE
# ==============================================================================

match_summary = (
    df_adopt[
        df_adopt["Adopted"]
    ]
    .groupby(
        ["Type", "Match_Type"]
    )
    .size()
    .reset_index(
        name="Count"
    )
)


print("\n" + "=" * 80)
print("TYPE OF METAPHORICAL UPTAKE")
print("=" * 80)

if len(match_summary) > 0:
    print(
        match_summary.to_string(
            index=False
        )
    )
else:
    print("No uptake candidates found.")


# ==============================================================================
# 16. UPTAKE BY DIRECTION AND MATCH TYPE
# ==============================================================================

direction_match_summary = (
    df_adopt[
        df_adopt["Adopted"]
    ]
    .groupby(
        [
            "Direction",
            "Type",
            "Match_Type",
        ]
    )
    .size()
    .reset_index(
        name="Count"
    )
)


print("\n" + "=" * 80)
print("UPTAKE TYPE BY DIRECTION")
print("=" * 80)

if len(direction_match_summary) > 0:
    print(
        direction_match_summary.to_string(
            index=False
        )
    )
else:
    print("No uptake candidates found.")


# ==============================================================================
# 17. TURN-DISTANCE DISTRIBUTION
# ==============================================================================

distance_summary = (
    df_adopt[
        df_adopt["Adopted"]
        & df_adopt["Turn_Distance"].notna()
    ]
    .groupby("Type")["Turn_Distance"]
    .agg(
        Count="count",
        Mean="mean",
        Median="median",
        Min="min",
        Max="max",
    )
    .reset_index()
)


print("\n" + "=" * 80)
print("TURN DISTANCE OF METAPHORICAL UPTAKE")
print("=" * 80)

if len(distance_summary) > 0:
    print(
        distance_summary.to_string(
            index=False
        )
    )


# ==============================================================================
# 18. NON-LEXICALIZED METAPHOR UPTAKE CANDIDATES
# ==============================================================================

creative_candidates = df_adopt[
    (
        df_adopt["Type"]
        == "Non-lexicalized"
    )
    &
    (
        df_adopt["Adopted"]
    )
].copy()


print("\n" + "=" * 80)
print(
    "NON-LEXICALIZED METAPHOR UPTAKE CANDIDATES"
)
print("=" * 80)

print(
    f"Candidates found: "
    f"{len(creative_candidates)}"
)


for _, row in creative_candidates.head(30).iterrows():

    print("\n" + "-" * 70)

    print(
        f"[{row['Direction']}] "
        f"Conversation: {row['Conv_ID']}"
    )

    print(
        f"Turn {row['Turn_Index']} "
        f"({row['Speaker']}):"
    )

    print(
        f'  "{row["Span_Text"]}"'
    )

    print(
        f"→ Turn {row['Adopting_Turn']}:"
    )

    print(
        f'  "{row["Adopting_Span"]}"'
    )

    print(
        f"Match type: {row['Match_Type']}"
    )

    print(
        f"Turn distance: "
        f"{row['Turn_Distance']}"
    )


# ==============================================================================
# 19. EXPORT DATA FOR MANUAL VALIDATION
# ==============================================================================

# Full event-level dataset
df_adopt.to_csv(
    "metaphor_uptake_events.csv",
    index=False
)

# Summary tables
summary.to_csv(
    "metaphor_uptake_summary.csv",
    index=False
)

type_summary.to_csv(
    "metaphor_uptake_by_type.csv",
    index=False
)

match_summary.to_csv(
    "metaphor_uptake_match_types.csv",
    index=False
)

direction_match_summary.to_csv(
    "metaphor_uptake_by_direction.csv",
    index=False
)

distance_summary.to_csv(
    "metaphor_uptake_turn_distance.csv",
    index=False
)

# Specifically useful for manual coding
creative_candidates.to_csv(
    "creative_metaphor_uptake_candidates.csv",
    index=False
)


print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    "  metaphor_uptake_events.csv"
)

print(
    "  metaphor_uptake_summary.csv"
)

print(
    "  metaphor_uptake_by_type.csv"
)

print(
    "  metaphor_uptake_match_types.csv"
)

print(
    "  metaphor_uptake_by_direction.csv"
)

print(
    "  metaphor_uptake_turn_distance.csv"
)

print(
    "  creative_metaphor_uptake_candidates.csv"
)