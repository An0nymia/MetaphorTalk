# MetaphorTalk: Exploring Metaphor in Human–LLM Dialog

This repository provides the code, annotation prompts, documentation,
and evaluation scripts supporting the paper:

**A Matter of Interpretation: Exploring Metaphor in Human–LLM Dialog**  
*Under Review*

The repository documents the computational and annotation workflow,
including corpus filtering and sampling, LLM-assisted annotation,
inter-annotator agreement, adjudication, statistical evaluation, and
metaphor uptake analysis.

The code is openly available. Access to the underlying annotated
conversation data is handled separately because the source conversations
may remain identifiable through text-based linkage.

---

## Data Access and Reproducibility

The repository does not contain the original conversational texts or
the full annotated gold standard. Although identifying information was
manually reviewed and removed or generalized, some utterances remain
searchable in the publicly available source corpus. The resulting
linkage risk means that de-identification cannot be assumed to eliminate
all privacy risks.

The annotated gold standard comprises 779 turns and 1,184 metaphor
spans. Any sharing of the full dataset is subject to institutional
data-protection review and an approved access procedure. If access is
authorized, it will be restricted to approved research purposes under
a data use agreement. No unrestricted public download is promised.

A small synthetic dataset is provided in `data/mock/` to demonstrate
selected scripts. It is intended for software testing, not for
reproducing the paper's empirical findings.

The code and documentation allow inspection and reuse of the methods.
Reproducing analyses that depend on the annotated gold standard requires
authorized access to the corresponding data.

---

## Repository Structure

```text
MetaphorTalk/
├── notebooks/
│   └── corpus_filtering_and_sampling.ipynb   # ShareChat filtering, length filtering, regex pruning, and window sampling
├── scripts/
│   ├── run_llm_annotation.py                 # Structured Pydantic LLM annotation runner via API
│   ├── corpus_statistics.py                  # Metaphor density, speaker distributions, and descriptive statistics
│   ├── metaphor_uptake.py                    # Metaphor adoption and uptake analysis across dialogue turns
│   ├── benchmark_llms.py                     # Evaluation across turn identification, spans, conventionality, and functions
│   ├── compute_human_iaa.py                  # Human agreement metrics: Fleiss' Kappa, character-level F1, Hungarian alignment
│   ├── prepare_adjudication_sheet.py        # Prepares conflict resolution table and meeting agenda dossier
│   ├── compile_gold_standard.py              # Compiles completed adjudication decisions into gold standard format
│   └── function_confusion.py                 # Communicative function confusion matrix and case study extraction
├── data/
│   └── mock/                                 # Minimal synthetic dialogue data for software testing
│       ├── mock_turns.csv
│       └── mock_spans.csv
├── requirements.txt                          # Python package dependencies
└── README.md
```

---

## Installation and Setup

Clone the repository and install the dependencies:

```bash
git clone https://github.com/An0nymia/MetaphorTalk.git
cd MetaphorTalk
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Software Verification with Synthetic Mock Data

A minimal synthetic dataset is provided in `data/mock/` to verify that the scripts execute as intended:

```bash
python scripts/metaphor_uptake.py \
  --input data/mock/mock_spans.csv \
  --out-dir outputs/
```

This synthetic data demonstrates the execution mechanics of the pipeline and does not reflect empirical distributions reported in the paper.

---

## Empirical Evaluation on the Annotated Gold Standard

Reproducing the empirical analyses described in the paper requires access to the corresponding annotated data files (`gold_standard_release_turns.csv`, `gold_standard_release_spans.csv`, `gold_standard_release.json`). Once authorized access is obtained, place these files in the `data/` directory.

### 1. Corpus Statistics and Speaker Distributions
Computes turn counts, word lengths, metaphor counts, normalized density (spans per 100 words), and User versus LLM speaker distributions:

```bash
python scripts/corpus_statistics.py \
  --turns data/gold_standard_release_turns.csv \
  --spans data/gold_standard_release_spans.csv \
  --out outputs/corpus_statistics_summary.csv
```

### 2. Metaphorical Uptake Dynamics
Analyzes metaphor adoption, repetition, and extension across conversational turns (Human-to-LLM versus LLM-to-Human):

```bash
python scripts/metaphor_uptake.py \
  --input data/gold_standard_release_spans.csv \
  --out-dir outputs/
```

### 3. LLM Annotation and Model Benchmarking
To annotate turns using a language model and evaluate the resulting predictions against the gold standard:

1. **Run Structured LLM Annotation**:
```bash
python scripts/run_llm_annotation.py \
  --input data/gold_standard_release_turns.csv \
  --output outputs/model_predictions.json \
  --model-name "meta-llama/Llama-3.3-70B-Instruct" \
  --api-base "https://api.example.com/v1" \
  --api-key "$API_KEY"
```

2. **Evaluate Model Predictions against the Gold Standard**:
```bash
python scripts/benchmark_llms.py \
  --gold data/gold_standard_release.json \
  --predictions outputs/model_predictions.json \
  --model-name "Llama-3.3-70B" \
  --out-csv outputs/benchmark_results.csv
```

---

## Annotation and Adjudication Workflow

The repository also includes the scripts implementing the multi-annotator agreement and adjudication workflow. These scripts allow inspection of the annotation methodology and can be applied to custom multi-annotator studies:

### 1. Human Inter-Annotator Agreement
Computes turn-level agreement (Fleiss' Kappa), character-level continuous span F1, and Hungarian-aligned conventionality and communicative function metrics from independent annotator exports:

```bash
python scripts/compute_human_iaa.py \
  --files data/annotator1.json data/annotator2.json data/annotator3.json \
  --labels A1 A2 A3
```

### 2. Adjudication Preparation and Gold Standard Compilation
Generates structured conflict resolution files for human adjudication and compiles the completed decisions:

```bash
# Step 1: Cluster spans across annotators and generate priority-ranked meeting agenda:
python scripts/prepare_adjudication_sheet.py \
  --files data/annotator1.json data/annotator2.json data/annotator3.json \
  --labels A1 A2 A3 \
  --out-csv outputs/adjudication_sheet.csv \
  --out-dossier outputs/adjudication_dossier.txt

# Step 2: Compile adjudicated decisions into gold standard JSON and CSV formats:
python scripts/compile_gold_standard.py \
  --sheet data/adjudication_sheet_completed.csv \
  --base-export data/annotator1.json \
  --out-json data/gold_standard_annotations.json \
  --out-csv data/gold_standard_spans.csv
```

### 3. Communicative Function Confusion Matrix
Computes the confusion matrix across annotators for communicative functions and extracts illustrative case studies:

```bash
python scripts/function_confusion.py \
  --ann1 data/annotator1.json \
  --ann2 data/annotator2.json \
  --ann3 data/annotator3.json \
  --gold data/gold_standard_release.json \
  --out-dir outputs/
```

---

## Citation and License

The code and accompanying documentation in this repository are
released under the MIT License, except where otherwise noted.

The underlying conversations originate from the ShareChat corpus.
The source corpus's license does not by itself resolve the data-protection
requirements for redistributing the annotated conversational data.
The data are therefore not included in this public repository.