# MetaphorTalk: Exploring Metaphor in Human–LLM Dialog

This repository provides the code, scripts, annotation prompts, and evaluation pipelines supporting the paper:

> **A Matter of Interpretation: Exploring Metaphor in Human–LLM Dialog**  
> *Under Review (ACL)*

The repository contains the complete experimental pipeline: from raw ShareChat corpus filtering and window sampling to LLM annotation, inter-annotator agreement (IAA), adjudication, statistical benchmarking, and metaphor uptake analysis.

---

## Data Access & Reproducibility Notice

To protect user privacy and comply with data protection standards, conversational texts containing user interactions are **not hosted directly in this open GitHub repository**. Re-identification and linkage tests demonstrated that full-text and phrase queries against public raw conversational dumps can reveal original web share URLs.

### Accessing the Full Annotated Gold Standard
The adjudicated evaluation corpus (*779 turns, 1,184 metaphor spans*) will be made available to credentialed researchers under an **Academic Data Use Agreement (DUA)** with a strict re-identification and linkage ban:
* **Repository / Gated Access:** Request access via our [Zenodo / Hugging Face Gated Dataset Page](https://huggingface.co/datasets/An0nymia/MetaphorTalk) *(Placeholder link during double-blind review)*.
* **Minimal Mock Data Included:** To test all scripts immediately without waiting for DUA access approval, this repository includes a synthetic mock dataset in [`data/mock/`](data/mock/).

---

## Repository Structure

```text
MetaphorTalk/
├── notebooks/
│   └── 01_corpus_filtering_and_sampling.ipynb   # Raw ShareChat filtering, turn cutoffs (10-100), regex pruning, and 6-8 turn window sampling (§ 3.2, § 4)
├── scripts/
│   ├── run_llm_annotation.py                    # Structured Pydantic LLM annotation runner (KISSKI SAIA / OpenAI / vLLM API) (§ 4.3, App. B.2)
│   ├── corpus_statistics.py                     # Metaphor density, speaker distributions, and sampling representativeness (Table 1)
│   ├── metaphor_uptake.py                       # Metaphor adoption / uptake analysis across speaker directions (Table 4)
│   ├── benchmark_llms.py                        # LLM vs. Gold evaluation across Turn, Span, Lexicalization, and Functions (Table 3)
│   ├── compute_human_iaa.py                     # Human Inter-Annotator Agreement: Fleiss' Kappa, character-level F1, Hungarian alignment (Table 2)
│   ├── prepare_adjudication_sheet.py           # Prepares conflict resolution table & meeting agenda dossier from multi-annotator files (§ 5.1)
│   ├── compile_gold_standard.py                 # Compiles completed adjudication decisions into final Gold Standard format (§ 5.1)
│   └── function_confusion.py                    # 8x8 communicative function confusion matrix & case study extraction (Table 5, App. C)
├── data/
│   └── mock/                                    # Minimal synthetic dialogue data for testing scripts locally
│       ├── mock_turns.csv
│       └── mock_spans.csv
├── requirements.txt                             # Python package dependencies
└── README.md
```

---

## Quickstart & Installation

### 1. Environment Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/An0nymia/MetaphorTalk.git
cd MetaphorTalk
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Testing Scripts on Mock Data
To verify that the code runs without errors on your machine, test the uptake analysis pipeline on the provided mock data:
```bash
python scripts/metaphor_uptake.py --input data/mock/mock_spans.csv --out-dir outputs/
```

---

## Experiments & Analyses on the Released Gold Corpus

Once you have obtained the approved de-identified gold dataset release files (`gold_standard_release_turns.csv`, `gold_standard_release_spans.csv`, `gold_standard_release.json`), place them in the `data/` directory.

### 1. Corpus Statistics & Speaker Distributions (Table 1)
Computes turn counts, word lengths, metaphor counts, normalized density (spans / 100 words), and User vs. LLM distributions:
```bash
python scripts/corpus_statistics.py \
  --turns data/gold_standard_release_turns.csv \
  --spans data/gold_standard_release_spans.csv \
  --out outputs/corpus_statistics_summary.csv
```

### 2. Metaphorical Uptake Dynamics (Table 4)
Analyzes metaphor adoption, repetition, and extension across conversation turns (Human → LLM vs. LLM → Human):
```bash
python scripts/metaphor_uptake.py \
  --input data/gold_standard_release_spans.csv \
  --out-dir outputs/
```

### 3. LLM Annotation & Model Benchmarking (Table 3)
To benchmark an LLM against the human Gold standard:

1. **Run Structured LLM Annotation** (§ 4.3, App. B.2):
```bash
python scripts/run_llm_annotation.py \
  --input data/gold_standard_release_turns.csv \
  --output outputs/my_model_predictions.json \
  --model-name "meta-llama/Llama-3.3-70B-Instruct" \
  --api-base "https://api.example.com/v1" \
  --api-key "$API_KEY"
```

2. **Evaluate Model against Gold Standard**:
```bash
python scripts/benchmark_llms.py \
  --gold data/gold_standard_release.json \
  --predictions outputs/my_model_predictions.json \
  --model-name "Llama-3.3-70B" \
  --out-csv outputs/benchmark_results.csv
```

---

## Multi-Annotator Agreement & Adjudication Pipeline

The repository also includes the full methodological pipeline developed for multi-annotator agreement, span clustering, and expert adjudication. Researchers conducting their own annotation studies can use these tools:

### 1. Human Inter-Annotator Agreement (Table 2)
Computes Turn-level Fleiss' $\kappa$, continuous character-level $F_1$, and Hungarian-aligned conventionality and communicative function metrics:
```bash
python scripts/compute_human_iaa.py \
  --files data/annotator1.json data/annotator2.json data/annotator3.json \
  --labels A1 A2 A3
```

### 2. Adjudication Preparation & Gold Standard Compilation (§ 5.1)
```bash
# Step 1: Cluster spans and generate priority-ranked meeting agenda:
python scripts/prepare_adjudication_sheet.py \
  --files data/annotator1.json data/annotator2.json data/annotator3.json \
  --labels A1 A2 A3 \
  --out-csv outputs/adjudication_sheet.csv \
  --out-dossier outputs/adjudication_dossier.txt

# Step 2: Compile finalized adjudication sheet into Gold standard JSON & CSV:
python scripts/compile_gold_standard.py \
  --sheet data/adjudication_sheet_completed.csv \
  --base-export data/annotator1.json \
  --out-json data/gold_standard_annotations.json \
  --out-csv data/gold_standard_spans.csv
```

### 3. Communicative Function Confusion Matrix (Table 5)
Computes the 8×8 multi-annotator confusion matrix for non-lexicalized communicative functions and extracts illustrative case studies:
```bash
python scripts/function_confusion.py \
  --ann1 data/annotator1.json \
  --ann2 data/annotator2.json \
  --ann3 data/annotator3.json \
  --gold data/gold_standard_release.json \
  --out-dir outputs/
```

---

## Citation & License
The code in this repository is released under the **MIT License**.  
The underlying conversation data is derived from the ShareChat corpus (licensed under CC BY-NC 4.0).
