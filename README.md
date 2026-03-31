# MedicalExplAIner

Predict ESI triage acuity levels (1-5) using Large Language Models on [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/2.2/) emergency department data.

## Overview

MedicalExplAIner feeds patient data from MIMIC-IV-ED (demographics, triage vitals, ED vital signs, and diagnosis codes) to an LLM and asks it to predict the Emergency Severity Index (ESI) triage acuity level on a scale of 1 (most severe) to 5 (least severe). Predictions are compared against the ground-truth `acuity` column from the triage table.

Key features:

- **Dynamic Ollama model support** -- any Ollama model can be used without code changes; models are auto-pulled if not present locally.
- **Google API models** -- Gemini and Gemma models are also supported.
- **Reasoning model support** -- DeepSeek-R1 and other thinking models are detected automatically; use `--no-think` to skip the reasoning chain for faster inference.
- **Logprobs** -- for Ollama models, log-probabilities for each acuity level (1-5) are collected.
- **ESI v4 algorithm** -- the full Emergency Severity Index v4 decision tree is embedded in every prompt, including physiological thresholds and resource-count logic.
- **Variable selection** -- choose which patient variables to include in the prompt.
- **Sub-question decomposition** -- optionally break the prediction into sub-questions before making a final determination.
- **Structured CSV output** -- results include model name, patient identifiers, prediction, ground truth, logprobs, and sub-question details.
- **Accuracy summary** -- a per-model accuracy table is printed to the console at the end of each evaluation run.
- **Retry logic with exponential backoff** -- handles transient API/service failures gracefully.

## Data

This project uses the **MIMIC-IV-ED v2.2** dataset. Specifically, these four tables:

| Table | Description |
|-------|-------------|
| `edstays.csv` | Patient demographics and ED stay info |
| `triage.csv` | Triage vitals, chief complaint, and **acuity** (ground truth) |
| `vitalsign.csv` | Time-series vital signs during the ED stay (aggregated to median) |
| `diagnosis.csv` | ICD diagnosis codes |

### Demo data (free)

The demo subset is publicly available and can be downloaded automatically:

```bash
make download-demo-data
```

This runs:
```bash
wget -r -np -nd -A "*.csv.gz" -P data/ https://physionet.org/files/mimic-iv-ed-demo/2.2/ed/
gunzip -f data/*.csv.gz
```

### Full dataset (credentialed access)

The complete MIMIC-IV-ED v2.2 dataset (~425,000 ED stays) requires [credentialed access on PhysioNet](https://physionet.org/content/mimic-iv-ed/2.2/). After completing the required training and signing the Data Use Agreement:

1. Download the CSV files from PhysioNet.
2. Place `edstays.csv`, `triage.csv`, `vitalsign.csv`, and `diagnosis.csv` in the `data/` directory.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/hossam1522/MedicalExplAIner.git
   cd MedicalExplAIner
   ```

2. **Install uv** (Python package manager)
   ```bash
   make install-uv
   ```

3. **Install dependencies**
   ```bash
   make install
   ```

4. **Configure environment variables**

   Create a `.env` file in the project root:
   ```bash
   GOOGLE_API_KEY=your_api_key_here  # only needed for Google API models
   OLLAMA_HOST=localhost:11434        # optional; defaults to localhost:11434
   ```

5. **Ensure Ollama is running** (if using Ollama models)
   ```bash
   ollama serve
   ```

## Usage

### Using Make

**Single Ollama model (with sub-questions):**
```bash
make run MODELS=llama3.1
```

**Multiple models (direct prediction, no sub-questions):**
```bash
make run-nodiv MODELS='llama3.1 gemma3:4b mistral'
```

**Reasoning model with thinking disabled:**
```bash
make run-nothink MODELS=deepseek-r1:8b
```

**Reasoning model, no sub-questions, no thinking:**
```bash
make run-nodiv-nothink MODELS=deepseek-r1:8b
```

### Direct execution

```bash
python -m medicalexplainer --models <model1> [model2 ...] [options]
```

### Parameters

| Parameter | Type | Required | Description | Default |
|-----------|------|----------|-------------|---------|
| `--models` | list | Yes | Model names (Ollama models are auto-pulled) | -- |
| `--subtasks` | flag | No | Decompose into sub-questions before predicting | Off |
| `--no-think` | flag | No | Disable thinking chain on reasoning models (faster) | Off |
| `--limit` | int | No | Max number of patient records to evaluate | All |
| `--variables` | list | No | Subset of variables to include | All |
| `--data-dir` | string | No | Directory containing CSV files | `data/` |

### Available variables

Variables are grouped by source table:

**edstays:** `gender`, `race`, `arrival_transport`, `disposition`

**triage:** `temperature`, `heartrate`, `resprate`, `o2sat`, `sbp`, `dbp`, `pain`, `chiefcomplaint`

**vitalsign (aggregated):** `vs_temperature`, `vs_heartrate`, `vs_resprate`, `vs_o2sat`, `vs_sbp`, `vs_dbp`, `vs_rhythm`, `vs_pain`

**diagnosis:** `diagnoses`

### Examples

**Evaluate with all variables (default):**
```bash
python -m medicalexplainer --models llama3.1
```

**Evaluate with only triage data:**
```bash
python -m medicalexplainer \
    --models llama3.1 \
    --variables temperature heartrate resprate o2sat sbp dbp pain chiefcomplaint
```

**Evaluate with sub-question decomposition, limited to 10 records:**
```bash
python -m medicalexplainer \
    --models llama3.1 gemma3:4b \
    --subtasks \
    --limit 10
```

**Use a reasoning model with thinking disabled:**
```bash
python -m medicalexplainer --models deepseek-r1:8b --no-think
```

**Use a Google API model:**
```bash
python -m medicalexplainer --models gemini-2.0-flash
```

## Ollama configuration

### Custom host (`OLLAMA_HOST`)

By default the tool connects to Ollama at `localhost:11434`.  If your Ollama
instance runs on a different address (e.g. a remote GPU server), set
`OLLAMA_HOST` before running:

```bash
export OLLAMA_HOST=0.0.0.0:11436          # no scheme required
# or
export OLLAMA_HOST=http://gpu-server:11434
```

The value can be set in `.env` as well.

### Reasoning models (`--no-think`)

Reasoning models (e.g. DeepSeek-R1, QwQ) are detected automatically by
querying Ollama's `/api/show` endpoint for the `"thinking"` capability.  When
such a model is used:

- By default (`think=True`) the model produces a hidden chain-of-thought before
  the visible answer.  This is slower but generally more accurate.
- With `--no-think` the reasoning chain is skipped entirely.  Inference is
  significantly faster (similar to a standard model) at the cost of some
  deliberateness.

```bash
# With thinking (default)
python -m medicalexplainer --models deepseek-r1:8b

# Without thinking (faster)
python -m medicalexplainer --models deepseek-r1:8b --no-think
```

## Results

Results are saved as CSV files in the `results/` directory with the naming pattern:

```
results/results_{mode}_{timestamp}.csv
```

Where `{mode}` is either `direct` or `subtasks`.

At the end of each run a per-model accuracy summary table is printed to the
console, showing correct predictions, total records evaluated, and accuracy
percentage for every model in the run.

### CSV columns

| Column | Description |
|--------|-------------|
| `model` | Model name used for prediction |
| `subject_id` | Patient identifier |
| `stay_id` | ED stay identifier |
| `ground_truth_acuity` | Actual ESI acuity level (1-5) from triage |
| `predicted_acuity` | LLM's predicted acuity level |
| `correct` | Whether prediction matches ground truth |
| `use_subtasks` | Whether sub-question decomposition was used |
| `subquestions` | JSON array of sub-questions (if subtasks enabled) |
| `subanswers` | JSON array of sub-answers (if subtasks enabled) |
| `logprobs_1` .. `logprobs_5` | Log-probability for each acuity level |
| `prob_1` .. `prob_5` | Probability (exp of logprob) for each acuity level |

## Adding models

### Ollama models

No code changes needed. Just pass the model name:

```bash
python -m medicalexplainer --models any-ollama-model-name
```

If the model is not already pulled, it will be downloaded automatically.

### Google API models

Google API models are defined in `medicalexplainer/llm.py` in the `API_MODELS` dictionary. To add a new one:

```python
API_MODELS["your-model-name"] = {
    "backend": "google",
    "model_id": "the-google-model-id",
    "temperature": 0,
}
```

## Project structure

```
MedicalExplAIner/
├── data/                     # MIMIC-IV-ED CSV files (not committed)
│   ├── edstays.csv
│   ├── triage.csv
│   ├── vitalsign.csv
│   └── diagnosis.csv
├── results/                  # Evaluation output (not committed)
├── medicalexplainer/
│   ├── __init__.py           # Package exports
│   ├── __main__.py           # CLI entry point
│   ├── dataset.py            # CSV loading, merging, aggregation
│   ├── evaluator.py          # Evaluation pipeline, CSV output
│   ├── llm.py                # LLM wrapper (Ollama + Google API)
│   ├── logger.py             # Logging configuration
│   └── paths.py              # Centralised path constants
├── tests/
│   ├── test_dataset.py
│   ├── test_llm.py
│   └── test_logger.py
├── Makefile
├── pyproject.toml
└── README.md
```

## Running tests

```bash
make test
```

## Citation

When using MIMIC-IV-ED data, please cite:

> Johnson, A., Bulgarelli, L., Pollard, T., Celi, L. A., Mark, R., & Horng, S. (2023). MIMIC-IV-ED (version 2.2). *PhysioNet*. https://doi.org/10.13026/5ntk-km72

## License

See `LICENSE` file for details.
