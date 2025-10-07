# MedicalExplAIner - Usage Guide

## Overview

MedicalExplAIner is a system that evaluates LLM models on their ability to answer medical questions by decomposing complex queries into more manageable sub-questions.

## System Flow

```
┌─────────────────────────────────────────────────────────────┐
│              MedicalExplAIner - Complete Flow               │
└─────────────────────────────────────────────────────────────┘

1. Load JSON dataset with medical records
   └─> Format: {context, question, answer}

2. For each question:

   a) Generate sub-questions
      └─> LLM decomposes the complex question

   b) Answer each sub-question
      └─> Using the patient's medical context

   c) Synthesize final answer
      └─> Integrate all partial answers

   d) Evaluate the answer
      └─> Compare with expected answer

3. Generate result charts
   └─> Pie charts and bar charts per model
```

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

4. **Configure environment variables** (if using API models)

   Create a `.env` file in the project root:
   ```bash
   # For Google models (Gemini, Gemma)
   GOOGLE_API_KEY=your_api_key_here
   ```

## Basic Usage

### Using Make

The easiest way to run the program is using the Makefile:

**Single model:**
```bash
make run MODELS=gemini-2.5-flash
```

**Multiple models:**
```bash
make run MODELS='gemini-2.5-flash qwen2.5-7b llama3.1-8b'
```

This command is equivalent to:
```bash
uv run python -m medicalexplainer \
    --dataset medicalexplainer/data/test.final.json \
    --models gemini-2.5-flash qwen2.5-7b llama3.1-8b
```

### Main Command (Direct Execution)

```bash
python -m medicalexplainer --dataset <dataset_path> --models <model1> <model2> ...
```

### Parameters

| Parameter | Type | Required | Description | Default |
|-----------|------|----------|-------------|---------|
| `--dataset` | string | ✅ Yes | Path to the JSON dataset file | - |
| `--models` | list | ❌ No | List of models to evaluate | `["gemini-2.0-flash"]` |
| `--tools` | flag | ❌ No | Enable tool usage | `False` |
| `--limit` | int | ❌ No | Limit number of questions | `None` |


## Usage Examples

### 1. Basic Evaluation (single model)

```bash
python -m medicalexplainer \
    --dataset medicalexplainer/data/test.final.json \
    --models gemini-2.0-flash
```

### 2. Multiple Model Evaluation

```bash
python -m medicalexplainer \
    --dataset medicalexplainer/data/test.final.json \
    --models gemini-2.0-flash qwen2.5-7b llama3.1-8b
```

### 3. Evaluation with Limit (for quick testing)

```bash
python -m medicalexplainer \
    --dataset medicalexplainer/data/test.final.json \
    --models gemini-2.0-flash \
    --limit 10
```

### 4. Evaluation with Tools (not recommended for medical context)

```bash
python -m medicalexplainer \
    --dataset medicalexplainer/data/test.final.json \
    --models gemini-2.0-flash \
    --tools
```

## Dataset Format

The dataset must be a JSON file with the following format (SQuAD-like):

## Dataset Format

The dataset must be a JSON file with the following format (SQuAD-like):

```json
{
    "data": [
        {
            "title": "case_ID",
            "paragraphs": [
                {
                    "context": "Complete patient medical history...",
                    "qas": [
                        {
                            "question": "Does the patient have diabetes?",
                            "id": "0",
                            "answers": [
                                {
                                    "answer_start": 123,
                                    "text": "The patient has type 2 diabetes"
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    ]
}
```

## Results

Results are saved in `medicalexplainer/data/evaluation/{model}/`:

### Generated Files

1. **`answers.txt`**: Detailed results per question
   ```
   Model: gemini-2.0-flash
   Correct and incorrect answers:
   Question 0: Correct: 8, Incorrect: 2
   Question 1: Correct: 7, Incorrect: 3
   ...
   ```

2. **`grouped_bar_answers.png`**: Grouped bar chart
   - Correct answers (green)
   - Incorrect answers (red)
   - Per question

3. **`answers_pie_chart.txt`**: Global percentages
   ```
   Model: gemini-2.0-flash
   Correct and incorrect answers:
   Correct (YES): 75.5%
   Incorrect (NO): 20.3%
   Problematic (PROBLEM): 4.2%
   ```

4. **`answers_pie_chart.png`**: Pie chart
   - Global distribution of answers

## Adding New Models

To add a new model, edit `medicalexplainer/llm.py`:

```python
# 1. Create a class for the model
class LLM_YOUR_MODEL(LLM):
    def __init__(self, tools: bool = False):
        super().__init__(tools)
        self.model = "model-name"
        self.tools = tools

        llm = ChatOllama(  # or ChatGoogleGenerativeAI, etc.
            model=self.model,
            num_ctx=32768,
        )

        self.llm = llm
        logger.debug("Using Your Model LLM")

# 2. Add to the models dictionary
models = {
    # ... existing models ...
    "your-model": (LLM_YOUR_MODEL, "big"),  # "big" or "small"
}
```

## Logs

Logs are saved in `medicalexplainer/logs/medicalexplainer.log`:

- **DEBUG**: Detailed information for each step
- **INFO**: General progress and results
- **ERROR**: Errors and exceptions

To view logs in real-time:
```bash
tail -f medicalexplainer/logs/medicalexplainer.log
```

## Troubleshooting

### Error: "Dataset file not found"
- Verify that the dataset path is correct
- Use absolute paths if you have issues with relative paths

### Error: "Model not found in models dictionary"
- Verify that the model name is in the `models` dictionary in `llm.py`
- Check the spelling of the model name

### Error with Ollama models
- Make sure Ollama is installed: `ollama --version`
- Download the model: `ollama pull model-name`
- Verify that Ollama is running: `ollama list`

### Error with API models (Gemini)
- Verify that the API key is configured in `.env`
- Verify that you have available credits
- Check the API rate limits

### Answers with "PROBLEM"
- May be API timeout errors
- May be response parsing issues
- Check the logs for more details

## Contributing

To contribute to the project:

1. Fork the repository
2. Create a branch with your feature: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature/new-feature`
5. Create Pull Request

## License

See `LICENSE` file for more details.

## Contact

- GitHub: [@hossam1522](https://github.com/hossam1522)
- Repository: [MedicalExplAIner](https://github.com/hossam1522/MedicalExplAIner)
