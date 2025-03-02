# Interruption is All You Need: Improving Reasoning Model Refusal Rates through measuring Parallel Reasoning Diversity

This repository contains tools for evaluating large language models on the SimpleQA benchmark, focusing on diversity and divergence of model responses. The pipeline allows generating answers, injecting interruption tokens to create "thinking continuations," and analyzing how these continuations affect answer quality.

## Overview

The system implements a research pipeline for:
1. Generating initial answers to SimpleQA benchmark questions
2. Creating "thinking continuations" (interrupting model reasoning and letting it continue)
3. Measuring diversity and divergence in model responses
4. Using diversity metrics to improve answer accuracy

## Prerequisites

- Python 3.8+
- Required API keys (store in your environment or a `.env` file):
  - `FIREWORKS_API_KEY` (for DeepSeek model access)
  - `GEMINI_API_KEY` (for diversity analysis)
  - `OPENAI_API_KEY` (for evaluation)
  - `ANTHROPIC_API_KEY` (for comparing with Claude models)

## Setup

### 1. Clone this Repo

### 2. Set Up SimpleQA Benchmark

The SimpleQA benchmark is available through the simple-evals repository. You need to either:

```bash
# Option 1: Clone the simple-evals repository
git clone https://github.com/openai/simple-evals.git
cd simple-evals
pip install -e .
```

install the subrepos needed dependencies and add __init__.py files to make it accessible via Python modules.

### 3. Install Required Dependencies

Since there is no requirements.txt file, install the necessary packages:

```bash
pip install openai anthropic google-api-python-client pandas matplotlib numpy tqdm requests
```

Additional packages may be required depending on which components you use.

## Usage

### 1. Generate Initial Answers

Generate baseline answers to SimpleQA questions:

```bash
python driver.py --mode generate --num-examples 50 --responses-file simpleqa_responses.jsonl
```

Options:
- `--num-examples`: Number of SimpleQA examples to process
- `--responses-file`: Output file for responses
- `--slice`: Optional slice of examples (format: 'start:end')

### 2. Generate Thinking Continuations

Create "interrupted thinking" continuations:

```bash
python driver.py --mode continuations --responses-file simpleqa_responses.jsonl --continuations-file simpleqa_continuations.jsonl --num-continuations 3
```

This extracts the "thinking" part of responses (text before "</think>") and creates multiple variation prompts.

### 3. Process Continuations

Generate complete answers for the continuations:

```bash
python driver.py --mode process --continuations-file simpleqa_continuations.jsonl --continuation-responses-file simpleqa_continuations_responses.jsonl
```

This sends the interrupted thinking to the model to complete the reasoning and generate final answers.

### 4. Analyze Diversity with Gemini

Analyze the diversity of model responses using Gemini, making sure to subsitute the apprpriate names:

```bash
python gemini.py"
```

Note: By default, this looks for specific filenames. You can modify the source code to point to your files or modify the function call as needed.

This script:
- Compares continuations to original answers
- Measures reasoning diversity and final answer diversity
- Generates visualizations of diversity metrics
- Saves results to CSV for further analysis

### 5. Apply Diversity Threshold to Improve Accuracy

Apply a threshold to determine when to say "I don't know" based on reasoning diversity:

```bash
python driver.py --mode diversity_threshold --diversity-threshold 7.0
```

Options:
- `--diversity-threshold`: Threshold on a 0-10 scale; questions with reasoning diversity above this will return "I don't know"

### 6. Evaluate Answers

Evaluate the accuracy of generated answers:

```bash
python driver.py --mode evaluate --responses-file simpleqa_diversity_threshold_responses.jsonl
```

### 7. Compare Different Models (Optional)

Compare performance across different models:

```bash
python model_benchmarks.py --num-examples 20 --claude-model claude-3-5-sonnet-20240620 --openai-model gpt-4o
```

## File Format Conversion

Convert between different JSONL formats for evaluation:

```bash
python convert.py --input simpleqa_responses.jsonl --output simpleqa_eval.jsonl
```

## Example Data can be found in hallucination_experiment_data

## Key Files

- `driver.py`: Main script for answer generation and evaluation
- `gemini.py`: Analysis of diversity and divergence using Gemini
- `model_benchmarks.py`: Model comparison utilities
- `embedding.py`: Alternative diversity analysis using embeddings
- `convert.py` & `refactor_qa_responses.py`: Utilities for file format conversion

## Research Pipeline

This system implements a research workflow to study:
1. How diverse are model responses to the same question?
2. How much do interrupted thinking continuations diverge from original responses?
3. Can diversity metrics predict answer correctness?
4. Is there a relationship between reasoning diversity and final answer diversity?

The hypothesis is that high reasoning diversity may indicate model uncertainty, which can be leveraged to improve accuracy by abstaining from answering.

## Troubleshooting

- This is quite a messy repo! Please reach out to David, one of the collaborators if you have any questions!
- If you encounter path or import errors, ensure you're running commands from the root directory of the project
- Check that all required API keys are properly set in your environment
- For file not found errors, verify that all output files from previous steps exist in the expected locations
- If necessary, examine the source code of each script to understand the expected file paths and formats
