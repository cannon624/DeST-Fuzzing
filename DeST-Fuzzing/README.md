# DeST-Fuzzing: Defense-State Transition Optimization for Stable Jailbreak Discovery

Official implementation of the paper **"DeST-Fuzzing: Defense-State Transition Optimization for Stable Jailbreak Discovery in Large Language Models"** (ACM CCS 2026).

## Overview

DeST-Fuzzing is an automated jailbreak fuzzing framework that models target-model responses as **four ordered defense states** (Complete Refusal, Partial Refusal, Partial Acceptance, Full Acceptance) and uses **boundary potential** and **transition-aware mutation** to guide the search for effective jailbreak templates.

Key features:
- **Defense-State Estimation**: Maps responses into 4 states with boundary potential scoring
- **Transition-Aware Operator Selection**: State-conditioned UCB for choosing mutation operators
- **Stability-Controlled Tree Search**: Uncertainty-gated value backup + dynamic intermediate expansion
- **7 Mutation Operators**: 4 Radical Exploration + 3 Conservative Infiltration

---

## Directory Structure

```
.
├── ACM.pdf                          # Paper
├── README.md
├── requirements.txt                 # Python dependencies
├── Info-for-test.txt                # API & model configuration
├── destfuzzinguzz.py                       # CLI entry point (argparse)
├── run_full_experiments.py          # Full experiment suite (5 experiments)
├── test_minimal.py                  # Quick smoke test (3 questions, 3 seeds)
├── destfuzzing/                       # Core library
│   ├── fuzzer/
│   │   ├── core.py                  # destfuzzing, PromptNode, Algorithm 1
│   │   ├── mutator.py               # 7 mutation operators + policies
│   │   └── selection.py             # Node selection strategies (MCTS, EnhancedMCTS)
│   ├── llm/
│   │   └── llm.py                   # LLM interfaces (API, Local, VLLM)
│   └── utils/
│       ├── predict.py               # Defense-state estimators (GPT + RoBERTa)
│       ├── template.py              # Prompt template synthesis
│       └── openai.py                # OpenAI API helpers
├── datasets/
│   ├── prompts/destfuzzing.csv        # 80 seed jailbreak templates
│   ├── questions/question_list.csv  # 100 malicious evaluation queries
│   ├── responses/                   # Initial target-model responses
│   └── responses_labeled/           # Human-annotated response labels
└── example/
    ├── finetune_roberta.py          # Fine-tune RoBERTa defense-state classifier
    ├── validate.py                  # Validate trained classifier
    └── output_example.csv           # Example output format
```

---

## Environment Setup

### Requirements

- **Python** >= 3.8 (tested on 3.13)
- **CUDA** (optional, required only for local LLM inference or RoBERTa training)

### 1. Install Dependencies

All dependencies are listed in `requirements.txt`.

**Core dependencies** (required for all experiments):

```bash
pip install -r requirements.txt
```

This installs: `pandas`, `numpy`, `openai`, `requests`, `tqdm`.

**Local LLM support** (optional, for running open-source target models):

```bash
pip install torch fschat vllm
```

**RoBERTa training** (optional, for training your own defense-state classifier):

```bash
pip install transformers datasets evaluate accelerate scikit-learn
```

### 2. Configure API Keys

Edit `Info-for-test.txt`:

```
model-name: gpt-3.5-turbo          # Mutator model (GPT-3.5-Turbo recommended)
API: sk-your-api-key-here           # Your OpenAI-compatible API key
API地址:                            # Custom base URL (optional)
judge-model-path:                   # Path to trained RoBERTa model (optional; uses GPT judge if empty)
```

| Field | Description |
|-------|-------------|
| `model-name` | Model name for the mutator LLM |
| `API` | API key for OpenAI-compatible service |
| `API地址` | Custom API base URL (leave empty for default) |
| `judge-model-path` | Path to fine-tuned RoBERTa defense-state classifier. If empty, uses **GPT-based judge** |

---

## How to Run

### Quick Test (smoke test)

Runs DeST-Fuzzing on 3 questions with 3 seeds, 15-query budget:

```bash
python test_minimal.py
```

### Full Experiment Suite

Runs all 5 experiments (100 questions x 80 seeds):

```bash
python run_full_experiments.py
```

Results are saved to `full_experiment_results_<timestamp>/`:

| Experiment | Output File | Description |
|-----------|-------------|-------------|
| Exp 1: Static Seeds | `exp1_static_full.csv`, `exp1_metrics.json` | Baseline seed effectiveness (Table 2) |
| Exp 2: Full Search | `exp2_search.json` | Hard-subset automated search (Table 3) |
| Exp 3: Ablation | `exp3_ablation.json` | Component ablation (Table 4/5) |
| Exp 4: Operators | `exp4_operators.json` | Operator transition diagnostics (Figure 6) |
| Exp 5: Hyperparams | `exp5_hyperparams.json` | Hyperparameter sensitivity (Table 6) |

### CLI Usage (Advanced)

For custom configurations, use the CLI entry point:

```bash
python destfuzzinguzz.py \
    --openai_key sk-your-key \
    --target_model gpt-3.5-turbo \
    --model_path gpt-3.5-turbo \
    --judge_model_path ./path/to/roberta_model \
    --max_query 300 \
    --K 3 \
    --max_depth 10 \
    --lambda_uncertainty 0.3 \
    --alpha_gate 2.0
```

#### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | `gpt-3.5-turbo` | Mutator model name |
| `--target_model` | (required) | Target model to attack |
| `--judge_model_path` | (empty) | RoBERTa classifier path (uses GPT judge if empty) |
| `--max_query` | `300` | Query budget B per question |
| `--K` | `3` | Repeated evaluations for state estimation (Eq 4) |
| `--max_depth` | `10` | Maximum tree depth H |
| `--lambda_uncertainty` | `0.3` | Uncertainty penalty in transition reward (Eq 20) |
| `--alpha_gate` | `2.0` | Uncertainty gate attenuation strength (Eq 21) |
| `--c_v` | `1.0` | Exploration constant for node selection |
| `--c_a` | `1.0` | Exploration constant for operator selection |
| `--gamma_uncertainty` | `0.3` | Uncertainty penalty in node selection |
| `--beta_smooth` | `0.1` | Smoothing constant for transition kernel |
| `--n_min` | `3` | Minimum visits for dynamic intermediate expansion |
| `--delta_die` | `0.05` | Improvement margin for dynamic intermediate expansion |

---

## Defense-State Estimator

DeST-Fuzzing supports two defense-state estimators:

### 1. GPT-Based Judge (default)

Used when `judge-model-path` is empty. Classifies responses into 4 states using a prompted LLM call. Requires `API` key configured.

### 2. RoBERTa-Based Judge

A fine-tuned RoBERTa-large classifier trained on annotated target-model responses. Set `judge-model-path` to the trained model path.

**To train your own RoBERTa classifier:**

```bash
# Fine-tune on labeled data
python example/finetune_roberta.py \
    --task_name cola \
    --dataset_name csv \
    --train_file datasets/responses_labeled/train2.csv \
    --validation_file datasets/responses_labeled/evaluate2.csv \
    --max_seq_length 512 \
    --per_device_train_batch_size 80 \
    --learning_rate 1e-5 \
    --num_train_epochs 15 \
    --output_dir ./roberta_defense_model
```

Then set `judge-model-path: ./roberta_defense_model` in `Info-for-test.txt`.

---

## Mutation Operators

| Category | Operator | Description | Temperature |
|----------|----------|-------------|-------------|
| **Radical Exploration** | `Expand` | Add sentences at the beginning | 0.8 |
| | `CrossOver` | Crossover two templates | 0.8 |
| | `ChangeStyle` | Rewrite in different style | 0.8 |
| | `GenerateSimilar` | Generate template with similar style | 0.8 |
| **Conservative Infiltration** | `Rephrase` | Rephrase awkward sentences | 0.5 |
| | `Shorten` | Condense long sentences | 0.5 |
| | `Polish` | Improve word choice and flow | 0.5 |

---

## Citation

```bibtex
@inproceedings{destfuzzing2026,
  title     = {DeST-Fuzzing: Defense-State Transition Optimization for
               Stable Jailbreak Discovery in Large Language Models},
  author    = {Anonymous Authors},
  booktitle = {ACM CCS},
  year      = {2026}
}
```
