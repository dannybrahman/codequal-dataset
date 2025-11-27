# Model Training for Code Quality Assessment

Train smaller models on LLM-generated synthetic quality scores to demonstrate the utility of the CodeQual dataset.

## Overview

This project trains lightweight models to predict 5-dimensional code quality scores using synthetic annotations from LLMs. The goal is to show that:

1. **Synthetic data is useful** for training quality assessment models
2. **Smaller models work** and can approximate expensive LLM evaluations
3. **Cost-effective deployment** is possible with trained models

## Quality Dimensions

Models predict 5 continuous scores (1-5 scale):
- **Functionality** - Does the code work correctly?
- **Readability** - How clear and maintainable?
- **Idiomatic** - Follows language best practices?
- **Error Handling** - Manages edge cases well?
- **Efficiency** - Scalable and performant?

## Installation

```bash
cd projects/model-training
pip install -r requirements.txt
```

## Quick Start

### Train CodeBERT Model

```bash
# Train on codenet and codeeval sources
python main.py train --model codebert --sources codenet codeeval --epochs 10

# Train with custom settings
python main.py train --model codebert \
    --sources codenet \
    --batch-size 32 \
    --epochs 20 \
    --lr 2e-5 \
    --device mps  # Use M3 Mac GPU
```

### Train Simple MLP Baseline

```bash
# Fast baseline model
python main.py train --model mlp --sources codenet codeeval
```

### Evaluate Trained Model

```bash
# Evaluate on human test set (use session name from training output)
python main.py evaluate --session run_session_XXXXXXXXXX
```

## Model Architectures

### CodeBERT Regressor
- Fine-tunes `microsoft/codebert-base`
- Multi-output regression head (5 dimensions)
- Best accuracy but requires GPU
- ~125M parameters

### Simple MLP
- Lightweight feedforward network
- Uses statistical features (LOC, complexity, etc.)
- Fast baseline for comparison
- ~50K parameters

## Data Flow

```
1. Load: ../dataset/generated/integrated/{source}/train.jsonl (LLM scores)
2. Extract: CodeBERT embeddings or simple features
3. Train: Predict 5D quality scores
4. Evaluate: Test on ../dataset/generated/integrated/{source}/test.jsonl (human median scores)
5. Compare: Trained model vs original LLM agreement
```

## Device Support

Automatically detects and uses:
- **CUDA** (NVIDIA GPUs)
- **MPS** (Apple Silicon M1/M2/M3) ✅
- **CPU** (fallback)

## Output Structure

Each training run creates a session directory:

```
generated/
└── run_session_<timestamp>/
    ├── models/
    │   ├── best_model.pt           # Best checkpoint (lowest validation loss)
    │   ├── final_model.pt          # Final checkpoint after all epochs
    │   ├── model.json              # Model info, config, and training params
    │   └── training_history.json   # Epoch-by-epoch metrics
    ├── results/                    # Evaluation results (populated by evaluate command)
    └── logs/                       # Reserved for future use
```

## Expected Results

- **Target**: Trained model achieves >80% of LLM's agreement with humans
- **Speedup**: 10-100x faster inference than LLM API
- **Baseline**: Best LLM's Spearman correlation with human annotations
- **Goal**: Trained model achieves >80% of best LLM's agreement with humans

## Evaluation Metrics

Same as dataset project's LLM-human agreement analysis:
- **Spearman correlation** - Rank-order agreement
- **Pearson correlation** - Linear relationship
- **MAE** - Mean absolute error
- **RMSE** - Root mean squared error

## CLI Commands

```bash
# Training (creates new session automatically)
python main.py train --model {codebert|mlp} [options]

# Evaluation (requires session name from training)
python main.py evaluate --session run_session_XXXXXXXXXX [options]

# Help
python main.py --help
python main.py train --help
python main.py evaluate --help
```

## Advanced Options

```bash
# Freeze encoder (faster training, may reduce accuracy)
python main.py train --model codebert --freeze-encoder

# Custom learning rate and patience
python main.py train --model codebert --lr 5e-5 --patience 10

# Specify device
python main.py train --model codebert --device mps  # M3 Mac
python main.py train --model codebert --device cuda # NVIDIA GPU
```

## Integration with Other Projects

- **Dataset Project**: Reads from `../dataset/generated/integrated/`
- **LLM Benchmarks**: Trains on LLM-generated synthetic scores
- **Agreement Analysis**: Uses same evaluation metrics

## Project Structure

See [CLAUDE.md](CLAUDE.md) for detailed architecture and development guidelines.
