# CodeQual: A Dataset and Benchmark for Code Quality Assessment

## Overview

CodeQual is a dataset of **5,819 Python code samples** from five diverse sources, annotated with quality scores across five dimensions on a continuous 1--5 scale. It includes **655 human-annotated test samples** for evaluation and synthetic LLM annotations for training. CodeQual enables research on automated code quality assessment, a task that goes beyond functional correctness to evaluate subjective properties like readability, efficiency, and adherence to language conventions.

This repository contains the complete toolkit for dataset construction, LLM benchmarking, human annotation collection, and model training.

**Paper**: Accepted at ICNLP 2026.
**Dataset**: [https://doi.org/10.5281/zenodo.17765078](https://doi.org/10.5281/zenodo.17765078) (CC BY license)

## Quality Dimensions

Each code sample is scored across five dimensions:

| Dimension | Description |
|-----------|-------------|
| **Functionality** | Does the code work correctly and handle edge cases? |
| **Readability** | How clear and maintainable is the code? |
| **Idiomatic** | Does it follow Python best practices and conventions? |
| **Error Handling** | How well does it manage errors and invalid inputs? |
| **Efficiency** | Is it scalable with good time/space complexity? |

## Key Results

- **17 LLMs evaluated** on the human-annotated test set; **o3-mini** achieved the highest average correlation with human judgments (ρ = 0.31)
- **CodeQualBERT** (fine-tuned CodeBERT, 124M parameters) exceeds inter-human agreement on all five dimensions (16--100% improvement)
- CodeQualBERT exceeds the best LLM on error handling (+36%) and efficiency (+18%), demonstrating effective knowledge distillation

## Data Sources

| Source | Samples | Domain |
|--------|---------|--------|
| CodeEval | 602 | Pedagogical problems (hand-crafted) |
| CodeNet | 2,250 | Competitive programming |
| CodeSearchNet | 1,669 | Real-world GitHub functions |
| HumanEval-X | 820 | Multi-language benchmarks |
| MBPP | 478 | General Python programming |

## Dataset Format

Each sample is stored in JSONL format:

```json
{
    "problem_id": "codeeval_001",
    "submission_id": "sub_123",
    "problem": "Write a function to reverse a string...",
    "submission": "def reverse_string(s):\n    return s[::-1]",
    "quality_scores": {
        "functionality": 4.2,
        "readability": 4.8,
        "idiomatic": 5.0,
        "error_handling": 2.1,
        "efficiency": 4.5
    },
    "source": "codeeval",
    "metadata": {
        "complexity": "easy",
        "language": "python"
    }
}
```

**Splits**: Train (80%), Validation (10%), Test (10%). The test set contains human annotations; train and validation sets contain synthetic LLM annotations from o3-mini.

## Repository Structure

```
codequal-dataset/
├── projects/
│   ├── dataset/              # Data integration and management CLI
│   ├── test-annotations/     # Qualtrics survey generation and annotation processing
│   ├── llm_benchmarks/       # LLM evaluation pipeline (17 models, multi-provider)
│   ├── model-training/       # CodeQualBERT and MLP training
│   ├── dataset-viewer/       # Web-based dataset visualization
│   └── paper/                # ICNLP 2026 paper
└── src/
    └── paper_submission/     # Tables and analysis scripts for thesis/paper
```

## Getting Started

### Installation

```bash
git clone https://github.com/dannybrahman/codequal-dataset.git
cd codequal-dataset
```

Each sub-project has its own dependencies:

```bash
# Dataset integration
cd projects/dataset && pip install -r requirements.txt

# LLM benchmarking
cd projects/llm_benchmarks && pip install -r requirements.txt

# Model training
cd projects/model-training && pip install -r requirements.txt
```

### Dataset Integration

```bash
cd projects/dataset

# Integrate a data source
python main.py integrate --source codeeval

# Add human annotation scores
python main.py add-human-scores --source codenet --method mean

# Add LLM quality scores
python main.py add-llm-scores --source codenet --session <session_id>
```

### LLM Benchmarking

```bash
cd projects/llm_benchmarks

# Run quality assessment collection
python scripts/run_collection.py --models gpt-4.1 claude-3-opus --categories codenet

# Resume an interrupted session
python scripts/resume_collection.py --session <session_id>
```

### Model Training

```bash
cd projects/model-training

# Train CodeQualBERT
python main.py train --model codebert --sources codenet mbpp codesearchnet humaneval-x codeeval

# Evaluate against human test set
python main.py evaluate --model codebert

# Compare models
python main.py compare
```

## Reproducing Results

### CodeQualBERT Training

- **Architecture**: CodeBERT [CLS] embedding → 256 → 128 → 5 outputs
- **Optimizer**: AdamW, learning rate 10⁻⁴, batch size 16
- **Training data**: Synthetic scores from o3-mini
- **Evaluation**: Spearman correlation with human ground truth on test set

### MLP Baseline

- **Architecture**: 21 hand-crafted features → 128 → 64 → 32 → 5 outputs with dropout
- **Features**: Line count, whitespace ratio, function/class counts, Python keyword frequencies

## Citation

If you use CodeQual in your research, please cite:

```bibtex
@inproceedings{brahman2025codequal,
  title={CodeQual: Learning Code Quality Assessment from Synthetic LLM Annotations},
  author={Brahman, Danny and Mahoor, Mohammad},
  booktitle={2026 8th International Conference on Natural Language Processing (ICNLP)},
  year={2025}
}
```

## License

This project is licensed under the MIT License. The dataset is available under CC BY license.

## Contact

Danny Brahman — [danny.brahman@du.edu](mailto:danny.brahman@du.edu)

University of Denver, Computer Vision and Social Robotics Laboratory
