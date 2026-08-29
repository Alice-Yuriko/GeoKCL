# GeoKCL

## Overview

GeoKCL contains research code for geological hyper-relational extraction with BERT-based encoders. The executable workflow trains and evaluates a model that predicts entities, primary relations, and qualifier relations. The repository also includes three auxiliary scripts for geological-knowledge contrastive learning and knowledge-aligned encoder training.

## Repository Structure

```text
GeoKCL/
├── main.py
├── requirements.txt
├── .python-version
├── .gitignore
├── scripts/
│   ├── __init__.py
│   ├── train.py
│   ├── train_geology_contrastive.py
│   ├── train_geology_contrastive_scl.py
│   └── train_knowledge_alignment.py
└── src/
    ├── __init__.py
    └── model.py
```

- `main.py`: main entry point for hyper-relational extraction training and evaluation.
- `scripts/train.py`: data loading, training, checkpointing, decoding, and evaluation logic.
- `src/model.py`: model components, including PosiNet, qualifier-interactive attention, supervised contrastive loss, and the extraction model.
- `scripts/train_geology_contrastive.py`: triplet-based geological contrastive encoder training.
- `scripts/train_geology_contrastive_scl.py`: geological contrastive training with relation-supervised contrastive loss.
- `scripts/train_knowledge_alignment.py`: joint masked-language-modeling and geological-knowledge alignment training.

## Requirements

- Python 3.11
- NumPy
- PyTorch
- Transformers
- tqdm

Create and activate a Python 3.11 environment, then install the pinned dependencies:

```bash
pip install -r requirements.txt
```

The PyTorch version is pinned, but the appropriate CPU or CUDA build depends on the target machine. Install the hardware-specific PyTorch build first when the default package is not suitable, then install the remaining requirements.

## Data

Datasets are intentionally excluded from version control. The main workflow expects one dataset directory containing these files by default:

```text
data/<dataset>/
├── train.json
├── dev.json
├── test.json
└── label.json
```

Despite the `.json` extension, each train/dev/test file is read as JSON Lines: one JSON object per line. Each object must contain:

- `sentences`: tokenized sentences as a list of token lists;
- `ner`: entity spans for each sentence, represented as `[start, end, label]`;
- `relations`: relations represented as `[head_start, head_end, tail_start, tail_end, relation_label, qualifiers]`;
- each qualifier in `qualifiers`: `[start, end, qualifier_label]`.

`label.json` must define the label metadata used by the loader, including `id`, `entity`, `relation`, and `qualifier`.

The knowledge-learning scripts read JSONL records with `原始文本`, `正样本`, `负样本`, and `地质知识`. `train_knowledge_alignment.py` additionally uses `地质知识编号`; `train_geology_contrastive_scl.py` additionally uses `relation`.

Data locations can be changed with `--data_dir` and file-name arguments in the main workflow. The two non-CLI auxiliary scripts expose their paths in the `Config` or `CONFIG` block near the top of each file.

## Usage

### Hyper-relational extraction

Place a compatible pretrained BERT model in a local directory (not committed to Git), then run training and evaluation together:

```bash
python main.py \
  --dataset geoshr_hyperrelation \
  --data_dir data/geoshr \
  --model_name_or_path pretrained_models/bert-base \
  --output_dir outputs/geoshr
```

The current entry point enables training, development-set evaluation, and test-set evaluation by default. Checkpoints, tokenizers, logs, and result files are written below `--output_dir`.

To inspect all available configuration arguments:

```bash
python main.py --help
```

### Geological contrastive learning

The relation-supervised script provides command-line path and training controls:

```bash
python scripts/train_geology_contrastive_scl.py \
  --data_path data/geology_contrastive_with_relations.jsonl \
  --model_name pretrained_models/bert-base \
  --output_dir outputs/geology_contrastive \
  --epochs 10
```

The other two auxiliary workflows use configuration blocks in their source files:

```bash
python scripts/train_geology_contrastive.py
python scripts/train_knowledge_alignment.py
```

Set their data, pretrained-model, and output paths before running them.

## Configuration

The main workflow is configured through command-line arguments. Important options include `--data_dir`, `--model_name_or_path`, `--output_dir`, `--num_train_epochs`, `--train_batch_size`, `--eval_batch_size`, `--learning_rate`, `--max_seq_length`, `--scl_alpha`, `--gkc_alpha`, `--projection_dim`, and `--temperature`.

Scientific defaults are preserved from the local project. Change experiment parameters deliberately and record them with each run.
