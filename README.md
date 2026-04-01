# Value-Bench: A Benchmark for Evaluating Value Alignment in Multimodal LLMs

Value-Bench is a benchmark for evaluating the ability of Multimodal Large Language Models (MLLMs) to detect harmful, unethical, and offensive content in internet memes. It covers 9 tasks across diverse unsafe content categories and measures both classification accuracy and model evasion behavior.

## Tasks

| Task | Dataset | Description |
|------|---------|-------------|
| `unethical` | ELEMENT | Unethical meme detection |
| `harmful` | Harm-C | Harmful content detection |
| `harmful (CN)` | CHMEMES | Chinese harmful content detection |
| `hateful` | HMC | Hateful meme classification |
| `offensive` | MultiOFF | Offensive content detection |
| `misogyny` | Misogyny | Misogynistic content detection |
| `shaming` | Misogyny | Gender-based shaming (subtask) |
| `stereotype` | Misogyny | Gender stereotyping (subtask) |
| `objectification` | Misogyny | Sexual/physical objectification (subtask) |
| `violence` | Misogyny | Violence against women (subtask) |

## Supported Models

| Model | Key |
|-------|-----|
| Qwen-VL | `Qwen-VL` |
| LLaVA | `LLaVA` |
| BLIP-2 | `blip2` |
| MoE-LLaVA | `MoELLaVA` |
| CogVLM | `CogVLM` |
| MiniCPM-V | `MiniCPM-V` |
| InstructBLIP | `instructblip` |
| mPLUG-Owl | `mplugowl` |
| mPLUG-Owl2 | `mPLUGOwl2` |
| InternLM-XComposer | `xcomposer` |
| InternLM-XComposer2 | `XComposer2` |
| VisCPM | `VisCPM` |
| VisualGLM | `visualGLM` |
| MMICL | `MMICL` |
| IDEFICS | `IDEFICS` |

## Data Format

Each dataset should be placed under `Alig/<DATASET>/` with the following structure:

```
Alig/
└── <DATASET>/
    ├── img/          # Image files (filename without extension is used as ID)
    └── test.jsonl    # Ground-truth labels
```

**Label format for binary tasks:**
```json
{"id": "123", "label": 1}
```

**Label format for misogyny (multi-label):**
```json
{"id": "123", "label": [1, 0, 1, 0, 0]}
```

## Evaluation

### Metrics

- **ACC**: Accuracy over all predictions across 3 prompt variants.
- **Avoidance Rate**: Percentage of "uncertain" responses — a high rate may indicate refusal or evasion behavior.

### Prompt Variants

Each task uses 3 prompt variants with different answer-option orderings (A/B/C) to control for position bias and verify consistent model reasoning. Final metrics aggregate across all variants.

## Usage

### Run evaluation

```bash
python run.py --model <MODEL_KEY> --task <TASK_NAME>
```

**Example:**
```bash
python run.py --model Qwen-VL --task unethical
```

### View available prompts

```python
from get_Prompts import get
prompts = get("hateful")
print(prompts)
```

### Run pre-check / scoring only

```python
from pre_check import check
# pres: list of {"id": ..., "answers": [...]} dicts
# arg: namespace with task and data path info
check(pres, arg)
```

## Project Structure

```
Value_Bench/
├── run.py              # Main entry point
├── get_Prompts.py      # Prompt template generation
├── pre_check.py        # Answer parsing and metric computation
├── MLLM/               # Model wrappers (one file per model)
│   ├── QwenVL.py
│   ├── LLaVA.py
│   ├── blip2.py
│   └── ...
└── Alig/               # Benchmark datasets
    ├── ELEMENT/
    ├── CHMEMES/
    ├── Harm-C/
    ├── HMC/
    ├── MultiOFF/
    └── Misogyny/
```

## Requirements

- Python 3.8+
- PyTorch (CUDA recommended)
- `transformers`
- `Pillow`
- Model-specific dependencies (see individual files in `MLLM/`)

For IDEFICS, the `gradio_client` package is required for remote inference via the HuggingFace Space API.

## Citation

If you use Value-Bench in your research, please cite our paper:

```bibtex
@article{valuebench2025,
  title     = {Value-Bench: Evaluating Value Alignment of Multimodal Large Language Models on Meme Content},
  year      = {2025}
}
```
