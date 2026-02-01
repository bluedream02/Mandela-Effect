# [ICLR 2026] When Agents “Misremember” Collectively: Exploring the Mandela Effect in LLM-based Multi-Agent Systems

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This is the official implementation of **"When Agents 'Misremember' Collectively: Exploring the Mandela Effect in LLM-based Multi-Agent Systems"** (ICLR 2026).

## 📑 Table of Contents

- [Overview](#-overview)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
  - [Evaluation](#evaluation)
  - [Result Analysis](#result-analysis)
- [Acknowledgement](#-acknowledgement)
- [Citation](#-citation)

## 📌 Overview

The **Mandela Effect** is a phenomenon where groups collectively misremember verifiable facts, arising from social reinforcement and internalized misinformation. This repository introduces **ManBench**, a comprehensive benchmark designed to evaluate the Mandela effect in LLM-based multi-agent systems.

## 📦 Installation

```bash
git clone https://github.com/bluedream02/Mandela-Effect.git
cd Mandela-Effect
pip install -r requirements.txt
```

Set up your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"

# Optional: for local models via Ollama
# Make sure Ollama is installed and running locally
```

## 🚀 Quick Start

### Evaluation

**Basic evaluation on a single task:**
```bash
python eval.py --task_subset disambiguation_qa --max_samples 2 \
  --save_path output/disambiguation_qa_test \
  --model gpt-4o-mini --total_agents 5
```

**Evaluate multiple tasks:**
```bash
python eval.py --task_subset "disambiguation_qa,auto_categorization" \
  --max_samples 5 \
  --save_path output/multi_task_test \
  --model gpt-4o-mini --total_agents 5
```

**Key arguments for `eval.py`:**
- `--task_subset`: Task(s) to evaluate (comma-separated) (default: all)
- `--max_samples`: Maximum samples per task (default: all)
- `--save_path`: Output directory (default: `output_923_2`)
- `--model`: Model name (default: `gpt-4o-mini`)
- `--total_agents`: Number of agents (default: `5`)
- `--use_cache`: Use cached results to avoid redundant API calls during development. (default: `True`)
- `--data_folder`: Data folder path (default: `bbh_all_small`)

**Evaluate defense strategies:**
```bash
python eval_defense.py --task_subset disambiguation_qa --max_samples 2 \
  --save_path output/defense_test \
  --model gpt-4o-mini --total_agents 5
```

### Result Analysis

After running evaluation, analyze the results:

```bash
# Analyze with specific model and directory
python analyze.py --model gpt-4o-mini --input-dir output/disambiguation_qa_test

# Auto-detect output directories
python analyze.py --model gpt-4o-mini

# Results saved to output_evaluation/{model_name}.xlsx
```

**Key arguments for `analyze.py`:**
- `--model`: Model name to analyze
- `--input-dir`: Input directory (auto-detects `output/*` if not specified)
- `--output-dir`: Output directory (default: `output_evaluation`)

## 🎁 Acknowledgement

This work builds upon several excellent open-source projects and related works:
- **Do as We Do, Not as You Think: the Conformity of Large Language Models** (ICLR 2025) - [Paper](https://arxiv.org/abs/2501.13381) | [GitHub](https://github.com/Zhiyuan-Weng/BenchForm)
- **Exploring Prosocial Irrationality for LLM Agents: A Social Cognition View** (ICLR 2025) - [Paper](https://arxiv.org/abs/2405.14744) | [GitHub](https://github.com/XuanL17/CogMir)
- **Exploring Collaboration Mechanisms for LLM Agents: A Social Psychology View** (ACL 2024)- [Paper](https://arxiv.org/abs/2310.02124) | [GitHub](https://github.com/zjunlp/MachineSoM)
- **Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models** (TMLR) - [Paper](https://arxiv.org/abs/2206.04615) | [GitHub](https://github.com/google/BIG-bench)

We thank the authors for their valuable contributions to the community.
<!-- 
## 📖 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
``` -->