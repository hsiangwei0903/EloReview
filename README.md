# Modeling LLM Agent Reviewer Dynamics in Elo-Ranked Review System

## Overview

This repository contains the implementation for studying **Large Language Model (LLM) agent reviewer dynamics** in an Elo-ranked review system. We simulate a multi-agent system where LLM-powered reviewers with distinct personas engage in multiple rounds of paper review interactions, moderated by an Area Chair agent.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Conda (recommended for environment management)
- Google AI API key (for Gemini LLM access)

### Setup via Conda

1. **Create a new conda environment**:
   ```bash
   conda create -n eloreview python=3.10
   conda activate eloreview
   ```

2. **Install dependencies**:
   ```bash
   pip install google-generativeai numpy
   ```

3. **Set up your API key**:
   - Open `main.py` and add your API key:
     ```python
     GEMINI_API_KEY = "your-api-key-here"
     ```

---

## Project Structure

```
EloReview/
├── main.py                    # Main entry point for running simulations
├── papers.json                # Paper dataset for review simulation
├── prompt/                    # LLM prompts for agents
│   ├── ac.txt                # Area Chair agent prompt
│   ├── reviewer.txt           # Reviewer agent prompt
│   └── persona/              # Persona-specific behavior prompts
│       ├── bluffer.txt
│       ├── critic.txt
│       ├── expert.txt
│       ├── harmonizer.txt
│       ├── optimist.txt
│       └── skimmer.txt
└── role/                      # Core simulation modules
    ├── __init__.py
    ├── simulator.py           # SimulationManager - orchestrates multi-round reviews
    ├── reviewer.py            # LLMAgentReviewer - individual reviewer agents
    ├── area_chair.py          # AreaChairAgent - moderator and decision maker
    └── utils.py               # Utility functions
```

---

## Quick Start

### 1. Basic Simulation

Run the default simulation with sample papers:

```bash
python main.py
```

This will:
- Load papers from `papers.json`
- Initialize LLM agent reviewers with different personas
- Run multiple rounds of reviews
- Aggregate results and generate analysis

### 2. Customize Simulation Parameters

In `main.py`, you can configure:

```python
num_rounds = 5              # Number of review rounds
num_reviewers_per_paper = 3 # Reviewers assigned per paper
num_papers_per_round = 2    # Papers reviewed per round
```




## Citation

If you use this work in your research, please cite:

```bibtex
@misc{huang2026modelingllmagentreviewer,
      title={Modeling LLM Agent Reviewer Dynamics in Elo-Ranked Review System}, 
      author={Hsiang-Wei Huang and Junbin Lu and Kuang-Ming Chen and Jenq-Neng Hwang},
      year={2026},
      eprint={2601.08829},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.08829}, 
}
```
