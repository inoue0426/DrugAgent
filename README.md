# DrugAgent: Multi-Agent LLM-Based Reasoning for Drug-Target Interaction Prediction

## Overview

DrugAgent is a multi-agent LLM system for drug-target interaction (DTI) prediction that combines AI model predictions, knowledge graph analysis, and literature search for comprehensive DTI prediction with interpretable reasoning.

## Key Features

- Multi-agent architecture providing predictions from multiple perspectives
- Integration of ML predictions, knowledge graphs, and literature search
- Interpretable reasoning using Chain-of-Thought and ReAct frameworks
- Detailed evidence-based predictions

## System Architecture
- Coordinator Agent: Task management and result integration
- AI Agent: ML-based DTI prediction using DeepPurpose
- Knowledge Graph Agent: Graph-based relationship analysis
- Search Agent: Literature-based evidence gathering
- Reasoning Agent: Final decision making with CoT/ReAct

## Environment

Supported environments:
- Mac with M1 chip (16GB unified memory)

## API Configuration
1. Create .env file in project root
2. Add API keys:
   OPENAI_API_KEY=your_key_here
   BING_API_KEY=your_key_here

This requires Open AI API and microsoft bing API.  
We are working on updating with Ollama for free use.

### Installation using Docker

```shell
git clone git@github.com:inoue0426/DrugAgent.git
cd DrugAgent
docker build -t tmp:latest .
docker run -it -p 9999:9999 -v $(pwd):/app tmp:latest
conda activate python_env
```

At the docker env.

```shell
cd agents/ & python integrate_agent.py
```

### Requirements
    - matplotlib
    - jupyter
    - numpy
    - pandas
    - lxml
    - indra
    - pyautogen==0.2.31
    - DeepPurpose==0.1.5
    - rdkit==2023.9.6
    - bs4==0.0.1
    - python-dotenv

