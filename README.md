# SymptoMap - Symptom Checker & GP Referral System

> Clinical NLP system for symptom classification and GP referral workflow,
> built as an MSc Artificial Intelligence dissertation project.

## Overview

SymptoMap is a production-deployed clinical AI application that classifies patient symptom
descriptions and allows for GP referral using synthetic data. The system benchmarks three
neural architectures; BioClinicalBERT, BiLSTM, and CNN-BiLSTM; using a production-grade
evaluation framework mainly tracking accuracy and cross-distribution generalisation.

## Key Results

| Model | Internal Accuracy | Cross-LLM Generalisation |
|---|---|---|
| BioClinicalBERT | 99.93% | 73.45% |
| BiLSTM | 100% | 67.75% |
| CNN-BiLSTM | 97%% | 44.9% |

**BioClinicalBERT selected for production deployment** based on optimal accuracy and highest cross-distribution generalisation.

## Architecture

Research 
- Synthetic dataset (10,000+ samples) created with Claude and GPT
- External held-out dataset (2,000+ samples) created with Gemini
- Preprocessed and trained 
- Evaluated internally and externally with held-out dataset (Cross-LLM evaluation)

Production
- BioClinicalBERT deployed as backend for Streamlit App -> disease classification from free text symptom description input -> GP referral workflow -> Email sent to GP


## Tech Stack

- **Models:** PyTorch, Hugging Face Transformers, BioClinicalBERT
- **NLP:** spaCy, scikit-learn
- **Data Generation:** LLM prompt engineering (synthetic data pipeline)
- **Deployment:** Streamlit
- **Language:** Python

## Features

- Three-way architecture comparison with systematic robustness testing
- LLM-based synthetic data generation with cross-model generalisation evaluation
- Production evaluation framework: accuracy, F1 score, cross LLM generalisation, synthetic data usability
- End-to-end GP referral workflow with email notification
- Responsible AI principles embedded throughout evaluation methodology


## Setup

```bash
git clone https://github.com/mithurssan/SymptoMap.git
cd SymptoMap
pip install -r requirements.txt
```
run preprocessing.py first and then train all models as desired (**BioClinicalBERT must be run** for the app to work)

Run the app:
```bash
streamlit run app.py
```

## Research Context

Built as an MSc Artificial Intelligence dissertation at Brunel University (2025–2026).
Focus: synthetic data viability and evaluation methodology for clinical NLP systems, incorporating
responsible AI principles and cross-distribution generalisation testing to simulate real world robustness.

> **Note:** My trained Models (`.pt`, `.json`), weight embeddings (`.npy`) and encoders (`.pickle`) are not included, you will have to retrain on your own.
