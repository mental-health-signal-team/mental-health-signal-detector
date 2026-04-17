# Technology Watch — Mental Health Signal Detector
## Competency C6 — Veille Technologique en Intelligence Artificielle

**Project**: RNCP37827BC02 — Bloc 2 — Développeur en Intelligence Artificielle
**Period covered**: September 2024 – April 2026
**Author**: Stanislav Grinchenko

---

## 1. Overview

This document records the technology watch activities conducted during the development of the Mental Health Signal Detector, an NLP-based system for detecting psychological distress signals in text. The watch covers four themes: NLP/transformer architectures for mental health, datasets and ethics, deployment tooling, and MLOps practices.

---

## 2. Domain: Mental Health NLP

### 2.1 Literature and Models Surveyed

| Date | Source | Finding |
|---|---|---|
| Sep 2024 | Ji et al., 2022 — "MentalBERT" (arXiv:2110.10610) | Domain-pretrained BERT on mental health corpora (Reddit/Twitter). Outperforms BERT on depression/suicidality tasks. |
| Sep 2024 | Pérez et al., 2023 — "Mental-RoBERTa" (HuggingFace: mental/mental-roberta-base) | RoBERTa pretrained on 13.8M mental health posts from Reddit. Strongest zero-shot performance among open-source models. Selected as primary deployment model. |
| Oct 2024 | Devlin et al., 2018 — "BERT: Pre-training of Deep Bidirectional Transformers" | Foundational architecture. Evaluated DistilBERT (Sanh et al., 2019) as a faster, lighter variant for production use. |
| Oct 2024 | HuggingFace Model Hub — survey of `mental-health` tag | Found 47 fine-tuned models. Key comparison: `mental/mental-roberta-base` (13.8M post pretraining) vs `distilbert-base-uncased-finetuned-sst-2-english` (generic sentiment). |
| Nov 2024 | Tadesse et al., 2019 — "Detection of Depression-Related Posts in Reddit Social Media Forum" | Classic ML baseline using TF-IDF + LR on Reddit data. Validated our baseline approach. |
| Jan 2025 | Coppersmith et al. — "Natural Language Processing of Social Media as Screening for Suicide Risk" | Ethics considerations: false positives in mental health screening have real-world harm. Informed our risk threshold design (`_RISK_THRESHOLDS = (0.33, 0.66)`). |

### 2.2 Model Selection Rationale

Four candidate architectures were evaluated:

| Model | Type | F1 Score | Decision |
|---|---|---|---|
| Logistic Regression + TF-IDF | Classical ML | 0.93 | Kept as fast baseline (model_type="lr") |
| XGBoost + TF-IDF | Gradient Boosting | 0.93 | Kept as alternative baseline |
| DistilBERT (fine-tuned) | Transformer | 0.96 | Kept for comparison |
| **MentalRoBERTa (fine-tuned)** | **Domain Transformer** | **0.96** | **Selected as primary deployment model** |

MentalRoBERTa was selected over DistilBERT despite equal F1 because: (1) higher precision on the positive (distress) class (0.96 vs 0.94), reducing false positives — critical for mental health safety; (2) domain pretraining on mental health text reduces distribution shift.

---

## 3. Dataset and Bias Watch

### 3.1 Spurious Correlation — 3 Iterations

A key finding from the technology watch was the identification of spurious correlations in NLP training data, documented in Ribeiro et al., 2020 ("Beyond Accuracy: Behavioral Testing of NLP Models with CheckList") and Geirhos et al., 2020 ("Shortcut Learning in Deep Neural Networks").

This led to three dataset iterations:

**Iteration 1 — r/teenagers Bias**
- Dataset: 7 subreddits including r/teenagers as the "positive" (no distress) class
- Observed: Model learned "teenager slang" as a proxy for mental wellness
- Detection: Manual error analysis showed slang-heavy posts from adults misclassified as no-distress
- Fix: Removed r/teenagers from the positive class

**Iteration 2 — DAIR-AI/emotion Bias**
- Dataset: Mixed Reddit posts + DAIR-AI/emotion (Ekman emotion taxonomy dataset)
- Observed: Model learned "joy"/"surprise" tokens as direct distress-negative signals
- Detection: Gradient attribution revealed `joy`, `happy`, `excited` as top-weighted negative tokens — domain mismatch artifact
- Fix: Removed DAIR-AI/emotion entirely; used Reddit-only sources

**Iteration 3 — Reddit-only (Final)**
- Dataset: `balanced_dataset_30k.csv` — 30,000 posts, 1:1 class balance
- Positive class: 15 subreddits (r/happy, r/CasualConversation, r/aww, r/gaming, r/fitness, etc.)
- Negative class: r/depression, r/SuicideWatch, r/mentalhealth, r/anxiety
- Result: +13 points F1 improvement vs Iteration 1. No detected shortcut features.

### 3.2 Ethical Watch — Privacy and Anonymization

**Source**: EU AI Act (2024), Article 10 — Data Governance; GDPR Article 4(1)

All API prediction logs are anonymized at write time using SHA-256 hashing of input text (`src/api/database.py:hash_text()`). No raw text is stored in the production database. This design was informed by:
- CNIL guidelines on AI system logs (France, 2023)
- Privacy-by-design principles (ISO/IEC 27701:2019)

---

## 4. Deployment Tooling Watch

### 4.1 Containerization

**Watched**: Docker best practices for ML inference services (2024)
- CPU-only PyTorch (`torch==2.2.0+cpu`) — avoids GPU cost for inference-only deployment
- Multi-stage build considered but rejected (model size makes layer caching more valuable)
- GCP Cloud Run selected over Cloud Functions: supports persistent in-memory model caching via async lifespan pattern (FastAPI 0.103+)

### 4.2 MLOps Frameworks Surveyed

| Tool | Evaluated | Decision |
|---|---|---|
| MLflow | Experiment tracking, model registry | Considered; not adopted due to added infra complexity for a solo project |
| Weights & Biases | Training curves, hyperparameter tracking | Used during training experiments (notebooks) |
| BentoML | Model serving | Evaluated; FastAPI chosen for flexibility and certification alignment |
| GCP Vertex AI | Managed ML platform | Considered for model registry; Cloud Run preferred for cost |

### 4.3 CI/CD Watch

**GitHub Actions** chosen over GitLab CI / CircleCI:
- Native GitHub integration (no token management)
- Free tier sufficient for this project's matrix (lint + pre-commit + pytest)
- 2024 GitHub Actions OIDC support enables keyless GCP authentication

**GCP Cloud Build** chosen over GitHub Actions for CD:
- Runs inside GCP network (avoids SA key management)
- Artifact Registry integration (Docker image provenance)
- Cloud Run revision management enables traffic-split rollbacks

---

## 5. NLP Libraries and Frameworks Watch

### 5.1 HuggingFace Transformers

| Version | Watch Finding |
|---|---|
| 4.x → 5.x migration | `from_pretrained()` replaced pickle serialization. Models saved with `save_pretrained()` (HF native format) are forward-compatible. Required migration script: `scripts/convert_mental_roberta.py` |
| `AutoModelForSequenceClassification` | Standard API for fine-tuning classification heads. Supports `output_attentions` and gradient-based attribution for explainability (C10). |

### 5.2 Explainability Techniques

**Source**: Simonyan et al., 2014 — "Deep Inside Convolutional Networks: Visualizing Image Classification Models and Saliency Maps"; Sundararajan et al., 2017 — "Axiomatic Attribution for Deep Networks (Integrated Gradients)"

Selected: **Gradient × Input attribution** — implemented in `src/api/services.py:_transformer_word_importance()`. Chosen over LIME/SHAP because:
- No sampling overhead (single forward+backward pass)
- Token-aligned output maps directly to input words
- Sufficient precision for the word-highlighting use case

---

## 6. Sources Monitored Regularly

| Source | Frequency | What Tracked |
|---|---|---|
| arXiv cs.CL (NLP section) | Weekly | New mental health NLP papers, dataset releases |
| HuggingFace Model Hub | Monthly | New mental health fine-tuned models |
| Papers With Code | Monthly | State-of-the-art benchmarks on depression detection tasks |
| FastAPI GitHub releases | Per release | Breaking changes, new async patterns |
| HuggingFace Transformers changelog | Per release | API changes affecting model loading |
| GCP Cloud Run release notes | Monthly | New features (min-instances, HEALTHCHECK support) |
| CNIL (France) AI guidelines | Quarterly | Regulatory updates on AI/health data |

---

## 7. Key Decisions Driven by the Watch

1. **MentalRoBERTa over DistilBERT**: domain pretraining on mental health text reduces false positive rate on ambiguous posts — critical safety property.
2. **Reddit-only dataset**: shortcut learning research (Geirhos 2020) motivated eliminating cross-domain data sources.
3. **Gradient attribution over LIME**: efficiency and token alignment make it better suited for real-time API responses.
4. **SHA-256 anonymization**: GDPR + CNIL guidelines drove privacy-by-design in the logging architecture.
5. **Three-tier risk levels**: inspired by clinical triage literature — binary classification output is insufficient for a health-adjacent application.
