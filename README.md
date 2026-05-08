#  Building Intelligent Systems (AI Production Applications)

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red?style=flat-square&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/Transformers-HuggingFace-yellow?style=flat-square&logo=huggingface)
![BERT](https://img.shields.io/badge/Model-BERT-green?style=flat-square)
![AGNews](https://img.shields.io/badge/Dataset-AG%20News-purple?style=flat-square)
![Gradio](https://img.shields.io/badge/Deployment-Gradio-orange?style=flat-square)
![Status](https://img.shields.io/badge/Project-Active-success?style=flat-square)

---

##  Fine-Tuning BERT for News Topic Classification  
### Transformer-Based News Intelligence System (Research-Grade NLP System)

---

##  Live Research Outputs & Deployed Systems

This project represents a complete **end-to-end applied AI research system**, covering dataset preparation, transformer fine-tuning, evaluation, and production deployment.

### 🌐 Live AI Application (Real-Time Inference System)
https://huggingface.co/spaces/Mudassir-08/bert-news-intelligence-live  

This interactive Gradio-based AI system demonstrates **real-time news classification**, where users can input any news text and receive instant predictions using the fine-tuned transformer model.

---

### 🤖 Fine-Tuned Transformer Model (Hugging Face Hub)
https://huggingface.co/Mudassir-08/bert-news-intelligence-v1-release  

This is the official **research-trained BERT checkpoint**, fine-tuned on the AG News dataset using transfer learning. It encapsulates learned semantic representations optimized for news domain classification.

---

##  Research Overview

This work focuses on applying **Transformer-based deep learning architectures** to solve structured **text classification problems in news intelligence systems**.

Traditional NLP pipelines rely on sparse representations (TF-IDF, n-grams), which fail to capture contextual meaning. This project instead uses **BERT (Bidirectional Encoder Representations from Transformers)** to model deep bidirectional language context.

The objective is to demonstrate:
- Effectiveness of transfer learning in NLP classification tasks  
- Context-aware semantic understanding using transformers  
- End-to-end deployment of research-grade models into production systems  

---

##  Model Architecture (Theoretical Pipeline)

Input Text  
↓  
WordPiece Tokenizer (Subword Segmentation)  
↓  
Token IDs + Attention Mask Generation  
↓  
BERT Encoder (12 Transformer Layers, Bidirectional Self-Attention)  
↓  
Contextual Embedding Representation  
↓  
Fine-Tuned Classification Head (Fully Connected Layer)  
↓  
Softmax Probability Distribution  
↓  
Final Predicted News Category  

---

##  Dataset

The model is fine-tuned on the **AG News Dataset**, a widely recognized benchmark in NLP research:

- 120,000 training samples  
- 7,600 test samples  
- 4 balanced classification labels  

### Classes:
-  World  
-  Sports  
-  Business  
-  Science & Technology  

---

##  Training Configuration

- Base Model: bert-base-uncased  
- Task: Sequence Classification  
- Loss Function: Cross-Entropy Loss  
- Optimizer: AdamW  
- Learning Rate: 2e-5  
- Batch Size: 16  
- Epochs: 3  
- Max Sequence Length: 128  
- Precision: FP16 (Mixed Precision Training)  

---

##  Evaluation Results

- Accuracy: **94.8%**  
- F1 Score: **94.8%**  
- Loss: **0.19**

---

##  Research Insights

- Transformer models outperform traditional ML methods in semantic classification tasks  
- Bidirectional attention significantly improves contextual understanding  
- Sports category shows high separability due to strong lexical patterns  
- Sci/Tech and Business show moderate semantic overlap  
- Fine-tuning dramatically improves domain adaptation performance  

---

#  HOW TO LOAD THIS MODEL (SIMPLE USAGE)

This section explains how to load and use the fine-tuned model in any Python environment such as Colab, local machine, or production APIs.

---

##  Step 1 — Install Dependencies

```python
pip install transformers torch
```

##  Step 2 — Load Model from Hugging Face

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "Mudassir-08/bert-news-intelligence-v1-release"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

model.eval()
```

---

#  Deployment Summary

This project represents a full **end-to-end Transformer-based NLP system**, combining research, fine-tuning, and real-time deployment using BERT for news topic classification.

---

## 🌐 Live AI System

https://huggingface.co/spaces/Mudassir-08/bert-news-intelligence-live  

A fully deployed **Gradio-based AI application** that performs real-time classification of news text into predefined categories using a fine-tuned BERT model.

---

## 🤖 Model Repository

https://huggingface.co/Mudassir-08/bert-news-intelligence-v1-release  

This is the official **fine-tuned BERT checkpoint**, trained on the AG News dataset using transfer learning with transformer-based architecture.

---

##  Author Identity

**mudassir-08**

Malik Muhammad Mudassir Iqbal

### Roles:
- NLP & Transformer Engineer  
- Applied AI Researcher  
- Deep Learning Practitioner and Researcher  
- GenAI and Agentic AI Developer  

---

## License

This project is intended for **research, academic, and educational purposes in NLP and transformer-based AI systems**.

---

