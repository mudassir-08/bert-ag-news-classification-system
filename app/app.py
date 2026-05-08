import os

# 
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import gradio as gr
from transformers import pipeline

MODEL_NAME = "Mudassir-08/bert-news-intelligence-v1-release"

classifier = pipeline("text-classification", model=MODEL_NAME)

def predict(text):
    if not text.strip():
        return "Empty input", 0.0

    result = classifier(text)[0]
    return result["label"], float(result["score"])

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=2),
    outputs=["text", "number"],
    title="BERT News Classifier"
)

demo.launch()
