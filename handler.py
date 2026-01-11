import os
import torch
import runpod
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "/models/hf/utrobinmv/t5"

tokenizer = None
model = None

def load_model():
    global tokenizer, model

    if model is not None:
        return

    print("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    use_fast=False
     )


    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )

    model.eval()
    print("Model loaded successfully.")

def translate_text(text: str) -> str:
    prompt = "translate Russian to English:\n" + text

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

def handler(event):
    load_model()

    pages = event["input"]["pages"]

    for page in pages:
        page["text"] = translate_text(page["text"])

    return {"pages": pages}

runpod.serverless.start({"handler": handler})
