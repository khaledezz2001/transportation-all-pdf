import os
import time
import torch
import runpod
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

MODEL_PATH = "/models/hf/utrobinmv/t5"

tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if model is not None:
        return

    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        use_fast=False
    )
    log("Tokenizer loaded")

    log("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )
    model.eval()
    log(f"Model loaded on {model.device}")

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
    log("Handler started")
    load_model()

    pages = event["input"]["pages"]
    log(f"Pages: {len(pages)}")

    for i, page in enumerate(pages):
        log(f"Translating page {i+1}")
        page["text"] = translate_text(page["text"])

    log("Handler finished")
    return {"pages": pages}

runpod.serverless.start({"handler": handler})
