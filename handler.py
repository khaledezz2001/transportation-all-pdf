import time
import torch
import runpod

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    MarianTokenizer,
    MarianMTModel,
)

# -------------------------------------------------
# Logging helper
# -------------------------------------------------
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# -------------------------------------------------
# Model paths
# -------------------------------------------------
SUMMARY_MODEL_PATH = "/models/hf/t5-summary"
TRANSLATE_MODEL_PATH = "/models/hf/marian-ru-en"

summary_tokenizer = None
summary_model = None
translate_tokenizer = None
translate_model = None

# -------------------------------------------------
# Load summary model (T5)
# -------------------------------------------------
def load_summary_model():
    global summary_tokenizer, summary_model
    if summary_model is not None:
        return

    log("Loading SUMMARY model (T5)")
    summary_tokenizer = AutoTokenizer.from_pretrained(
        SUMMARY_MODEL_PATH,
        local_files_only=True,
        use_fast=False
    )

    summary_model = AutoModelForSeq2SeqLM.from_pretrained(
        SUMMARY_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )
    summary_model.eval()
    log("SUMMARY model loaded")

# -------------------------------------------------
# Load translation model (Marian)
# -------------------------------------------------
def load_translate_model():
    global translate_tokenizer, translate_model
    if translate_model is not None:
        return

    log("Loading TRANSLATION model (Marian)")
    translate_tokenizer = MarianTokenizer.from_pretrained(
        TRANSLATE_MODEL_PATH,
        local_files_only=True
    )

    translate_model = MarianMTModel.from_pretrained(
        TRANSLATE_MODEL_PATH,
        torch_dtype=torch.float16,
        local_files_only=True
    ).to("cuda")

    translate_model.eval()
    log("TRANSLATION model loaded on cuda")

# -------------------------------------------------
# Summarize all pages together
# -------------------------------------------------
def summarize_all_pages(pages):
    full_text = "\n\n".join(p["text"] for p in pages)
    prompt = "summarize:\n" + full_text

    inputs = summary_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(summary_model.device)

    with torch.no_grad():
        output = summary_model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )

    return summary_tokenizer.decode(output[0], skip_special_tokens=True)

# -------------------------------------------------
# Translate single page (format preserved)
# -------------------------------------------------
def translate_text(text):
    inputs = translate_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(translate_model.device)

    with torch.no_grad():
        output = translate_model.generate(**inputs)

    return translate_tokenizer.decode(output[0], skip_special_tokens=True)

# -------------------------------------------------
# RunPod handler
# -------------------------------------------------
def handler(event):
    log("Handler started")

    pages = event["input"]["pages"]

    load_summary_model()
    load_translate_model()

    log("Creating summary")
    summary = summarize_all_pages(pages)

    log("Translating pages")
    for p in pages:
        p["text"] = translate_text(p["text"])

    log("Handler finished")

    return {
        "summary": summary,
        "pages": pages
    }

runpod.serverless.start({"handler": handler})
