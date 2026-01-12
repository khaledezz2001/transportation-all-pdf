import time
import re
import torch
import runpod

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    MarianTokenizer,
    MarianMTModel,
)

# =====================================================
# Logging helper
# =====================================================
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# =====================================================
# Model paths
# =====================================================
SUMMARY_MODEL_PATH = "/models/hf/t5-summary"
TRANSLATE_MODEL_PATH = "/models/hf/marian-ru-en"

summary_tokenizer = None
summary_model = None
translate_tokenizer = None
translate_model = None

# =====================================================
# Load SUMMARY model (T5)
# =====================================================
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

# =====================================================
# Load TRANSLATION model (Marian)
# =====================================================
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

# =====================================================
# Detect pure separator lines (---, ___)
# =====================================================
def is_layout_line(line: str) -> bool:
    return bool(re.match(r"^[\-\._\s]{5,}$", line))

# =====================================================
# Translate text (TABLES INCLUDED, STRUCTURE SAFE)
# =====================================================
def translate_text(text: str) -> str:
    lines = text.split("\n")
    out_lines = []

    for line in lines:
        # Empty line
        if not line.strip():
            out_lines.append(line)
            continue

        # FULL separator row (even with pipes) → keep EXACT
        if re.match(r"^\|\s*[-\s_\.]+\|\s*[-\s_\.]+\|\s*$", line):
            out_lines.append(line)
            continue

        # Pure separator (no pipes)
        if is_layout_line(line):
            out_lines.append(line)
            continue

        # -------- TABLE ROW --------
        if "|" in line:
            cells = line.split("|")
            new_cells = []

            for cell in cells:
                cell_text = cell.strip()

                # Empty or separator-only cell → keep as-is
                if not cell_text or re.match(r"^[-\s_\.]+$", cell_text):
                    new_cells.append(cell)
                    continue

                inputs = translate_tokenizer(
                    cell_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128
                ).to(translate_model.device)

                with torch.no_grad():
                    output = translate_model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False
                    )

                translated = translate_tokenizer.decode(
                    output[0], skip_special_tokens=True
                )

                new_cells.append(f" {translated} ")

            out_lines.append("|".join(new_cells))
            continue

        # -------- NORMAL TEXT LINE --------
        inputs = translate_tokenizer(
            line,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(translate_model.device)

        with torch.no_grad():
            output = translate_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )

        out_lines.append(
            translate_tokenizer.decode(output[0], skip_special_tokens=True)
        )

    return "\n".join(out_lines)

# =====================================================
# Summarize ALL pages together (professional legal style)
# =====================================================
def summarize_all_pages(pages):
    full_text = "\n\n".join(p["text"] for p in pages)

    prompt = (
        "Provide a concise, professional legal summary in English of the "
        "following Russian contract. Identify the parties, subject matter, "
        "key obligations, payment terms, duration, and liability:\n\n"
        + full_text
    )

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

# =====================================================
# RunPod handler
# =====================================================
def handler(event):
    log("Handler started")

    pages = event["input"]["pages"]

    load_summary_model()
    load_translate_model()

    # ---- Summary ----
    log("Creating summary")
    raw_summary = summarize_all_pages(pages)
    summary = translate_text(raw_summary)  # ensure English

    # ---- Translation ----
    log("Translating pages")
    for p in pages:
        p["text"] = translate_text(p["text"])

    log("Handler finished")

    return {
        "summary": summary,
        "pages": pages
    }

# =====================================================
# Start RunPod serverless
# =====================================================
runpod.serverless.start({"handler": handler})

