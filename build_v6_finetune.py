"""
build_v6_finetune.py — Tao notebook huan luyen Qwen3-8B voi Unsloth QLoRA
"""
import json

def make_cell(source, cell_type="code"):
    return {
        "cell_type": cell_type,
        "source": source,
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None,
    }

cells = []

# ── Cell 0: Header ────────────────────────────────────────────────
cells.append(make_cell("""#!/usr/bin/env python3
\"\"\"
notebook_v6_finetune.py -- Unsloth QLoRA Fine-Tuning for Logic Reasoning

EXACT 2026 -- XAI Challenge @ IJCNN
Qwen3-8B + Unsloth + QLoRA | Kaggle T4x2

Strategy:
  - Knowledge Distillation: Teach Qwen3-8B logic reasoning via CoT
  - Data: Logic_Based_Educational_Queries (Train split 85%)
  - Output: LoRA adapter saved to /kaggle/working/qwen3_logic_lora
\"\"\"
"""))

# ── Cell 1: Kaggle T4 Fix ─────────────────────────────────────────
cells.append(make_cell("""# ==================================================================
# FIX cho Kaggle T4 -- CELL DAU TIEN
# ==================================================================
import os, shutil, glob

STUB_DIR = "/tmp/cuda_stubs"
os.makedirs(STUB_DIR, exist_ok=True)
stub = os.path.join(STUB_DIR, "libcuda.so")
if os.path.exists(stub) or os.path.islink(stub):
    os.remove(stub)
for candidate in [
    "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
    "/usr/lib/x86_64-linux-gnu/libcuda.so",
    *glob.glob("/usr/**/libcuda.so*", recursive=True),
]:
    if os.path.exists(candidate) and not os.path.islink(candidate):
        os.symlink(candidate, stub)
        print(f"Symlink: {stub} -> {candidate}")
        break
os.environ["LIBRARY_PATH"] = f"{STUB_DIR}:" + os.environ.get("LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{STUB_DIR}:" + os.environ.get("LD_LIBRARY_PATH", "")
shutil.rmtree("/root/.cache/flashinfer", ignore_errors=True)
print("Kaggle T4 fixes applied!")
"""))

# ── Cell 2: Install ───────────────────────────────────────────────
cells.append(make_cell("""# ==================================================================
# INSTALL DEPENDENCIES
# ==================================================================
import subprocess, sys

# Unsloth (auto-detects CUDA version)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "--break-system-packages", "unsloth"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--upgrade",
                "--break-system-packages", "--no-cache-dir",
                "trl", "peft", "accelerate", "bitsandbytes"], check=True)

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name} ({p.total_memory / 1024**3:.1f} GB)")
print("Install OK")
"""))

# ── Cell 3: Config ────────────────────────────────────────────────
cells.append(make_cell("""# ==================================================================
# CONFIG
# ==================================================================

# Model
MODEL_PATH = "/kaggle/input/models/qwen-lm/qwen-3/transformers/8b/1"
MAX_SEQ_LENGTH = 4096
LOAD_IN_4BIT = True

# LoRA
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",       # MLP (quan trong cho logic)
]

# Training
EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 4          # Effective batch = 8
LEARNING_RATE = 2e-4
WARMUP_STEPS = 10
WEIGHT_DECAY = 0.01
OPTIMIZER = "adamw_8bit"
LR_SCHEDULER = "linear"
LOGGING_STEPS = 5

# Data
DATASET_PATH = "/kaggle/input/logic-based-educational-queries/Logic_Based_Educational_Queries (2).json"
SEED = 42
TRAIN_RATIO = 0.85
VAL_RATIO = 0.10
TEST_RATIO = 0.05

# Output
LORA_OUTPUT_DIR = "/kaggle/working/qwen3_logic_lora"
CHECKPOINT_DIR = "/kaggle/working/checkpoints"

print("Config OK")
"""))

# ── Cell 4: Load Model ───────────────────────────────────────────
cells.append(make_cell("""# ==================================================================
# LOAD MODEL WITH UNSLOTH
# ==================================================================
from unsloth import FastLanguageModel

print(f"Loading {MODEL_PATH} (4-bit)...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,           # Auto-detect (float16 on T4)
    load_in_4bit=LOAD_IN_4BIT,
)

print(f"Model loaded: {model.config._name_or_path}")
print(f"Parameters: {model.num_parameters() / 1e9:.2f}B (quantized)")

# --- Attach LoRA ---
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Tiet kiem 60% VRAM
    random_state=SEED,
)

# Print trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
print("Model + LoRA ready!")
"""))

# ── Cell 5: Prepare Data ─────────────────────────────────────────
cells.append(make_cell("""# ==================================================================
# PREPARE TRAINING DATA
# ==================================================================
import json, random

# Load dataset
with open(DATASET_PATH, encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"Raw dataset: {len(raw_data)} samples")

# Shuffle with fixed seed
rng = random.Random(SEED)
rng.shuffle(raw_data)

# Split
n = len(raw_data)
n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)

train_raw = raw_data[:n_train]
val_raw = raw_data[n_train:n_train + n_val]
test_raw = raw_data[n_train + n_val:]  # KHONG DUNG

print(f"Train: {len(train_raw)} | Val: {len(val_raw)} | Test: {len(test_raw)} (reserved)")

# --- Flatten: 1 sample (N questions) -> N training examples ---
SYSTEM_PROMPT = (
    "Ban la chuyen gia suy luan logic. "
    "Dua vao cac tien de, hay phan tich tung buoc roi tra loi cau hoi. "
    "Tra loi bang JSON: {\\\"reasoning\\\": \\\"...\\\", \\\"answer\\\": \\\"A|B|C|D|Yes|No|Unknown\\\"}"
)

def flatten_samples(data_split):
    samples = []
    for item in data_split:
        premises = item.get("premises-NL", [])
        premises_text = "\\n".join(f"P{i+1}: {p}" for i, p in enumerate(premises))

        questions = item.get("questions", [])
        answers = item.get("answers", [])
        explanations = item.get("explanation", [])

        for q_idx in range(len(questions)):
            q = questions[q_idx]
            a = answers[q_idx] if q_idx < len(answers) else "Unknown"
            e = explanations[q_idx] if q_idx < len(explanations) else ""

            # Clean explanation (remove newlines that break JSON)
            e_clean = e.replace('\\n', ' ').replace('"', '\\\\"')

            user_msg = f"## Tien de:\\n{premises_text}\\n\\n## Cau hoi:\\n{q}"
            assistant_msg = f'{{\\"reasoning\\": \\"{e_clean}\\", \\"answer\\": \\"{a}\\"}}'

            samples.append({
                "conversations": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ]
            })
    return samples

train_flat = flatten_samples(train_raw)
val_flat = flatten_samples(val_raw)

print(f"Flattened -- Train: {len(train_flat)} | Val: {len(val_flat)}")

# --- Oversample Unknown (x2) ---
unknown_samples = [s for s in train_flat
                   if '"answer": "Unknown"' in s["conversations"][-1]["content"]]
print(f"Unknown samples in train: {len(unknown_samples)}")
train_flat.extend(unknown_samples)  # x2
rng.shuffle(train_flat)

print(f"After oversampling: Train = {len(train_flat)}")

# --- Convert to HuggingFace Dataset ---
from datasets import Dataset

train_ds = Dataset.from_list(train_flat)
val_ds = Dataset.from_list(val_flat)

print(f"HF Dataset -- Train: {len(train_ds)} | Val: {len(val_ds)}")
print("Sample[0] keys:", train_ds[0].keys())
print("Data ready!")
"""))

# ── Cell 6: Setup Trainer ─────────────────────────────────────────
cells.append(make_cell("""# ==================================================================
# SETUP TRAINER
# ==================================================================
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import get_chat_template, train_on_responses_only_with_padding

# Apply chat template (Qwen3 uses qwen-2.5 compatible template)
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",
)

# Formatting function
def formatting_func(examples):
    convos = examples["conversations"]
    texts = []
    for convo in convos:
        text = tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}

# SFT Config
sft_config = SFTConfig(
    output_dir=CHECKPOINT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    optim=OPTIMIZER,
    lr_scheduler_type=LR_SCHEDULER,
    logging_steps=LOGGING_STEPS,
    save_strategy="epoch",
    save_total_limit=2,
    seed=SEED,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=False,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=sft_config,
    formatting_func=formatting_func,
)

# Train on responses only (CRITICAL: only compute loss on assistant output)
trainer = train_on_responses_only_with_padding(
    trainer,
    instruction_part="<|im_start|>user\\n",
    response_part="<|im_start|>assistant\\n",
)

print("Trainer ready!")
print(f"  Epochs: {EPOCHS}")
print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM}")
print(f"  Train samples: {len(train_ds)}")
print(f"  Steps/epoch: ~{len(train_ds) // (BATCH_SIZE * GRAD_ACCUM)}")
"""))

# ── Cell 7: Train ────────────────────────────────────────────────
cells.append(make_cell("""# ==================================================================
# TRAIN
# ==================================================================
import time

print("=" * 60)
print("  BAT DAU HUAN LUYEN")
print("=" * 60)

t0 = time.time()
train_result = trainer.train()
t_train = time.time() - t0

print(f"\\nTraining done in {t_train / 60:.1f} minutes")
print(f"  Final loss: {train_result.training_loss:.4f}")

# Eval
eval_result = trainer.evaluate()
print(f"  Val loss: {eval_result.get('eval_loss', 'N/A')}")
"""))

# ── Cell 8: Save LoRA Adapter ─────────────────────────────────────
cells.append(make_cell("""# ==================================================================
# SAVE LORA ADAPTER
# ==================================================================
import os

print(f"Saving LoRA adapter to {LORA_OUTPUT_DIR}...")
model.save_pretrained(LORA_OUTPUT_DIR)
tokenizer.save_pretrained(LORA_OUTPUT_DIR)

# Verify
adapter_size = sum(
    os.path.getsize(os.path.join(LORA_OUTPUT_DIR, f))
    for f in os.listdir(LORA_OUTPUT_DIR)
    if os.path.isfile(os.path.join(LORA_OUTPUT_DIR, f))
) / 1024 / 1024

print(f"Adapter saved! Size: {adapter_size:.1f} MB")
print(f"Files: {os.listdir(LORA_OUTPUT_DIR)}")
print()
print("=" * 60)
print("  HUAN LUYEN HOAN TAT")
print("  LoRA adapter: " + LORA_OUTPUT_DIR)
print("  Su dung adapter nay trong notebook_v6_inference.ipynb")
print("=" * 60)
"""))

# ── Cell 9: Quick Sanity Check ────────────────────────────────────
cells.append(make_cell("""# ==================================================================
# QUICK SANITY CHECK (Optional)
# ==================================================================
# Test the fine-tuned model on 1 sample

FastLanguageModel.for_inference(model)

test_premise = "P1: Every student with GPA 4.0 gets a scholarship.\\nP2: Alice has GPA 4.0."
test_question = "Does Alice get a scholarship? (Yes/No)"

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": f"## Tien de:\\n{test_premise}\\n\\n## Cau hoi:\\n{test_question}"},
]

inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to("cuda")

outputs = model.generate(
    input_ids=inputs, max_new_tokens=256,
    temperature=0.1, do_sample=True,
)

response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
print("Test Input:")
print(f"  {test_premise}")
print(f"  Q: {test_question}")
print(f"\\nModel Output:")
print(f"  {response}")
print("\\nSanity check done!")
"""))

# ── Assemble Notebook ─────────────────────────────────────────────
notebook = {
    "metadata": {
        "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.12"},
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",
            "dataSources": [
                {"sourceType": "datasetVersion", "sourceId": 16409819,
                 "datasetId": 10492742, "databundleVersionId": 17405679},
                {"sourceType": "modelInstanceVersion", "sourceId": 166258,
                 "databundleVersionId": 10159036, "modelInstanceId": 141469, "modelId": 164048},
            ],
            "dockerImageVersionId": 31329,
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
            "isGpuEnabled": True,
        },
    },
    "nbformat_minor": 4,
    "nbformat": 4,
    "cells": cells,
}

OUTPUT = r"C:\Users\Minh Triet\Desktop\SKIBIDI\notebook_v6_finetune.ipynb"
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"SUCCESS: Created {OUTPUT}")
print(f"  Cells: {len(cells)}")
