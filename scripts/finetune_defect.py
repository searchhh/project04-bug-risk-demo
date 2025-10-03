import os
import sys
import random
import numpy as np
import torch
import evaluate
from datasets import load_dataset, Value
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
import transformers as tf

print("USING FILE:", __file__)
print("TRANSFORMERS VERSION:", tf.__version__)
print("TORCH VERSION:", torch.__version__)
print("CUDA AVAILABLE:", torch.cuda.is_available())

# 固定随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

MODEL_ID = "microsoft/codebert-base"
DATA_ID = "code_x_glue_cc_defect_detection"

tok = AutoTokenizer.from_pretrained(MODEL_ID)


def tokenize(batch):
    return tok(batch["func"], truncation=True, padding="max_length", max_length=256)


ds = load_dataset(DATA_ID)  # splits: train / validation / test
print("AVAILABLE SPLITS:", list(ds.keys()))
val_key = "validation" if "validation" in ds else (
    "valid" if "valid" in ds else None)
assert val_key is not None, f"No validation split found. Available: {list(ds.keys())}"

# 分词 & 列处理
ds = ds.map(tokenize, batched=True)
ds = ds.rename_column("target", "labels")

# 👇 关键：确保 labels 是整型单标签，而不是 float
ds = ds.cast_column("labels", Value("int64"))

cols = ["input_ids", "attention_mask", "labels"]
ds = ds.remove_columns([c for c in ds["train"].column_names if c not in cols])

# 让 HF 直接返回 torch.Tensor，labels dtype 保持 int64
ds = ds.with_format(type="torch", columns=cols)

# 模型 & 明确问题类型为单标签分类（避免走 BCEWithLogits）
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, num_labels=2)
model.config.problem_type = "single_label_classification"

# 度量
metric_acc = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1":       metric_f1.compute(predictions=preds, references=labels, average="binary")["f1"],
    }


# 训练参数（新旧兼容）
common_kwargs = dict(
    output_dir="codebert-defect-out",
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    logging_steps=50,
    fp16=True,
    report_to="none",
    seed=SEED,
)
try:
    args = TrainingArguments(
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        **common_kwargs,
    )
except TypeError:
    args = TrainingArguments(save_steps=500, **common_kwargs)
    print("⚠️ Using legacy TrainingArguments (no evaluation_strategy/save_strategy).")

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds[val_key],
    compute_metrics=compute_metrics,
)

trainer.train()
print("\n== EVAL on TEST ==")
print(trainer.evaluate(ds["test"]))

out_dir = ".\output\codebert-defect-model"
trainer.save_model(out_dir)
print(f"Saved model to: {os.path.abspath(out_dir)}")
