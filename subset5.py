import pandas as pd
import zipfile
import os
import json

# 🔓 ZIP DOSYALARINI AÇ
zip1 = "Network Traffic/phase1_NetworkData_FIXED.zip"
zip2 = "Network Traffic/phase2_NetworkData_FIXED.zip"

extract1 = "Network Traffic/phase1_extracted"
extract2 = "Network Traffic/phase2_extracted"

with zipfile.ZipFile(zip1, 'r') as zip_ref:
    zip_ref.extractall(extract1)

with zipfile.ZipFile(zip2, 'r') as zip_ref:
    zip_ref.extractall(extract2)

print("✅ ZIP'ler açıldı")

# 📄 CSV dosya yolları
file1 = os.path.join(extract1, "phase1_NetworkData.csv")
file2 = os.path.join(extract2, "phase2_NetworkData.csv")

# 📌 Veri işleme fonksiyonu
def process_file(path, label):
    chunk = pd.read_csv(path, nrows=200_000)
    chunk['label'] = label
    if 'subLabel' not in chunk.columns:
        chunk['subLabel'] = 'None'
    sample = chunk.sample(frac=0.05, random_state=42)
    return sample

# 🔁 Örnekleme yap
df1 = process_file(file1, "benign")
df2 = process_file(file2, "apt")
df = pd.concat([df1, df2], ignore_index=True)

# 🔧 Prompt üret
def create_prompt(row):
    return (
        f"{row['Protocol_name']} connection from {row['Source IP']} to {row['Destination IP']} "
        f"on port {row['Destination Port']}, SYN={row['syn_flag_number']}, ACK={row['ack_flag_number']}. "
        "Is this traffic benign or a threat?"
    )

df["input"] = df.apply(create_prompt, axis=1)
df["output"] = df["label"]

# 💾 JSONL formatında kaydet
output_path = "llm_prompts.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        json.dump({"input": row["input"], "output": row["output"]}, f)
        f.write("\n")

print(f"✅ Prompt dosyası kaydedildi: {output_path}")



from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

# 📂 Dataset yükle
dataset = Dataset.from_json("llm_prompts.jsonl")

# 🎯 Etiketleri sayıya çevir
label_map = {"benign": 0, "apt": 1}
dataset = dataset.map(lambda e: {"label": label_map[e["output"]]})

# 🔑 Tokenizer ve model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 🔄 Tokenize fonksiyonu
def tokenize(example):
    return tokenizer(example["input"], padding="max_length", truncation=True, max_length=256)

tokenized_dataset = dataset.map(tokenize)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# 📊 Train/Test ayır
split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)

# 🎯 Metrikler
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./llm-apt-model",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    learning_rate=2e-5,
    do_train=True,
    do_eval=True,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,
    save_steps=100,
    eval_steps=100
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()


import torch

text = "TCP connection from 192.168.1.12 to 172.16.0.3 on port 445, SYN=1, ACK=1. Is this traffic benign or a threat?"

# GPU'da çalışacağımız için model ve input'u CUDA'ya taşıyoruz
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
inputs = {k: v.to(device) for k, v in inputs.items()}  # input'u da GPU'ya al

# Tahmin
outputs = model(**inputs)
pred = torch.argmax(outputs.logits, dim=1).item()

# Etiket çözümlemesi
label_map_inv = {0: "benign ✅", 1: "APT ⚠️"}
print("🧠 Prediction:", label_map_inv[pred])

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Test seti: split_dataset["test"]
test_dataset = split_dataset["test"]
true_labels = test_dataset["label"]

# Model ve tokenizer zaten hazır — tokenları modelin beklediği forma çevir
inputs = tokenizer(test_dataset["input"], padding=True, truncation=True, max_length=256, return_tensors="pt")

# GPU'ya taşı
inputs = {k: v.to(model.device) for k, v in inputs.items()}
model.eval()

# 🔁 Tahmin yap
with torch.no_grad():
    outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

# 🎯 Confusion matrix
cm = confusion_matrix(true_labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "APT"])
disp.plot(cmap="Blues", values_format='d')
plt.title("Confusion Matrix - Test Set")
plt.show()

# 📊 Ek rapor
print("\n📋 Classification Report:\n")
print(classification_report(true_labels, preds, target_names=["Benign", "APT"]))


