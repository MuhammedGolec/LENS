import pandas as pd
import zipfile
import os
import json

# 📁 ZIP dosyalarını aç
zip1 = "Network Traffic/phase1_NetworkData_FIXED.zip"
zip2 = "Network Traffic/phase2_NetworkData_FIXED.zip"

extract1 = "Network Traffic/phase1_extracted"
extract2 = "Network Traffic/phase2_extracted"

with zipfile.ZipFile(zip1, 'r') as zip_ref:
    zip_ref.extractall(extract1)

with zipfile.ZipFile(zip2, 'r') as zip_ref:
    zip_ref.extractall(extract2)

print("✅ ZIP dosyaları açıldı")

# 📄 CSV dosya yolları
file1 = os.path.join(extract1, "phase1_NetworkData.csv")
file2 = os.path.join(extract2, "phase2_NetworkData.csv")

# 📌 Tüm veriyi oku ve etiketle
df1 = pd.read_csv(file1)
df1["label"] = "benign"

df2 = pd.read_csv(file2)
df2["label"] = "apt"

# 🔗 Birleştir
df = pd.concat([df1, df2], ignore_index=True)

# 🧾 subLabel eksikse None ekle
if "subLabel" not in df.columns:
    df["subLabel"] = "None"

# ✍️ Prompt üretimi
def create_prompt(row):
    return (
        f"{row['Protocol_name']} traffic from {row['Source IP']} to {row['Destination IP']} "
        f"on port {row['Destination Port']} | flow_duration={row['flow_duration']}ms, "
        f"Rate={row['Rate']} bytes/s, SourceRate={row['Srate']} pkts/s | "
        f"SYN={row['syn_flag_number']}, ACK={row['ack_flag_number']}. "
        "Is this traffic benign or an APT threat?"
    )

df["input"] = df.apply(create_prompt, axis=1)
df["output"] = df["label"]

# 💾 JSONL olarak kaydet
output_path = "llm_prompts_full.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        json.dump({"input": row["input"], "output": row["output"]}, f)
        f.write("\n")

print(f"✅ Tüm veriyle JSONL dosyası oluşturuldu: {output_path}")



from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, RocCurveDisplay
import matplotlib.pyplot as plt
import numpy as np
import json
import torch

# 🔽 JSONL veri setini yükle (önceden oluşturulmuş)
dataset = Dataset.from_json("llm_prompts_full.jsonl")

# 🎯 Etiketleri sayıya çevir
label_map = {"benign": 0, "apt": 1}
dataset = dataset.map(lambda e: {"label": label_map[e["output"]]})

# 🧪 ✅ SADECE 2 MİLYON SATIR KULLAN (hızlı eğitim için)
dataset = dataset.shuffle(seed=42).select(range(2_000_000))

# 🧠 Tokenizer ve model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 🔠 Tokenizer uygulaması
def tokenize(example):
    return tokenizer(example["input"], padding="max_length", truncation=True, max_length=256)

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ✂️ Eğitim/test böl
split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)

# 🎯 Metrikler
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# ⚙️ HIZLI EĞİTİM İÇİN EĞİTİM AYARLARI
training_args = TrainingArguments(
    output_dir="./llm-apt-model",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    learning_rate=3e-5,
    logging_dir="./logs",
    logging_steps=1000,
    save_total_limit=1,
    fp16=True  # ⚡ GPU hızlandırma
)

# 🚀 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 🏋️‍♂️ Eğitimi başlat
trainer.train()

# 🔍 Değerlendirme
predictions = trainer.predict(split_dataset["test"])
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)

# 📋 Classification Report
print("📋 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=["Benign", "APT"]))

# 🔄 Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("🔄 Confusion Matrix:\n", cm)

# 📈 ROC Curve
RocCurveDisplay.from_predictions(y_true, predictions.predictions[:, 1], name="DistilBERT APT Detector")
plt.title("🧠 ROC Curve")
plt.grid(True)
plt.show()