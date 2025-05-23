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

# 📌 Veri işleme fonksiyonu (güncellenmiş oranlar)
def process_file(path, label, apt_frac=0.3, benign_frac=0.25):
    chunk = pd.read_csv(path, nrows=200_000)
    chunk['label'] = label
    if 'subLabel' not in chunk.columns:
        chunk['subLabel'] = 'None'
    
    if label == "apt":
        sample = chunk.sample(frac=apt_frac, random_state=42)
    else:
        sample = chunk.sample(frac=benign_frac, random_state=42)
    
    return sample

# 🔁 Örnekleme yap
df1 = process_file(file1, "benign")
df2 = process_file(file2, "apt")
df = pd.concat([df1, df2], ignore_index=True)

# ✍️ Zenginleştirilmiş prompt üretimi
def create_prompt(row):
    return (
        f"{row['Protocol_name']} traffic from {row['Source IP']} to {row['Destination IP']} "
        f"on port {row['Destination Port']} | flow_duration={row['flow_duration']}ms, "
        f"Rate={row['Rate']} bytes/s, SourceRate={row['Srate']} pkts/s | "
        f"SYN={row['syn_flag_number']}, ACK={row['ack_flag_number']}. "
        "Is this a typical benign flow or does it indicate an advanced persistent threat?"
    )

df["input"] = df.apply(create_prompt, axis=1)
df["output"] = df["label"]

# 💾 JSONL formatında kaydet
output_path = "llm_prompts.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        json.dump({"input": row["input"], "output": row["output"]}, f)
        f.write("\n")

print(f"✅ Zenginleştirilmiş prompt dosyası oluşturuldu: {output_path}")





from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import torch.nn.functional as F
import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import Trainer
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay

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

# ⚖️ Weighted Trainer tanımı
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # ✅ kwargs eklendi
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        weights = torch.tensor([0.6, 0.4]).to(logits.device)  # Benign daha önemli
        loss = F.cross_entropy(logits, labels, weight=weights)
        return (loss, outputs) if return_outputs else loss

# ⚙️ Eğitim ayarları
training_args = TrainingArguments(
    output_dir="./llm-apt-model",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    learning_rate=3e-5,
    logging_dir="./logs",
    logging_steps=50
)

# 🚀 WeightedTrainer ile eğit
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# 📊 Değerlendirme
predictions = trainer.predict(split_dataset["test"])
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)

print("📋 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=["Benign", "APT"]))

cm = confusion_matrix(y_true, y_pred)
print("🔄 Confusion Matrix:\n", cm)

RocCurveDisplay.from_predictions(y_true, predictions.predictions[:, 1], name="DistilBERT APT Detector")
plt.title("🧠 ROC Curve")
plt.grid(True)
plt.show()

