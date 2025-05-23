# ✅ 1. ZIP'ten veri çıkar ve örnekle
import pandas as pd
import zipfile
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

# 🔓 ZIP aç
zip1 = "Network Traffic/phase1_NetworkData_FIXED.zip"
zip2 = "Network Traffic/phase2_NetworkData_FIXED.zip"
extract1 = "Network Traffic/phase1_extracted"
extract2 = "Network Traffic/phase2_extracted"
with zipfile.ZipFile(zip1, 'r') as zip_ref:
    zip_ref.extractall(extract1)
with zipfile.ZipFile(zip2, 'r') as zip_ref:
    zip_ref.extractall(extract2)

# 📄 Dosya yolları
file1 = os.path.join(extract1, "phase1_NetworkData.csv")
file2 = os.path.join(extract2, "phase2_NetworkData.csv")

# 📌 Örnekleme
def process_file(path, label, apt_frac=0.3, benign_frac=0.25):
    chunk = pd.read_csv(path, nrows=200_000)
    chunk['label'] = label
    if 'subLabel' not in chunk.columns:
        chunk['subLabel'] = 'None'
    if label == "apt":
        return chunk.sample(frac=apt_frac, random_state=42)
    else:
        return chunk.sample(frac=benign_frac, random_state=42)

df_benign = process_file(file1, label="benign")
df_apt = process_file(file2, label="apt")

# ✅ 2. Özellik seçimi ve sequence yapısı
feature_col = "Protocol_name"  # daha soyut, sınırlayıcı değil
encoder = LabelEncoder()
all_values = pd.concat([df_benign[feature_col], df_apt[feature_col]])
encoder.fit(all_values)

benign_encoded = encoder.transform(df_benign[feature_col])
apt_encoded = encoder.transform(df_apt[feature_col])

def make_sequences(data, label, window_size=10):
    sequences = []
    labels = []
    for i in range(len(data) - window_size):
        seq = data[i:i+window_size]
        sequences.append(seq)
        labels.append(0 if label == "benign" else 1)
    return sequences, labels

X_benign, y_benign = make_sequences(benign_encoded, "benign")
X_apt, y_apt = make_sequences(apt_encoded, "apt")

X = X_benign + X_apt
y = y_benign + y_apt

# ✅ 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

class CICAPTSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).long()
        self.y = torch.tensor(y).long()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = CICAPTSequenceDataset(X_train, y_train)
test_dataset = CICAPTSequenceDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ✅ 4. DeepLog Modeli
class DeepLogBinary(nn.Module):
    def __init__(self, vocab_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)
    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

model = DeepLogBinary(vocab_size=len(encoder.classes_))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ✅ 5. Eğitim Döngüsü
train_losses = []
model.train()
for epoch in range(10):
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

# 📉 Training Loss grafiği
plt.plot(train_losses, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()

# ✅ 6. Değerlendirme (Test verisiyle)
model.eval()
y_true = []
y_pred = []
y_prob = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())
        y_prob.extend(probs[:, 1].tolist())

# 🔍 ROC Curve ve AUC
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"DeepLog (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# 📋 Classification Report & Confusion Matrix
print("📋 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=["Benign", "APT"]))
print(f"✅ Accuracy: {accuracy_score(y_true, y_pred):.4f}")

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "APT"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()


#Raspberry Pi'de Test Kodu

import onnxruntime as ort
import numpy as np

# Yükle
session = ort.InferenceSession("deeplog_protocolname.onnx")

# Dummy input (örnek token dizisi)
input_ids = np.random.randint(0, 100, (1, 10)).astype(np.int64)

# Tahmin yap
outputs = session.run(["output"], {"input": input_ids})
print("🎯 Logits:", outputs[0])
