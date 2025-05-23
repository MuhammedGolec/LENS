import pandas as pd
import zipfile, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)
import matplotlib.pyplot as plt

# === YOLLAR ===
zip1 = "Network Traffic/phase1_NetworkData_FIXED.zip"
zip2 = "Network Traffic/phase2_NetworkData_FIXED.zip"
extract1 = "Network Traffic/phase1_extracted"
extract2 = "Network Traffic/phase2_extracted"

with zipfile.ZipFile(zip1, 'r') as zip_ref:
    zip_ref.extractall(extract1)
with zipfile.ZipFile(zip2, 'r') as zip_ref:
    zip_ref.extractall(extract2)

file1 = os.path.join(extract1, "phase1_NetworkData.csv")
file2 = os.path.join(extract2, "phase2_NetworkData.csv")

# === VERİ OKU ===
def process_file(path, label, apt_frac=0.3, benign_frac=0.25):
    df = pd.read_csv(path, nrows=200_000)
    df['label'] = 1 if label == "apt" else 0
    return df.sample(frac=apt_frac if label == "apt" else benign_frac, random_state=42)

df_apt = process_file(file2, "apt")
df_benign = process_file(file1, "benign")
df = pd.concat([df_apt, df_benign], ignore_index=True)

# === GEREKLİ DÜZENLEMELER ===
df['pair_id'] = df['Source IP'].astype(str) + "→" + df['Destination IP'].astype(str)
df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
df = df.sort_values('ts')
df['delta_time'] = df.groupby('pair_id')['ts'].diff().dt.total_seconds().fillna(0)

# === TEMEL FLOW ÖZNİTELİKLERİ ===
flow_cols = [
    'flow_duration', 'Rate', 'Srate', 'Drate',
    'Tot sum', 'Tot size', 'IAT', 'Number', 'MAC',
    'Magnitue', 'Radius', 'Covariance', 'Variance'
]
df = df.dropna(subset=flow_cols + ['label'])

# === GRUPLA & ÖZETLE ===
features = df.groupby('pair_id').agg({
    'delta_time': ['mean', 'std', 'max'],
    'flow_duration': 'mean',
    'Rate': 'mean',
    'Srate': 'mean',
    'Drate': 'mean',
    'Tot sum': 'mean',
    'Tot size': 'mean',
    'IAT': 'mean',
    'MAC': 'mean',
    'Magnitue': 'mean',
    'Radius': 'mean',
    'Covariance': 'mean',
    'Variance': 'mean',
    'label': 'first'
}).fillna(0)

features.columns = ['_'.join(col).strip() for col in features.columns.values]
features = features.reset_index(drop=True)

# === MODEL EĞİT ===
X = features.drop(columns=['label_first'])
y = features['label_first']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# === METRİKLER ===
print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred))

print("✅ Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("✅ Precision:", round(precision_score(y_test, y_pred), 4))
print("✅ Recall   :", round(recall_score(y_test, y_pred), 4))
print("✅ F1 Score :", round(f1_score(y_test, y_pred), 4))

print("\n🧱 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# === ROC ===
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - EARLYCROW (CICAPT-IIoT Flow Features)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
