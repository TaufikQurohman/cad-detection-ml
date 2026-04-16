# ==============================
# 0. PREPROCESSING
# ==============================

import pandas as pd
import numpy as np
import os

# 🔥 Path file (PASTIKAN file ada di folder yang sama)
file_path = 'CAD alizadeh.xls'

# Debug (opsional, buat cek)
print("Current Directory:", os.getcwd())
print("Files in folder:", os.listdir())

# Load dataset
df = pd.read_excel(file_path)

print("\nDataset Preview:")
print(df.head())

# ==============================
# Cek Missing Values
# ==============================

print("\nMissing Values:")
print(df.isnull().sum())

# ==============================
# Encoding (Categorical → Numeric)
# ==============================

from sklearn.preprocessing import LabelEncoder

df_clean = df.copy()
categorical_cols = df_clean.select_dtypes(include=['object']).columns

le = LabelEncoder()
for col in categorical_cols:
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))

# Pastikan semua numerik
df_clean = df_clean.apply(pd.to_numeric, errors='coerce')

# ==============================
# Normalisasi
# ==============================

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_normalized = pd.DataFrame(
    scaler.fit_transform(df_clean),
    columns=df_clean.columns
)

# ==============================
# 1. RANDOM FOREST
# ==============================

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

# Pisahkan fitur & target
X = df_normalized.drop(columns=["Cath"])
y = df_normalized["Cath"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Prediksi
y_pred_rf = rf_model.predict(X_test)

# Evaluasi
conf_matrix = confusion_matrix(y_test, y_pred_rf)

accuracy = accuracy_score(y_test, y_pred_rf)
precision = precision_score(y_test, y_pred_rf)
recall = recall_score(y_test, y_pred_rf)
f1 = f1_score(y_test, y_pred_rf)

print("\n=== RANDOM FOREST ===")
print("Confusion Matrix:\n", conf_matrix)
print(f"Akurasi: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# ==============================
# 2. K-MEANS
# ==============================

from sklearn.metrics import silhouette_score
from scipy.stats import mode

# Mapping target
df['Cath'] = df['Cath'].map({'Normal': 0, 'Cad': 1})
df = df.dropna(subset=['Cath'])
df['Cath'] = df['Cath'].astype(int)

# Pilih fitur
features = ['Age', 'BMI', 'LDL', 'HDL', 'FBS', 'TG', 'HTN', 'DM']
X_kmeans = df[features]
y_true = df['Cath'].values

# Normalisasi
scaler_k = MinMaxScaler()
X_kmeans_norm = scaler_k.fit_transform(X_kmeans)

# ==============================
# K-Means Manual
# ==============================

def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

k = 3
centroids = X_kmeans_norm[np.random.choice(len(X_kmeans_norm), k, replace=False)]

# Iterasi
for _ in range(20):
    clusters = [[] for _ in range(k)]

    for point in X_kmeans_norm:
        dists = [euclidean(point, c) for c in centroids]
        clusters[np.argmin(dists)].append(point)

    centroids = np.array([
        np.mean(cluster, axis=0) if len(cluster) > 0 else centroids[i]
        for i, cluster in enumerate(clusters)
    ])

# Assign label
labels = np.zeros(len(X_kmeans_norm))
for i, point in enumerate(X_kmeans_norm):
    dists = [euclidean(point, c) for c in centroids]
    labels[i] = np.argmin(dists)

# ==============================
# Evaluasi K-Means
# ==============================

sil_score = silhouette_score(X_kmeans_norm, labels)
print("\n=== K-MEANS ===")
print("Silhouette Score:", sil_score)

# Mapping cluster ke label
mapped_labels = np.zeros_like(labels)

for i in range(k):
    mask = (labels == i)
    if np.sum(mask) > 0:
        mapped_labels[mask] = mode(y_true[mask], keepdims=True).mode[0]

accuracy_kmeans = accuracy_score(y_true, mapped_labels)
print("Mapped Accuracy:", accuracy_kmeans)

# ==============================
# Dummy Prediction
# ==============================

dummy_input = pd.DataFrame([{
    'Age': 58,
    'BMI': 27.5,
    'LDL': 150,
    'HDL': 40,
    'FBS': 120,
    'TG': 180,
    'HTN': 1,
    'DM': 0
}])

dummy_scaled = scaler_k.transform(dummy_input)

dists = [euclidean(dummy_scaled[0], c) for c in centroids]
cluster = np.argmin(dists)

cluster_label = mode(y_true[labels == cluster], keepdims=True).mode[0]
label_name = "CAD" if cluster_label == 1 else "Normal"

print("\nDummy Prediction:")
print("Cluster:", cluster)
print("Hasil:", label_name)