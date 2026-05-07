# Coronary Artery Disease Detection Using Random Forest and K-Means Clustering

## Deskripsi Project

Project ini bertujuan untuk melakukan analisis data medis untuk mendeteksi Coronary Artery Disease (CAD) menggunakan metode Machine Learning dan Clustering.

Metode yang digunakan:

- Random Forest (Supervised Learning)
- K-Means Clustering (Unsupervised Learning)

Dataset yang digunakan adalah dataset CAD Alizadeh yang berisi informasi klinis pasien seperti usia, BMI, tekanan darah, kolesterol, diabetes, dan berbagai indikator medis lainnya.

---

## Dataset

Dataset yang digunakan:

- `CAD alizadeh.xls`

Dataset memiliki:

- 303 baris data
- 56 kolom fitur medis
- Target klasifikasi: `Cath`

Target:
- `Cad` → pasien terindikasi CAD
- `Normal` → pasien normal

---

## Tahapan Project

### 1. Preprocessing Data

Tahapan preprocessing meliputi:

- Membaca dataset Excel menggunakan Pandas
- Mengecek missing values
- Encoding data kategorikal menggunakan LabelEncoder
- Konversi seluruh fitur menjadi numerik
- Normalisasi data menggunakan MinMaxScaler

---

## Random Forest Classification

### Pembagian Data

Dataset dibagi menjadi:

- 80% data training
- 20% data testing

Menggunakan:

```python
train_test_split(test_size=0.2, random_state=42)
```

---

### Training Model

Model Random Forest dibuat menggunakan:

```python
RandomForestClassifier(n_estimators=100, random_state=42)
```

---

### Hasil Evaluasi Random Forest

| Metric | Value |
|---|---|
| Accuracy | 85.25% |
| Precision | 84.62% |
| Recall | 61.11% |
| F1-Score | 70.97% |
| Specificity | 95.35% |

### Confusion Matrix

```text
[[41  2]
 [ 7 11]]
```

### Kesimpulan Random Forest

Model Random Forest memberikan performa yang cukup baik dalam mendeteksi CAD dengan akurasi mencapai 85.25%. Nilai specificity yang tinggi menunjukkan model sangat baik dalam mengenali pasien normal, meskipun recall masih dapat ditingkatkan untuk mendeteksi lebih banyak pasien positif CAD.

---

# K-Means Clustering

## Fitur yang Digunakan

Fitur yang dipilih:

- Age
- BMI
- LDL
- HDL
- FBS
- TG
- HTN
- DM

---

## Penentuan Jumlah Cluster

Metode Elbow digunakan untuk menentukan jumlah cluster optimal.

Visualisasi dilakukan menggunakan grafik distortion terhadap jumlah cluster.

---

## Implementasi K-Means Manual

Algoritma K-Means dibangun tanpa library clustering bawaan, meliputi:

- Inisialisasi centroid
- Perhitungan Euclidean Distance
- Assignment cluster
- Update centroid
- Iterasi hingga konvergen

---

## Evaluasi Clustering

### Silhouette Score

```text
0.5242
```

### Mapped Accuracy terhadap Label CAD

```text
71.28%
```

### Kesimpulan K-Means

Hasil clustering menunjukkan pemisahan cluster yang cukup baik dengan silhouette score sebesar 0.52. Setelah dilakukan mapping terhadap label asli CAD, diperoleh akurasi sekitar 71.28%.

---

## Simulasi Data Dummy

Project ini juga melakukan pengujian menggunakan data dummy untuk melihat hasil cluster pasien baru.

### Contoh Hasil

#### Dummy 1

```text
Cluster: 1
Mayoritas Label: CAD
```

#### Dummy 2

```text
Cluster: 2
Mayoritas Label: Normal
```

---

# Library yang Digunakan

- pandas
- numpy
- matplotlib
- scikit-learn
- scipy

---

# Cara Menjalankan Project

1. Clone repository:

```bash
git clone https://github.com/username/repository-name.git
```

2. Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn scipy openpyxl
```

3. Jalankan notebook:

```bash
jupyter notebook
```

---

# Struktur Project

```text
├── CAD alizadeh.xls
├── cad_analysis.ipynb
├── README.md

Nama: [Nama Kamu]

Project Machine Learning - Coronary Artery Disease Detection
