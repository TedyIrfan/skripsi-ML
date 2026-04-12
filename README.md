# 🤖 Deteksi Tulisan AI vs Manusia — Bahasa Indonesia


https://drive.google.com/drive/folders/1CSNpGkm1fTLvriVB1SFY5XK1-1WlYg-w?usp=sharing

**Skripsi S1 Teknik Informatika**
Sistem klasifikasi machine learning untuk membedakan teks buatan AI dan manusia dalam Bahasa Indonesia.

---

## 📊 Status Project

| Phase | Deskripsi | Status |
|-------|-----------|--------|
| ✅ Phase 1 | Pengumpulan & pembersihan dataset | **SELESAI** |
| ✅ Phase 2 | Pembangunan dataset final | **SELESAI** |
| ✅ Phase 3 | Pembersihan & reorganisasi project | **SELESAI** |
| 🔄 Phase 4 | Training 4 model algoritma | **BELUM** |
| ⏳ Phase 5 | Evaluasi & visualisasi | Menunggu Phase 4 |
| ⏳ Phase 6 | Analisis lanjutan | Menunggu Phase 4 |
| ⏳ Phase 7 | Web App demo | Menunggu Phase 5 |

---

## 📦 Dataset

### File Dataset Utama

| File | Keterangan | Jumlah |
|------|-----------|--------|
| `dataset_skripsi_manusia_ai_1510.csv` | **Dataset Final** siap training | 1.510 teks |
| `data_ai_all_clean.csv` | Raw data AI | 760 teks |
| `data_manusia_all_clean.csv` | Raw data Manusia | 750 teks |
| `dataset_ai_tidak_terpakai.csv` | Cadangan AI (tidak dipakai) | 500 teks |

### Komposisi Dataset Final

```
dataset_skripsi_manusia_ai_1510.csv
├── AI      : 760 teks (50.3%)
└── MANUSIA : 750 teks (49.7%)
    Total   : 1.510 teks
```

### Sumber Data Manusia (750 teks)

| Sumber | Jumlah | Jenis |
|--------|--------|-------|
| IndoSum | 249 | Artikel berita |
| Kaggle Reddit Indonesia | 223 | Diskusi internet |
| Kaggle Twitter PPKM | 161 | Komentar Twitter |
| Kaggle Tweet Terrorism | 82 | Berita terorisme |
| Kaggle YouTube Comments | 29 | Komentar YouTube |
| Kaggle Marketplace Reviews | 6 | Review produk |

### Sumber Data AI (760 teks)

| Model | Jumlah |
|-------|--------|
| OpenRouter-Trinity | 100 |
| OpenRouter-TNG-R1T-Chimera | 100 |
| HF-Qwen3-Coder-Next | 100 |
| OpenRouter-DeepSeekR1-0528 | 99 |
| OpenRouter-GLM-4.5-Air | 82 |
| Groq-GPT-OSS-120B | 80 |
| OpenRouter-Step-3.5-Flash | 71 |
| GPT_Casual | 50 |
| Groq_Llama_3.3 | 48 |
| HF-Kimi-K2.5 | 30 |

---

## 🧠 Algoritma yang Digunakan (4 Model)

| No | Algoritma | Deskripsi |
|----|-----------|-----------|
| 1 | **Logistic Regression** | Cara paling simpel, menarik garis pemisah linier |
| 2 | **SVM (RBF Kernel)** | Cari hyperplane pemisah paling optimal |
| 3 | **Random Forest** | 100 pohon keputusan, hasil voting bersama |
| 4 | **IndoBERT** | Transformer khusus Bahasa Indonesia, paling canggih |

**Feature Extraction:** TF-IDF (5.000 fitur, unigram + bigram)

---

## 🗂️ Struktur Folder & File

```
📁 1_data/
└── build_dataset.py          → Gabungkan AI + Manusia → dataset final ✅

📁 2_training/
├── train_strict_cv.py        → Training LR + SVM + RF (10-fold CV, no leakage)
├── train_indobert.py         → Training IndoBERT (transformer)
├── train_group_cv.py         → Validasi per sumber data (lebih ketat)
└── hyperparameter_tuning.py  → Grid search parameter terbaik

📁 3_evaluasi/
├── eda_analysis.py           → Grafik distribusi, wordcloud, top words
├── visualizations.py         → ROC curve, confusion matrix, learning curve
├── linguistic_analysis.py    → Analisis linguistik AI vs Manusia
├── compare_models.py         → Tabel perbandingan 4 model
├── ablation_study.py         → Kontribusi fitur TF-IDF
├── threshold_tuning.py       → Optimasi threshold confidence
├── error_analysis.py         → Analisis teks salah prediksi
├── ambiguous_analysis.py     → Analisis teks susah diklasifikasi
├── per_source_analysis.py    → Akurasi per sumber data
├── per_category_analysis.py  → Analisis per kategori
└── external_validation.py    → Uji generalisasi ke data baru

📁 4_tools/
├── inference.py              → Prediksi teks baru via terminal
└── save_and_inference.py     → Simpan model + prediksi
```

---

## 🚀 Cara Menjalankan

### Phase 4 — Training Model (Langkah Berikutnya)

```bash
# Step 1: Training LR + SVM + Random Forest
python train_strict_cv.py

# Step 2: Training IndoBERT
python train_indobert.py
```

### Phase 5 — Evaluasi & Visualisasi (setelah training)

```bash
python eda_analysis.py
python visualizations.py
python linguistic_analysis.py
python compare_models.py
python ablation_study.py
python threshold_tuning.py
python error_analysis.py
python per_source_analysis.py
```

### Prediksi Teks Baru

```python
import joblib

# Load model pipeline
pipeline = joblib.load('models_strict/best_pipeline_logistic_regression.pkl')

# Prediksi
text = "Masukkan teks di sini..."
label = pipeline.predict([text])[0]
proba = pipeline.predict_proba([text])[0]

print(f"Prediksi : {'AI' if label == 1 else 'MANUSIA'}")
print(f"Confidence: {max(proba)*100:.2f}%")
```

---

## 📋 Requirements

```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn wordcloud
pip install torch transformers  # untuk IndoBERT
pip install kagglehub           # untuk download dataset
```

---

## 📁 Folder Struktur

```
.
├── dataset_skripsi_manusia_ai_1510.csv   ← Dataset final
├── data_ai_all_clean.csv                 ← Raw AI data
├── data_manusia_all_clean.csv            ← Raw Manusia data
├── dataset_ai_tidak_terpakai.csv         ← Cadangan AI
├── backup_hasil_lama/                    ← Hasil training sebelumnya
├── [script python...]
└── README.md
```

---

## 👤 Author

Skripsi Sarjana Teknik Informatika
Dataset: 1.510 teks Bahasa Indonesia (760 AI + 750 Manusia)
