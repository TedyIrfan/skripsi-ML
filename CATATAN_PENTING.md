# 📌 CATATAN PENTING — Ringkasan Percakapan & Progress Skripsi

> Dibuat: 10 April 2026  
> Berdasarkan sesi kerja bersama AI assistant

---

## 🗂️ STRUKTUR FOLDER PROYEK (setelah sesi ini)

```
📁 Praktek skripsi ml/
│
├── 📁 0_dataset/              ← Raw data mentah
│   ├── data_ai_all_clean.csv
│   ├── data_manusia_all_clean.csv
│   ├── dataset_skripsi_manusia_ai_1510.csv  ← DATASET FINAL
│   └── dataset_ai_tidak_terpakai.csv
│
├── 📁 1_data/
│   └── build_dataset.py       ← Gabung AI + Manusia → dataset final
│
├── 📁 2_training/
│   ├── train_strict_cv.py     ← ✅ SUDAH DIJALANKAN (training LR + SVM + RF)
│   ├── train_indobert.py      ← ⏳ Belum dijalankan
│   ├── train_group_cv.py      ← ✅ SUDAH DIJALANKAN (validasi Group CV)
│   └── hyperparameter_tuning.py
│
├── 📁 3_evaluasi/             ← ⏳ Belum dijalankan
│   ├── eda_analysis.py
│   ├── compare_models.py      ← sudah di-fix bug
│   ├── ablation_study.py      ← sudah di-fix bug
│   ├── threshold_tuning.py
│   ├── error_analysis.py
│   ├── external_validation.py ← ✅ SUDAH DIJALANKAN
│   └── ... (script lainnya)
│
├── 📁 4_tools/
│   └── inference.py           ← sudah di-fix bug, siap pakai setelah training
│
├── 📁 models_strict/          ← Output training (file asli untuk script lain)
│   ├── best_pipeline_logistic_regression.pkl
│   └── strict_cv_results.json
│
├── 📁 hasil_phase4_training_klasik/   ← ✅ BUNDLING HASIL SESI INI
│   ├── README.md              ← Penjelasan lengkap hasil
│   ├── strict_cv_results.json
│   ├── best_pipeline_logistic_regression.pkl
│   ├── external_validation_data.csv
│   ├── external_validation_results.json
│   └── external_validation_detailed.csv
│
├── dataset_skripsi_manusia_ai_1510.csv  ← Copy di root untuk training
├── external_validation.csv
├── CATATAN_PENTING.md         ← FILE INI
└── README.md
```

---

## 🐛 BUG YANG DITEMUKAN & DIPERBAIKI

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `compare_models.py` | `df['teks']` → kolom tidak ada | Ganti ke `df['text']` |
| 2 | `train_group_cv.py` | `min(5, n_splits)` → variabel belum ada | Ganti ke `min(5, n_groups)` |
| 3 | `ablation_study.py` | Load pipeline 2x, panggil `.transform()` dan `.feature_importances_` salah | Ekstrak step pipeline dengan `.named_steps['tfidf']` dan `.named_steps['clf']`, feature importance sesuai jenis model |
| 4 | `inference.py` | Load `label_mapping.pkl` dari path yang tidak ada, pakai variabel lama | Hardcode label mapping, gunakan `pipeline.predict()` langsung |
| 5 | `train_strict_cv.py` | Karakter `✓` error di Windows (encoding cp1252) | Ganti jadi `[OK]`, set `PYTHONIOENCODING=utf-8` |
| 6 | Semua training script | Dataset tidak ditemukan (path relatif) | Copy dataset ke root project, jalankan dari root |

---

## 📊 HASIL TRAINING PHASE 4 — 3 ALGORITMA KLASIK

**Script:** `2_training/train_strict_cv.py`  
**Dataset:** 1510 teks (760 AI + 750 Manusia)  
**Split:** 80% train (1208) / 20% test (302), `stratify=y`, `random_state=42`  
**Metode:** 10-Fold Stratified Cross-Validation dengan Pipeline (anti data leakage)

### Hasil

| Model | CV Mean | CV Std | Test Accuracy | FNR |
|-------|---------|--------|--------------|-----|
| Logistic Regression | 99.75% | ±0.53% | **99.67%** | **0.00%** |
| SVM (RBF Kernel) | 99.75% | ±0.53% | **99.67%** | **0.00%** |
| Random Forest | 98.59% | ±0.98% | 98.34% | **0.00%** |

Best Model: **Logistic Regression** → disimpan di `models_strict/`

---

## ⚠️ TEMUAN PENTING: HASIL TERLALU SEMPURNA

Akurasi 99.67% memunculkan pertanyaan ilmiah penting. Dilakukan 2 validasi:

### Validasi 1 — Group CV (`train_group_cv.py`)

Sumber data dipisah antar fold agar tidak bocor.

| Model | Group CV | Standard CV | Penurunan |
|-------|----------|-------------|-----------|
| SVM | 96.74% | 99.74% | **-3.00%** ← paling robust |
| Logistic Regression | 87.29% | 99.74% | -12.44% |
| Random Forest | 81.49% | 98.61% | -17.12% |

### Validasi 2 — External Validation (30 teks baru)

Teks benar-benar baru, tidak dari sumber dataset training.

```
Accuracy:  73.33%
Precision: 70.59%
Recall:    80.00%
F1-Score:  75.00%
Errors:    8/30 (5 FP + 3 FN)
```

### Kesimpulan Validasi

| Cek | Hasil |
|-----|-------|
| Data leakage teknis? | ✅ TIDAK ADA (Pipeline + overlap=0) |
| Model overfit distribusi training? | ⚠️ YA — ada domain gap |
| Penyebab | Dataset AI sangat formal, dataset Manusia punya penanda sumber (Twitter @#, nama media) |
| Konsekuensi | Model kesulitan pada teks baru tanpa penanda sumber yang khas |

### Kalimat untuk Skripsi (Keterbatasan Penelitian)

> *"Meskipun model mencapai akurasi 99.67% pada uji internal (distribusi sama dengan training), pengujian Group CV menunjukkan penurunan performa menjadi 87.29% saat sumber data dipisah. External validation pada 30 teks baru menunjukkan akurasi 73.33%, mengindikasikan ketergantungan model terhadap karakteristik distribusi sumber data. Hal ini merupakan keterbatasan penelitian yang perlu dipertimbangkan."*

---

## 🔢 PEMAHAMAN ALGORITMA & ALUR SISTEM

### Feature Extraction: TF-IDF
```python
TfidfVectorizer(
    max_features=5000,   # 5000 kata/frasa terpenting
    ngram_range=(1, 2),  # unigram + bigram
    min_df=2,            # minimal muncul di 2 dokumen
    max_df=0.8           # maksimal di 80% dokumen
)
```

### Pipeline (Anti-Leakage)
```
Setiap Fold CV:
  Data train fold → fit TF-IDF → transform → latih model → ↓
  Data test fold             → transform saja → evaluasi ↑
```

### 4 Algoritma Model

| Model | Cara Kerja | Status |
|-------|-----------|--------|
| Logistic Regression | Garis batas linier di ruang 5000 dimensi | ✅ Terlatih |
| SVM RBF | Hyperplane + kernel non-linier | ✅ Terlatih |
| Random Forest | 100 pohon keputusan, voting mayoritas | ✅ Terlatih |
| IndoBERT | Transformer 110 juta parameter, pra-latih Bahasa Indonesia | ⏳ Belum |

---

## ⏭️ YANG HARUS DILAKUKAN SELANJUTNYA

### Segera (Phase 4 selesai)
```bash
# Training IndoBERT (estimasi 30-60 menit CPU)
$env:PYTHONIOENCODING='utf-8'
python 2_training/train_indobert.py
```

### Setelah IndoBERT (Phase 5 — Evaluasi)
```bash
python 3_evaluasi/eda_analysis.py
python 3_evaluasi/compare_models.py
python 3_evaluasi/visualizations.py
python 3_evaluasi/ablation_study.py
python 3_evaluasi/threshold_tuning.py
python 3_evaluasi/error_analysis.py
python 3_evaluasi/per_source_analysis.py
```

### Cara Jalankan yang Benar
Selalu jalankan dari **ROOT folder** project, bukan dari dalam subfolder:
```bash
# Posisi: c:\Users\irfan\Project Code\Praktek skripsi ml\
$env:PYTHONIOENCODING='utf-8'
python 2_training/train_strict_cv.py   # BENAR
# BUKAN: cd 2_training && python train_strict_cv.py  ← akan error path dataset
```

---

## 💾 FILE MODEL YANG ADA

| File | Lokasi | Keterangan |
|------|--------|-----------|
| `best_pipeline_logistic_regression.pkl` | `models_strict/` | Model aktif (digunakan script lain) |
| `strict_cv_results.json` | `models_strict/` | Angka hasil CV dan test |
| Salinan semua file di atas | `hasil_phase4_training_klasik/` | Arsip rapi sesi ini |
