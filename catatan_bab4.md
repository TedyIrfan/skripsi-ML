# CATATAN BAB 4 — Hasil dan Pembahasan
# Status: FINAL — semua training, evaluasi, dan visualisasi selesai

---

## 4.1 Hasil Exploratory Data Analysis (EDA)

**Masuk Bab 4 awal** — sebelum hasil model.
Grafik: `hasil_phase5/2_eda_dataset/`

| Analisis | Temuan |
|----------|--------|
| Distribusi label | AI: 760 (50.3%), Manusia: 750 (49.7%) — seimbang |
| Rata-rata panjang teks AI | Lebih panjang dan formal |
| Rata-rata panjang teks Manusia | Lebih pendek, ada @/#/nama media |
| Wordcloud AI | Kata dominan: implementasi, sistem, teknologi, optimal |
| Wordcloud Manusia | Kata dominan: -lah, -nya, @mention, nama media |

---

## 4.2 Hasil Pelatihan & Evaluasi 4 Model

### Tabel Perbandingan Final (Test Set = 302 data)

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | FNR |
|-------|---------|-----------|--------|----|---------|-----|
| Logistic Regression | **99.67%** | 99.35% | 100.00% | 99.67% | 99.99% | **0.00%** |
| SVM (RBF Kernel) | **99.67%** | 99.35% | 100.00% | 99.67% | 100.00% | **0.00%** |
| Random Forest | 98.34% | 96.82% | 100.00% | 98.38% | 99.86% | **0.00%** |
| IndoBERT | 97.35% | 95.00% | 100.00% | 97.44% | - | **0.00%** |

**Best Model: Logistic Regression & SVM** (seri 99.67%)
**Grafik: `hasil_phase5/1_visualisasi_utama/3_model_comparison_all.png`**

### Confusion Matrix 4 Model

| Model | TN | FP | FN | TP |
|-------|----|----|----|----|
| LR | 149 | 1 | 0 | 152 |
| SVM | 149 | 1 | 0 | 152 |
| RF | 145 | 5 | 0 | 152 |
| IndoBERT | 142 | 8 | 0 | 152 |

**Grafik: `hasil_phase5/1_visualisasi_utama/1_all_confusion_matrices.png`**

Perhitungan manual LR (untuk dimasukkan ke skripsi):
- Accuracy  = (152+149)/302 = **99.67%**
- Precision = 152/153 = **99.35%**
- Recall    = 152/152 = **100.00%**
- FNR       = 0/152 = **0.00%**

---

## 4.3 Hasil Cross-Validation

### 10-Fold Stratified CV (Standard)

| Model | CV Mean | CV Std | Test Accuracy |
|-------|---------|--------|--------------|
| LR | 99.75% | ±0.53% | 99.67% |
| SVM | 99.75% | ±0.53% | 99.67% |
| RF | 98.59% | ±0.98% | 98.34% |

**Grafik: `hasil_phase5/1_visualisasi_utama/4_cv_scores_all_models.png`**

### Group CV vs Standard CV (Domain Gap)

| Model | Standard CV | Group CV | Penurunan |
|-------|------------|---------|-----------|
| SVM | 99.75% | **96.74%** | -3.01% |
| LR | 99.75% | **87.29%** | -12.46% |
| RF | 98.59% | **81.49%** | -17.10% |

**Grafik: `hasil_phase5/1_visualisasi_utama/5_group_cv_vs_standard.png`**
**Grafik: `hasil_phase5/3_perbandingan_model/1_heatmap_metrics_lengkap.png`**

> SVM paling robust antar sumber — paling kecil penurunannya.
> Penurunan besar = ada domain gap → tulis sebagai keterbatasan.

---

## 4.4 Hyperparameter Tuning Random Forest

GridSearchCV 5-Fold, 81 kombinasi, 405 model RF dilatih.

| Parameter | Default | Best (Tuned) |
|-----------|---------|-------------|
| n_estimators | 100 | 50 |
| max_depth | 20 | None (unlimited) |
| min_samples_split | 5 | 2 |
| min_samples_leaf | 1 | 1 |

**Hasil:** Test Accuracy tetap **98.34%** → improvement = +0.00%

> Tuning membuktikan RF sudah di titik optimal (ceiling effect).
> Tulis: *"Variasi konfigurasi tidak memberikan peningkatan signifikan, membuktikan performa model sudah stabil pada dataset ini."*

---

## 4.5 Analisis Fitur TF-IDF

**Grafik: `hasil_phase5/1_visualisasi_utama/6_tfidf_feature_importance_LR_RF.png`**

- Panel kiri (LR): fitur dengan koefisien tinggi → cenderung AI (kata formal, baku)
- Panel tengah (LR): fitur koefisien negatif → cenderung Manusia (slang, @, nama media)
- Panel kanan (RF): feature importance RF → kata-kata paling membedakan

> Tulis di Bab 4: *"Analisis fitur TF-IDF menunjukkan bahwa model membedakan teks AI dan Manusia terutama berdasarkan kosakata formal vs. informal, serta penanda sumber data seperti nama media dan username di media sosial."*

---

## 4.6 Validasi Eksternal (External Validation)

Diuji pada 30 teks baru yang tidak ada di dataset training.

| Metrik | Nilai |
|--------|-------|
| Accuracy | **73.33%** |
| Precision | 70.59% |
| Recall | 80.00% |
| F1-Score | 75.00% |
| FP (Manusia dikira AI) | 5 |
| FN (AI lolos) | **3** ← ada yang lolos! |

> Penurunan dari 99.67% → 73.33% = bukti domain gap nyata.
> FN=3 menunjukkan model belum sempurna di luar distribusi training.

---

## 4.7 Uji Inferensi Langsung (3 Teks Uji Nyata)

| Tes | Jenis Teks | Label Benar | Prediksi IndoBERT | Status |
|-----|-----------|-------------|-------------------|--------|
| 1 | Formal + kata baku | AI | AI (95.9%) | ✅ |
| 2 | Gaul/informal | MANUSIA | AI (85.6%) | ❌ |
| 3 | Manusia nulis formal | MANUSIA | AI (88.0%) | ❌ |

> Bukti: model terlalu mengasosiasikan gaya formal = AI.
> Ini sesuai temuan External Validation dan Group CV.

---

## 4.8 Pembahasan — Kenapa Hasilnya Tinggi?

- **Bukan data leakage** → Pipeline + overlap=0 sudah diverifikasi
- **Domain gap** → dataset AI formal vs dataset Manusia ada penanda sumber khas
- **Ceiling effect di fitur** → TF-IDF sangat efektif tangkap pola leksikal dominan
- **Konfirmasi dari Group CV** → saat sumber dipisah, akurasi turun drastis

---

## Peta Grafik untuk Skripsi

| Grafik | Lokasi File | Masuk Sub-bab |
|--------|-------------|--------------|
| 4 Confusion Matrix | `1_visualisasi_utama/1_all_confusion_matrices.png` | 4.2 |
| ROC Curve 3 model | `1_visualisasi_utama/2_roc_curve_3models.png` | 4.2 |
| Bar Chart 4 model | `1_visualisasi_utama/3_model_comparison_all.png` | 4.2 |
| CV per Fold | `1_visualisasi_utama/4_cv_scores_all_models.png` | 4.3 |
| Group CV vs Standard | `1_visualisasi_utama/5_group_cv_vs_standard.png` | 4.3 |
| TF-IDF Features | `1_visualisasi_utama/6_tfidf_feature_importance_LR_RF.png` | 4.5 |
| Distribusi Label | `2_eda_dataset/1_distribusi_label.png` | 4.1 |
| Boxplot Teks | `2_eda_dataset/4_boxplot_panjang_teks.png` | 4.1 |
| Wordcloud AI | `2_eda_dataset/5_wordcloud_ai.png` | 4.1 |
| Wordcloud Manusia | `2_eda_dataset/6_wordcloud_manusia.png` | 4.1 |
| Heatmap Metrik | `3_perbandingan_model/1_heatmap_metrics_lengkap.png` | 4.2/4.3 |
| Radar Chart | `3_perbandingan_model/2_radar_chart_4models.png` | 4.2 |
| FP/FN Comparison | `3_perbandingan_model/3_kesalahan_klasifikasi.png` | 4.2 |
