# CATATAN BAB 3 — Letak Materi & Status

## Peta Letak Materi

| Sub-bab | Isi yang Masuk | Status Kita |
|---------|---------------|-------------|
| **3.1 Tahapan Penelitian** | Flowchart alur: pengumpulan data → preprocessing TF-IDF → split 80/20 → training → evaluasi CV → analisis hasil | ✅ Alur sudah jelas |
| **3.2 Instrumen Penelitian** | Tabel tools: Python, scikit-learn, PyTorch, HuggingFace Transformers, pandas, matplotlib | ✅ Library sudah dipakai |
| **3.3 Pengumpulan Data, Populasi, Sample** | - Total: 1510 teks (760 AI + 750 Manusia) dari sumber yang tercatat di README | ✅ Dataset final selesai |
| | - Split: 80% train (1208) / 20% test (302), stratify, random_state=42 | ✅ |
| | - Verifikasi overlap = 0 | ✅ |
| **3.4 Metode Analisis Data** | ← **RUMUS EVALUASI + METODE TUNING ADA DI SINI** (lihat di bawah) | ✅ Rumus bisa ditulis |
| | - GridSearchCV 5-Fold untuk optimasi RF (81 kombinasi) | ✅ Sudah dijalankan |

---

## Rumus yang Masuk 3.4

Semua diturunkan dari TP, TN, FP, FN:

- **Accuracy** = (TP+TN) / (TP+TN+FP+FN)
- **Precision** = TP / (TP+FP)
- **Recall** = TP / (TP+FN)
- **F1-Score** = 2×P×R / (P+R)
- **FNR** = FN / (FN+TP) ← metrik utama penelitian ini
- **AUC-ROC** = area under ROC curve

Juga tulis TF-IDF formula dan rumus masing-masing algoritma (sigmoid LR, hyperplane SVM, voting RF, softmax IndoBERT) di 3.4.

---

## Status Training Saat Ini

| Algoritma | Status | Keterangan |
|-----------|--------|-----------|
| Logistic Regression | ✅ Selesai | Test 99.67%, FNR 0% |
| SVM (RBF Kernel) | ✅ Selesai | Test 99.67%, FNR 0% |
| Random Forest | ✅ Selesai | Test 98.34%, FNR 0% |
| IndoBERT | ✅ Selesai | Test 97.35%, FNR 0%, ~21 menit CPU |
| RF Hyperparameter Tuning | ✅ Selesai | Best CV 98.84%, Test tetap 98.34% (ceiling effect) |

> **PHASE 4 TRAINING SELESAI 100%**

---

## Hasil Validasi yang Sudah Dijalankan (Verifikasi Keabsahan)

### Validasi 1 — 10-Fold Stratified CV (metode utama, sudah BENAR)
- Pakai **Pipeline** → TF-IDF di-fit ulang tiap fold → tidak ada data leakage ✅
- Overlap train-test = 0 teks ✅
- random_state=42 di semua script → hasil reproducible ✅
- Script: `2_training/train_strict_cv.py`

### Validasi 2 — Group CV (verifikasi tambahan, sudah dijalankan)
Tujuan: cek apakah hasil bagus karena model genuine atau karena sumber data sama di train & test

| Model | Group CV | Standard CV | Selisih |
|-------|----------|-------------|---------|
| SVM | 96.74% ±3.43% | 99.74% | -3.00% |
| LR | 87.29% ±15.11% | 99.74% | -12.44% |
| RF | 81.49% ±20.62% | 98.61% | -17.12% |

Kesimpulan: ada **domain gap** antar sumber → tulis di keterbatasan penelitian Bab 4/5
Script: `2_training/train_group_cv.py`

### Validasi 3 — External Validation (verifikasi tambahan, sudah dijalankan)
Tujuan: uji generalisasi model ke teks baru yang tidak ada di dataset training sama sekali

| Metrik | Nilai |
|--------|-------|
| Accuracy | 73.33% |
| Precision | 70.59% |
| Recall | 80.00% |
| F1-Score | 75.00% |
| Error | 8 dari 30 teks (5 FP + 3 FN) |

Kesimpulan: model bergantung pada pola sumber data → perlu disebutkan sebagai keterbatasan
Script: `3_evaluasi/external_validation.py` | Data: `external_validation.csv` (30 teks baru)

---

## Catatan Penting

- Angka hasil (99.67%, confusion matrix) → **BAB 4**, bukan Bab 3
- Bab 3 hanya rumus dan metodologi
- Ketiga validasi di atas → metodenya masuk **3.4**, angka hasilnya masuk **BAB 4**
- Validasi 2 & 3 jadi bahan **keterbatasan penelitian** di Bab 4 atau 5
