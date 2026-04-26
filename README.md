# Komparasi Algoritma Machine Learning untuk Deteksi Teks Tulisan Manusia dan Kecerdasan Buatan Berbahasa Indonesia

**Skripsi S1 Teknik Informatika**

---

## Tahapan Penelitian

Penelitian ini dilaksanakan dalam 8 tahapan utama:

### 1. Studi Literatur
Tahap pengumpulan dan kajian referensi ilmiah yang menjadi landasan penelitian, meliputi:
- Jurnal nasional dan internasional terkait deteksi teks AI
- Buku referensi machine learning dan NLP
- Penelitian terdahulu yang relevan
- Dokumentasi library dan framework yang digunakan

### 2. Pengumpulan Dataset

Tahap pengumpulan data teks untuk kedua kelas label, yaitu teks tulisan manusia dan teks kecerdasan buatan, dengan total 1.500 teks (750 AI + 750 Manusia).

Teks tulisan manusia dikumpulkan dari kumpulan teks berita, opini, dan percakapan berbahasa Indonesia yang diperoleh melalui berbagai platform publik meliputi media berita daring, Reddit, Twitter, dan YouTube, yang setelah melalui proses pembersihan menghasilkan 543 teks. Untuk melengkapi kekurangan data, ditambahkan 207 teks dari dataset Gojek App Reviews Bahasa Indonesia yang tersedia secara publik di Kaggle.

Sementara itu, data teks kecerdasan buatan dikumpulkan melalui dua pendekatan: pertama, menggunakan prompt formal dan kasual kepada berbagai model bahasa besar melalui platform OpenRouter, Groq, dan HuggingFace untuk menghasilkan 374 teks; kedua, menggunakan prompt jailbreak khusus yang menginstruksikan model untuk menghasilkan 376 teks bergaya slang internet Indonesia dengan singkatan dan typo natural guna mensimulasikan cara penulisan informal di media sosial.

### 3. Preprocessing & Cleaning Data

Teks yang terkumpul melalui proses pembersihan (text cleaning), pelabelan kelas, dan ekstraksi fitur menggunakan TF-IDF. Tahapan ini menghasilkan dataset final sebesar 1.500 teks yang terdiri dari 750 teks manusia dan 750 teks AI.

- Hapus teks pengantar atau atribusi sumber (judul, URL awal) — `[0_dataset/clean_dataset.py]`
- Hapus URL (`https?://...`, `www...`), mention (`@user`), dan hashtag (`#tag`) — `[0_dataset/clean_dataset.py]`
- Hapus tag HTML dan emoji (karakter non-ASCII) — `[0_dataset/clean_dataset.py]`
- Pertahankan tanda baca dasar (hanya simpan huruf, angka, `. , ? ! " ' -`, hapus yang tidak umum) — `[0_dataset/clean_dataset.py]`
- Rapikan spasi dan newline berlebih menjadi satu spasi — `[0_dataset/clean_dataset.py]`
- Filter noise dengan menghapus teks yang memiliki kurang dari 15 kata — `[0_dataset/clean_dataset.py]`
- Gabungkan kedua kelas label dan lakukan shuffle acak — `[1_data/build_dataset.py]`
- Ekstraksi fitur menggunakan TF-IDF (max_features=5000, ngram_range=(1,2), min_df=2, max_df=0.8, tanpa lowercasing) — `[2_training/train_strict_cv.py]`


### 4. Pembagian Data

Dataset dibagi dengan rasio 80% data latih (1.200 teks) dan 20% data uji (300 teks) menggunakan stratified sampling dengan `random_state=42` untuk memastikan distribusi kelas yang proporsional dan hasil yang dapat direproduksi.

- Load dataset final (`dataset_clean_1500.csv`) — `[2_training/train_strict_cv.py]`
- Mapping label ke numerik (MANUSIA=0, AI=1) — `[2_training/train_strict_cv.py]`
- Split data menggunakan `train_test_split` (test_size=0.2, stratify=y, random_state=42) — `[2_training/train_strict_cv.py]`
- Verifikasi tidak ada overlap antara data latih dan data uji — `[2_training/train_strict_cv.py]`

### 5. Training 4 Model

Empat algoritma machine learning dilatih pada data latih, yaitu Logistic Regression, Support Vector Machine (kernel RBF), Random Forest, dan IndoBERT. Setiap model dilatih menggunakan pipeline yang konsisten untuk mencegah kebocoran data (data leakage).

- Definisasi pipeline TF-IDF + Classifier untuk setiap model — `[2_training/train_strict_cv.py]`
- TF-IDF di-fit hanya pada data latih setiap fold, data uji hanya di-transform — `[2_training/train_strict_cv.py]`
- Training 3 model ML klasik (LR, SVM, RF) dengan 10-Fold Stratified Cross-Validation — `[2_training/train_strict_cv.py]`
- Training IndoBERT (`indobenchmark/indobert-base-p1`) dengan 3 epoch, batch size 8, max length 128 — `[2_training/train_indobert.py]`
- Hyperparameter tuning Random Forest menggunakan GridSearchCV 5-Fold (81 kombinasi) — `[2_training/hyperparameter_tuning.py]`
- Simpan model terbaik dalam format `.pkl` dan hasil evaluasi ke `.json` — `[2_training/train_strict_cv.py]`

### 6. Optimasi Hyperparameter

Optimasi dilakukan pada model Random Forest menggunakan GridSearchCV dengan validasi silang 5-fold terhadap 81 kombinasi parameter. Tahapan ini bertujuan menemukan konfigurasi optimal yang menghasilkan performa terbaik.

- Definisikan parameter grid (n_estimators, max_depth, min_samples_split, min_samples_leaf) — `[2_training/hyperparameter_tuning.py]`
- Jalankan GridSearchCV 5-Fold pada data latih — `[2_training/hyperparameter_tuning.py]`
- Evaluasi 81 kombinasi parameter dan pilih konfigurasi terbaik — `[2_training/hyperparameter_tuning.py]`
- Bandingkan performa baseline vs tuned model pada test set — `[2_training/hyperparameter_tuning.py]`

### 7. Evaluasi Model

Evaluasi dilakukan menggunakan tiga metode: (1) 10-Fold Stratified Cross-Validation sebagai metode utama, (2) Group Cross-Validation untuk menguji ketahanan model terhadap perbedaan sumber data, dan (3) External Validation menggunakan 250 teks baru yang tidak ada dalam dataset pelatihan.

- 10-Fold Stratified Cross-Validation dengan Pipeline (anti-leakage) — `[2_training/train_strict_cv.py]`
- Group Cross-Validation berdasarkan sumber data (GroupKFold 5-Fold) — `[2_training/train_group_cv.py]`
- External Validation pada dataset baru (250 teks OOD) — `[3_evaluasi/external_validation.py]`
- Adversarial Testing pada teks AI bergaya slang/typo — `[3_evaluasi/external_validation.py]`

### 8. Analisis Hasil & Kesimpulan

Hasil evaluasi keempat algoritma dibandingkan berdasarkan metrik yang ditetapkan. Analisis mencakup identifikasi algoritma terbaik, pembahasan keterbatasan penelitian, dan rekomendasi untuk penelitian selanjutnya.

- Perbandingan performa 4 model (Accuracy, Precision, Recall, F1, FNR, AUC-ROC) — `[3_evaluasi/compare_models.py]`
- Analisis error (FP, FN) dan confidence model — `[3_evaluasi/error_analysis.py]`
- Analisis teks ambigu dan distribusi probabilitas — `[3_evaluasi/ambiguous_analysis.py]`
- Analisis linguistik AI vs Manusia — `[3_evaluasi/linguistic_analysis.py]`
- Ablation Study untuk kontribusi fitur TF-IDF — `[3_evaluasi/ablation_study.py]`
- Threshold tuning untuk optimasi trade-off F1 vs FNR — `[3_evaluasi/threshold_tuning.py]`

---

## Hasil Training & Evaluasi

### Output Training

#### `models_strict/` — 3 Model ML Klasik
- 3 file `.pkl` — Model terlatih siap prediksi (LR, SVM, RF)
- `strict_cv_results.json` — Hasil 10-Fold CV & Test Set (Accuracy, Precision, Recall, F1, FNR, AUC)
- `hyperparameter_tuning_results.json` — Hasil GridSearchCV RF (81 kombinasi, best params)
- `threshold_analysis.json` — Analisis threshold confidence

#### `models_indobert/` — IndoBERT
- `final_model/` — Model terlatih (PyTorch + Transformers)
- `checkpoint-*/` — Checkpoint training (epoch 1, 2, 3)
- `indobert_results.json` — Hasil Test Set & training log

#### `models_group_cv/` — Group Cross-Validation
- `group_cv_results.json` — Hasil Group CV 3 model klasik (domain gap per sumber)
- `indobert_group_cv_results.json` — Hasil Group CV IndoBERT

### Output Evaluasi

#### `hasil_phase5/1_visualisasi_utama/` — 6 Grafik Utama
- Confusion Matrix (4 model)
- ROC Curve (3 model klasik)
- Perbandingan Model (Bar Chart)
- CV Scores per Fold
- Group CV vs Standard CV
- TF-IDF Feature Importance (LR & RF)

#### `hasil_phase5/2_eda_dataset/` — 5 Grafik EDA
- Distribusi label
- Distribusi panjang teks & jumlah kata
- Boxplot panjang teks
- Wordcloud AI
- Wordcloud Manusia

#### `hasil_phase5/3_perbandingan_model/` — 3 Grafik
- Heatmap metrik lengkap
- Radar chart 4 model
- Kesalahan klasifikasi (FP vs FN)

#### `hasil_phase5/4_linguistic_analysis/` — 2 Grafik
- Perbandingan linguistik 4 panel
- Statistik leksikal

#### `hasil_phase5/4_error_analysis/` — 3 Grafik
- Error count per model
- Panjang teks: benar vs salah
- Confidence false positives

#### `hasil_phase5/5_ambiguous_analysis/` — 4 Grafik
- Distribusi probabilitas + zona ambigu
- Confidence benar vs salah
- Probabilitas per label
- Perbandingan ambigu 3 model

#### `hasil_phase5/6_ablation_study/` — 3 Grafik
- Feature importance top 20
- Akurasi 9 skenario ablation
- Heatmap drop akurasi

#### `hasil_phase5/7_adversarial_test/` — 2 Grafik
- Adversarial vs baseline
- Confusion matrix adversarial

#### `hasil_phase5/8_indobert_inference/` — 1 Grafik
- Inferensi 3 teks edge case

#### `hasil_phase5/9_per_category_analysis/` — 3 Grafik
- Akurasi & confidence per label
- Distribusi panjang teks
- Akurasi per label 3 model

#### `hasil_phase5/10_threshold_tuning/` — 3 Grafik
- Threshold metrics comparison
- FNR analysis
- ROC Curve

---

## Algoritma yang Digunakan

| No | Algoritma | Tipe |
|----|-----------|------|
| 1 | Logistic Regression | ML Klasik |
| 2 | SVM (RBF Kernel) | ML Klasik |
| 3 | Random Forest | ML Klasik |
| 4 | IndoBERT | Deep Learning |

---

## Struktur Folder

```
📁 0_dataset/              → Data mentah & script generate AI
📁 1_data/                 → Script pembangunan dataset
📁 2_training/             → Script training model
📁 3_evaluasi/             → Script evaluasi & analisis
📁 4_tools/                → Tools inference
```
