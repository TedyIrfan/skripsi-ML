# CATATAN BAB 5 — Kesimpulan dan Saran
# Status: FINAL — semua training & evaluasi selesai

---

## 5.1 Kesimpulan

### 5.1.1 Jawaban Rumusan Masalah

1. **Keempat algoritma** (LR, SVM, RF, IndoBERT) berhasil mengklasifikasi teks AI vs
   Manusia Bahasa Indonesia dengan akurasi tinggi pada data internal.

2. **Model terbaik: Logistic Regression & SVM** — seri di 99.67% accuracy, FNR 0.00%.

3. **IndoBERT lebih rendah dari TF-IDF+ML** (97.35% vs 99.67%) karena:
   - Dataset kecil (1087 training) vs 124 juta parameter
   - Pola leksikal dominan → TF-IDF sangat efektif

4. **SVM paling robust** antar sumber data (Group CV hanya turun 3%),
   dibandingkan LR (-12%) dan RF (-17%).

5. **FNR = 0.00% pada semua model** (test set internal) — tidak ada teks AI yang
   lolos. Ini memenuhi tujuan utama penelitian.

### 5.1.2 Tabel Rangkuman Final

| Model | Test Acc | Group CV | External Val | FNR | Ranking |
|-------|---------|---------|-------------|-----|---------|
| LR | 99.67% | 87.29% | 73.33% | 0% | 1 (internal) |
| SVM | 99.67% | **96.74%** | 73.33% | 0% | **1 (robust)** |
| RF | 98.34% | 81.49% | 73.33% | 0% | 3 |
| IndoBERT | 97.35% | - | - | 0% | 4 |

> **Rekomendasi model untuk digunakan:** SVM — karena paling robust di Group CV (96.74%)
> walau akurasi internal sama dengan LR.

---

## 5.2 Keterbatasan Penelitian

| # | Keterbatasan | Bukti Empiris |
|---|-------------|---------------|
| 1 | **Domain gap** antar sumber data | Group CV turun 12-17% saat sumber dipisah |
| 2 | **Generalisasi terbatas** ke data dunia nyata | External Validation: 99.67% → 73.33% |
| 3 | Dataset AI didominasi gaya formal | Wordcloud & analisis fitur TF-IDF |
| 4 | Dataset Manusia punya penanda sumber khas | @mention, nama media terdeteksi sebagai fitur |
| 5 | Dataset relatif kecil | 1510 teks; IndoBERT butuh jauh lebih banyak |
| 6 | IndoBERT tidak bisa diuji GPU | Driver RTX 3050 tidak kompatibel PyTorch 2.11+cu126 |
| 7 | Model terlalu mengasosiasikan gaya formal = AI | Inferensi: manusia nulis formal dikira AI 88% |

---

## 5.3 Saran Penelitian Selanjutnya

1. **Perluas variasi dataset** — tambah teks Manusia yang lebih netral
   (bukan hanya Twitter/berita), dan teks AI yang lebih informal
2. **Tambah sumber AI baru** — Claude, Gemini, Llama, Mistral
   (model kita hanya dilatih pada output ChatGPT-like)
3. **Fine-tune IndoBERT dengan lebih banyak data** — minimal 5000 teks
   agar 124 juta parameter bisa teroptimalkan
4. **Gunakan Group CV sebagai metrik utama** di penelitian berikutnya
   untuk hasil yang lebih realistis
5. **Threshold tuning** pada External Validation untuk kurangi FN
6. **Uji dengan teks adversarial** — teks AI yang diedit manual agar mirip manusia

---

## Catatan Bab 3 yang Masuk Bab 5

Yang perlu disebutkan di Bab 5 tapi metodologinya di Bab 3:
- Group CV → metode di Bab 3, hasilnya di Bab 4, implikasinya di Bab 5
- External Validation → sama
- Hyperparameter Tuning → disebutkan sebagai bukti model sudah optimal
