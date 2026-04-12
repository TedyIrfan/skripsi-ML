# External Validation - Quick Guide

## 📋 Workflow Summary

Saya sudah generate 50 AI text prompts untuk Anda. Sekarang tinggal:

1. **Generate AI texts** (Anda)
2. **Collect human texts dari Kaggle** (Anda)
3. **Merge & validate** (Auto)

---

## Step 1: Generate AI Texts (30-60 menit)

### File yang sudah dibuat:
- `ai_generation_prompts.txt` - 50 prompts siap pakai

### Cara pakai:

1. **Buka file**: `ai_generation_prompts.txt`

2. **Copy paste ke ChatGPT/Gemini**:
   - Buka ChatGPT atau Gemini
   - Copy PROMPT #1
   - Paste ke chat
   - Copy response
   - Paste di bawah `[PASTE RESPONSE HERE]`
   - Ulangi untuk semua 50 prompts

3. **Tips**:
   - Bisa pakai ChatGPT untuk 25 prompts pertama
   - Gemini untuk 25 prompts terakhir
   - Atau mix sesuka Anda
   - **PENTING**: Jangan skip yang informal! Ini penting untuk test model

4. **Setelah selesai**:
   ```bash
   python compile_ai_texts.py
   ```
   Ini akan create: `external_validation_ai_only.csv`

---

## Step 2: Collect Human Texts dari Kaggle (30-60 menit)

### Dataset Kaggle yang bagus:

1. **Indonesian News Dataset**
   - Search: "indonesian news" di Kaggle
   - Ambil 15-20 artikel pendek

2. **Indonesian Social Media**
   - Search: "indonesian twitter" atau "indonesian reddit"
   - Ambil 15-20 posts

3. **Indonesian Reviews/Comments**
   - Search: "indonesian reviews"
   - Ambil 10-15 reviews

### Format CSV yang dibutuhkan:

Create file: `external_validation_human_only.csv`

```csv
text,label,source
"Ini adalah text manusia dari kaggle...",MANUSIA,kaggle_news
"Text kedua dari twitter...",MANUSIA,kaggle_twitter
...
```

### Tips:
- Total target: 50 human texts
- Mix dari berbagai sumber
- Panjang: 50-300 words
- Pastikan belum ada di dataset lama

---

## Step 3: Merge & Validate (5 menit)

Setelah punya kedua file:
- `external_validation_ai_only.csv` (50 AI texts)
- `external_validation_human_only.csv` (50 human texts)

Jalankan:
```bash
python merge_external_validation.py
```

Ini akan:
- Merge AI + Human texts
- Shuffle data
- Check overlap dengan training data
- Create: `external_validation.csv`

---

## Step 4: Run Validation (5 menit)

```bash
python external_validation.py
```

Ini akan:
- Load model yang sudah trained
- Test pada external data
- Calculate metrics
- Generate report
- Save results ke JSON

---

## 📊 Expected Results

| Accuracy | Interpretation |
|----------|----------------|
| 90-95% | ✓ Excellent - Model sangat robust |
| 85-90% | ✓ Good - Expected result |
| 80-85% | ⚠️ Acceptable - Ada room for improvement |
| <80% | ⚠️ Concerning - Mungkin overfit |

---

## 📁 Files Created

Setelah semua selesai, Anda akan punya:

1. `ai_generation_prompts.txt` - Prompts + responses
2. `external_validation_ai_only.csv` - 50 AI texts
3. `external_validation_human_only.csv` - 50 human texts (Anda buat)
4. `external_validation.csv` - Combined dataset
5. `external_validation_results.json` - Hasil testing
6. `external_validation_detailed.csv` - Detailed predictions

---

## ⏱️ Timeline

| Task | Time | Status |
|------|------|--------|
| Generate prompts | ✅ DONE | 0 min |
| Generate AI texts | 🔄 TODO | 30-60 min |
| Collect human texts | 🔄 TODO | 30-60 min |
| Merge datasets | ⏳ READY | 5 min |
| Run validation | ⏳ READY | 5 min |
| **TOTAL** | | **1-2 jam** |

---

## 🚀 Quick Start

**Mulai sekarang**:

1. Buka: `ai_generation_prompts.txt`
2. Copy PROMPT #1 ke ChatGPT
3. Paste response
4. Repeat untuk semua 50 prompts
5. Sambil itu, cari dataset Kaggle untuk human texts

**Parallel work**:
- Generate AI texts sambil download Kaggle dataset
- Bisa selesai dalam 1-2 jam total!

---

## ❓ Questions?

Kalau ada masalah atau butuh bantuan, tinggal bilang!

Siap mulai? 🚀
