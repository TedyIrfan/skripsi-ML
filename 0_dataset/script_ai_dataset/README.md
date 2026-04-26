# Script AI Dataset Generator

Folder ini berisi semua script yang digunakan untuk menghasilkan 750 data teks AI untuk dataset skripsi.

## Sumber Data AI

Data AI 750 dihasilkan dari berbagai model LLM melalui beberapa API:

| No | Script Utama | API/Platform | Model | Output |
|----|-------------|--------------|-------|--------|
| 1 | `generate_ai_openrouter.py` | OpenRouter | Hermes-3-Llama-3.1-405B | ~100-400 teks |
| 2 | `generate_groq.py` | Groq | Llama-3.3-70B | ~100-200 teks |
| 3 | `generate_hf_qwen.py` | HuggingFace | Kimi-K2.5 | ~100-200 teks |
| 4 | `generate_ai_slang.py` | OpenRouter | Gemma-2-27b-it | ~100 teks (slang) |
| 5 | `generate_ai_groq_100_long.py` | Groq | Llama-3.3-70B | ~100 teks |
| 6 | `generate_groq_gpt_oss.py` | Groq | GPT-OSS | ~80 teks |
| 7 | `add_ai_gpt_manual.py` | Manual (copy-paste) | ChatGPT | ~10-50 teks |
| 8 | `add_gemini_gpt_batch.py` | Manual/Batch | Gemini/GPT | batch |
| 9 | `add_gpt_batch2.py`, `add_gpt_batch4.py` | Manual/Batch | GPT | batch |

## Cara Pakai

### 1. Generate via API

Pastikan kamu punya API key, lalu jalankan:

```bash
python generate_ai_openrouter.py   # OpenRouter (free models)
python generate_groq.py            # Groq (Llama 3.3 70B)
python generate_hf_qwen.py         # HuggingFace (Kimi K2.5)
python generate_ai_slang.py        # OpenRouter (Gemma slang)
```

### 2. Tambah Manual (Copy-Paste dari ChatGPT)

```bash
python add_ai_gpt_manual.py
```

Script ini akan meminta kamu paste teks satu per satu, lalu otomatis menghitung statistik (kata, karakter, kalimat).

## Struktur Output

Setiap script menghasilkan CSV dengan kolom:
- `text` — Teks yang di-generate
- `label` — Selalu "AI"
- `source` — Sumber API (OpenRouter, Groq, HuggingFace, dll)
- `model` — Nama model yang dipakai
- `style` — Casual / Formal / Slang
- `topic` — Topik teks (Sosial, Budaya, Politik, dll)
- `batch` — Nomor batch
- `generated_at` — Timestamp

## Alur Data

```
Semua script generate di atas
        ↓
  Banyak file CSV kecil (data_ai_openrouter_96.csv, data_ai_groq_100.csv, dll)
        ↓
  Digabung + dibersihkan (filter, deduplikasi)
        ↓
  data_ai_all_clean.csv (di folder 0_dataset/)
        ↓
  Diambil 750 data AI untuk dataset final
```

## File Tambahan

### Test Scripts (untuk testing koneksi API)
- `test_openrouter.py` — Test OpenRouter API
- `test_groq.py`, `test_groq_quality.py` — Test Groq API
- `test_huggingface.py`, `test_hf_direct.py`, `test_hf_speed.py` — Test HuggingFace
- `test_qwen.py`, `test_qwen3_hf.py`, `test_qwen3_indo.py` — Test Qwen
- `test_kimi.py` — Test Kimi API
- `debug_openrouter.py`, `debug_qwen.py` — Debug script

### Filter Scripts (untuk filtering data dari sumber lain)
- `filter_ai_groq_100.py` — Filter data Groq
- `filter_indosum_or.py` — Filter IndoSum dari OpenRouter
- `filter_kaggle_twitter_or.py` — Filter Kaggle/Twitter
- `filter_reddit_auto.py`, `filter_reddit_max250.py` — Filter Reddit
- `filter_terrorism_max82.py` — Filter topik terorisme

### Data Mentah (hasil generate, sudah dipindahkan ke archive/)
- `data_ai_gpt_10.py` — Data GPT manual (10 teks)
