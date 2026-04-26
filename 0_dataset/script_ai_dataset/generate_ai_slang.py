"""
Script: Generate Dataset AI Slang & Formal via Google AI Studio
============================================================================
Tujuan: Men-generate 100 teks AI ganda dengan porsi 70 Slang dan 30 Formal.
Output: dataset_ai_slang.csv (mode APPEND / ditambah ke file yang sudah ada)
"""

import os
import csv
import time
import re
import requests

# ─────────────────────────────────────────────
# KONFIGURASI (Aman & Stabil)
# ─────────────────────────────────────────────
API_KEY        = "YOUR_OPENROUTER_API_KEY_HERE"
MODEL          = "google/gemma-2-27b-it"        
API_URL        = "https://openrouter.ai/api/v1/chat/completions"
OUTPUT_CSV     = "dataset_ai_slang.csv"

# Target per generasi
TARGET_SLANG   = 50  
TARGET_FORMAL  = 0  
BATCH_SIZE     = 5     
DELAY_SECONDS  = 15     

# ─────────────────────────────────────────────
# SYSTEM & USER PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """Kamu adalah generator teks yang mensimulasikan 
curhatan atau obrolan kasual di media sosial Indonesia 
(X/Twitter, WhatsApp, TikTok, dll).
ATURAN WAJIB (SANGAT KETAT):
1. Gunakan slang & singkatan internet Indonesia: udh, pdhl, bgt, 
   kyknya, anjir, wkwk, lu, gue, emg, tp, krn, yg, dll.
2. Sertakan minimal 1-2 typo natural per teks: nungggu, mksih, 
   ekspetasi, bales, tggu, dll.
3. Huruf: SEMUA kecil atau kapital berantakan, BUKAN kalimat formal.
4. Format bebas layaknya orang ngetik cepat di HP (2-4 kalimat pendek).
5. DILARANG KERAS menggunakan bahasa formal, rapi, atau logis."""

# Prompt Slang 
PROMPT_SLANG = """Buatkan {batch_size} teks berbeda dalam Bahasa Indonesia.

Teks ini harus mensimulasikan berbagai obrolan atau curhatan kasual di media sosial (X/Twitter, WhatsApp, TikTok) atau forum komunitas. Topiknya buat sangat acak dan bervariasi: bisa tentang kehidupan kampus, drama pekerjaan, keluhan soal cuaca, review makanan, telat bangun, ribut dengan teman, hingga masalah keuangan (tanggal tua).

Aturan Gaya Bahasa (SANGAT KETAT):
1. Gunakan Slang & Singkatan: Wajib menggunakan bahasa gaul dan singkatan internet Indonesia (contoh: udh, pdhl, bgt, kyknya, dospem, kating, anjir, wkwk, lu, gue, fyp).
2. Sertakan Typo Natural: Sengaja buat 1-2 kesalahan ketik (typo) di setiap teks yang sering dilakukan manusia (contoh: nungggu, mksih, ekspetasi, bales).
3. Format Berantakan: Jangan gunakan huruf kapital di awal kalimat dengan benar. Biarkan semua huruf kecil atau kapital tidak beraturan.
4. Anti-AI: JANGAN PERNAH menggunakan gaya bahasa kaku, rapi, atau baku. Dilarang keras memakai frasa khas AI seperti "kesimpulannya", "penting untuk diingat", atau "secara keseluruhan"."""

# Prompt Formal
PROMPT_FORMAL = """Buatkan {batch_size} teks berbeda dalam Bahasa Indonesia dengan spesifikasi berikut:

GAYA: FORMAL
TOPIK: Bervariasi dari list ini [Sosial, Budaya, Politik, Ekonomi, Teknologi, Kesehatan, Olahraga, Pendidikan, Lingkungan, Entertainment]
PANJANG: 150-400 kata per teks
FORMAT: Paragraf natural, bukan list atau poin-poin

Tulis dengan gaya formal dan informatif. Gunakan:
- Bahasa baku dan resmi
- Kata ganti "Anda", "kita"
- Struktur kalimat lengkap
- Tone profesional dan objektif
- Berbasis fakta dan analisis"""

def build_user_prompt(style: str, batch_size: int) -> str:
    # Memilih prompt berdasarkan gaya
    base = PROMPT_SLANG if style == "SLANG" else PROMPT_FORMAL
    
    # Memaksa AI menggunakan format TEKS_N agar script Python tidak error
    return f"""{base.format(batch_size=batch_size)}

Format output yang WAJIB ada di dalam tag OUTPUT (Jangan buat format CSV, cukup seperti ini):
<OUTPUT>
TEKS_1: [isi teks pertama]
TEKS_2: [isi teks kedua]
...
TEKS_{batch_size}: [isi teks ke-{batch_size}]
</OUTPUT>"""

# ─────────────────────────────────────────────
# PARSER: Ekstrak teks dari respons API
# ─────────────────────────────────────────────
def parse_response(text: str, expected: int) -> list[str]:
    results = []
    output_block = re.search(r'<OUTPUT>(.*?)</OUTPUT>', text, re.DOTALL | re.IGNORECASE)
    
    if output_block:
        target_text = output_block.group(1)
    else:
        target_text = text

    pattern = re.compile(r'TEKS_\d+\s*:\s*(.+?)(?=TEKS_\d+\s*:|$)', re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(target_text)
    
    for m in matches:
        cleaned = m.strip().replace('\n', ' ')
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if len(cleaned.split()) >= 5:  
            results.append(cleaned)
            
    return results[:expected]

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 65)
    print(f"  Generate Dataset AI via Google AI Studio")
    print("=" * 65)
    print(f"  Model      : {MODEL}")
    print(f"  Target     : {TARGET_SLANG} Slang + {TARGET_FORMAL} Formal")
    print(f"  Batch size : {BATCH_SIZE} teks/request")
    print(f"  Output     : {OUTPUT_CSV} (APPEND)")
    print("=" * 65)

    def run_generation(target, style_name):
        all_texts = []
        batch_num = 0
        fail_count = 0
        MAX_FAIL = 5

        print(f"\n--- Memulai Fase [{style_name}] Target: {target} ---")
        while len(all_texts) < target and fail_count < MAX_FAIL:
            batch_num += 1
            remaining   = target - len(all_texts)
            this_batch  = min(BATCH_SIZE, remaining)

            print(f"[{style_name} Batch {batch_num}] Generating {this_batch} teks... "
                  f"(sudah: {len(all_texts)}/{target})")

            try:
                headers = {
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": build_user_prompt(style_name, this_batch)},
                    ],
                    "temperature": 1.0,
                    "max_tokens": 4096,
                }

                response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
                
                if response.status_code != 200:
                    print(f"  [ERROR] API Error: {response.status_code} - {response.text}")
                    fail_count += 1
                    time.sleep(DELAY_SECONDS * 2)
                    continue
                    
                raw_text = response.json()["choices"][0]["message"]["content"]
                parsed   = parse_response(raw_text, this_batch)

                if not parsed:
                    print(f"  [WARNING] Parsing gagal. Model tidak mengembalikan teks sesuai format.")
                    fail_count += 1
                    time.sleep(DELAY_SECONDS)
                    continue

                # Simpan ke CSV, semua akan berlabel 'AI' sesuai instruksimu
                with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    for teks in parsed:
                        writer.writerow([teks, 'AI'])

                all_texts.extend(parsed)
                fail_count = 0
                print(f"  [OK] {len(parsed)} teks berhasil disimpan.")
                print(f"  Preview: {parsed[0][:80]}...")

            except Exception as e:
                print(f"  [ERROR] {str(e)}")
                fail_count += 1
                time.sleep(DELAY_SECONDS * 3)

            if len(all_texts) < target:
                time.sleep(DELAY_SECONDS)
                
        return len(all_texts)

    # Cek file & Tulis Header (text, label) jika file belum ada
    file_exists = os.path.isfile(OUTPUT_CSV)
    with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['text', 'label'])

    total_slang = run_generation(TARGET_SLANG, "SLANG")
    total_form  = run_generation(TARGET_FORMAL, "FORMAL")

    print(f"\n{'='*65}")
    print(f"  SELESAI GENERATE KEDUA GAYA!")
    print(f"  Berhasil: {total_slang} Slang, {total_form} Formal")
    print(f"  Disimpan ke: {OUTPUT_CSV}")
    print(f"{'='*65}\n")

if __name__ == "__main__":
    main()