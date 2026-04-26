# =====================================================
# TAMBAH AI DATA MANUAL (COPY-PASTE DARI GPT)
# =====================================================
#
# Cara pakai:
# 1. Copy teks hasil generate dari GPT
# 2. Run script ini
# 3. Paste teks ketika diminta
# 4. Tekan Enter dua kali untuk selesai
# 5. Ulangi untuk teks berikutnya
#
# =====================================================

import pandas as pd
import re
import sys

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OUTPUT_FILE = 'data_ai_gpt_manual.csv'

def count_words(text):
    return len(text.split())

def count_sentences(text):
    sentences = re.split(r'[.!?]+', str(text))
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)

# Load existing data
existing_data = []
try:
    df_existing = pd.read_csv(OUTPUT_FILE)
    existing_data = df_existing.to_dict('records')
    print(f"[INFO] Existing data: {len(existing_data)}")
except:
    print(f"[INFO] Creating new file: {OUTPUT_FILE}")

print("="*60)
print("TAMBAH AI DATA (GPT - MANUAL)")
print("="*60)
print("\nKriteria:")
print("  - 150-500 kata ATAU")
print("  - 1000-3000 karakter ATAU")
print("  - 8-20 kalimat ATAU")
print("  - 2-5 paragraf")
print("\n" + "="*60)

while True:
    print("\n" + "-"*60)
    print("Masukkan teks AI baru:")
    print("(Tekan Enter dua kali / Enter kosong untuk selesai)")
    print("-"*60)

    lines = []
    while True:
        try:
            line = input()
            if line == "":
                break
            lines.append(line)
        except EOFError:
            break

    if not lines:
        break

    text = "\n".join(lines).strip()

    if not text:
        break

    word_count = count_words(text)
    char_count = len(text)
    sentence_count = count_sentences(text)
    paragraph_count = max(1, text.count('\n\n') + 1)

    print(f"\nStatistik:")
    print(f"  Kata: {word_count}")
    print(f"  Karakter: {char_count}")
    print(f"  Kalimat: {sentence_count}")
    print(f"  Paragraf: {paragraph_count}")

    meets_criteria = (
        (150 <= word_count <= 500) or
        (1000 <= char_count <= 3000) or
        (8 <= sentence_count <= 20) or
        (2 <= paragraph_count <= 5)
    )

    if meets_criteria:
        existing_data.append({
            'text': text,
            'label': 'AI',
            'source': 'GPT_Manual',
            'word_count': word_count,
            'char_count': char_count
        })
        print(f"  [OK] DITAMBAHKAN!")
    else:
        print(f"  [SKIP] Tidak memenuhi kriteria")
        confirm = input("  Tetap simpan? (y/n): ").lower()
        if confirm == 'y':
            existing_data.append({
                'text': text,
                'label': 'AI',
                'source': 'GPT_Manual',
                'word_count': word_count,
                'char_count': char_count
            })
            print(f"  [OK] DITAMBAHKAN (manual override)")

    print(f"\nTotal data: {len(existing_data)}")

# Save all data
if existing_data:
    df = pd.DataFrame(existing_data)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "="*60)
    print(f"SELESAI!")
    print(f"Total data: {len(df)}")
    print(f"File: {OUTPUT_FILE}")
    print("="*60)

    if 'word_count' in df.columns:
        print(f"\nStatistics:")
        print(f"  Words: min={df['word_count'].min()}, max={df['word_count'].max()}, avg={df['word_count'].mean():.0f}")
else:
    print("\nTidak ada data yang disimpan.")
