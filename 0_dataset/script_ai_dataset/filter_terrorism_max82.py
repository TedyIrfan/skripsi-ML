# =====================================================
# FILTER KAGGLE INDONESIAN TWEET ABOUT TERRORISM
# Target: 82 data (biar total 750)
# =====================================================

import pandas as pd
import re
import random
import sys
import kagglehub

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("="*60)
print("DOWNLOAD & FILTER INDONESIAN TWEET TERRORISM")
print("="*60)

print("\n[1/2] Downloading dataset...")
path = kagglehub.dataset_download('linkgish/indonesian-tweet-about-teroris-terrorism')

import os
print(f"\nPath: {path}")
print(f"Files in folder:")
files = []
for f in os.listdir(path):
    if f.endswith('.csv'):
        files.append(f)
        print(f"  - {f}")

if not files:
    print("[ERROR] Tidak ada file CSV!")
    exit()

# Load first CSV
csv_file = files[0]
print(f"\n[2/2] Loading: {csv_file}")
df = pd.read_csv(f"{path}/{csv_file}", on_bad_lines='skip', low_memory=False)

print(f"Total: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Find text column - prioritized
text_col = None
for col in df.columns:
    if 'tweet text' in col.lower():
        text_col = col
        break

if not text_col:
    for col in df.columns:
        if 'text' in col.lower():
            text_col = col
            break

if not text_col:
    for col in df.columns:
        if df[col].dtype == 'object':
            text_col = col
            break

print(f"Using column: {text_col}")

def count_sentences(text):
    sentences = re.split(r'[.!?]+', str(text))
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)

# Filter by OR logic, max 82
print(f"\nFiltering (OR logic, MAX 82)...")
filtered = []
TARGET = 82

for idx, row in df.iterrows():
    text = str(row[text_col])
    word_count = len(text.split())
    char_count = len(text)
    sentence_count = count_sentences(text)

    if ((150 <= word_count <= 500) or
        (1000 <= char_count <= 3000) or
        (8 <= sentence_count <= 20)):
        filtered.append({
            'text': text,
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count
        })

    if len(filtered) >= TARGET:
        break

print(f"Filtered: {len(filtered)} data")

if len(filtered) > TARGET:
    random.seed(42)
    filtered = filtered[:TARGET]
    print(f"Taking first {TARGET}")

print(f"Selected: {len(filtered)} data")

if len(filtered) > 0:
    df_final = pd.DataFrame([{
        'text': item['text'],
        'label': 'MANUSIA',
        'source': 'Kaggle_Tweet_Terrorism'
    } for item in filtered])

    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    output_file = 'data_manusia_kaggle_terrorism.csv'
    df_final.to_csv(output_file, index=False)

    print(f"\n[OK] Saved: {output_file}")
    print(f"Total: {len(df_final)}")

    # Statistics
    words = [len(t.split()) for t in df_final['text']]
    chars = [len(t) for t in df_final['text']]

    print(f"\nStatistics:")
    print(f"  Words: min={min(words)}, max={max(words)}, avg={sum(words)/len(words):.0f}")
    print(f"  Characters: min={min(chars)}, max={max(chars)}, avg={sum(chars)/len(chars):.0f}")

    # Show which criteria were met
    words_ok = sum(1 for i in filtered if 150 <= i['word_count'] <= 500)
    chars_ok = sum(1 for i in filtered if 1000 <= i['char_count'] <= 3000)
    sents_ok = sum(1 for i in filtered if 8 <= i['sentence_count'] <= 20)
    print(f"\nKriteria terpenuhi:")
    print(f"  - 150-500 kata: {words_ok}")
    print(f"  - 1000-3000 karakter: {chars_ok}")
    print(f"  - 8-20 kalimat: {sents_ok}")

else:
    print("\n[ERROR] Tidak ada data yang memenuhi kriteria!")

print("\n" + "="*60)
print("DONE!")
print("="*60)
