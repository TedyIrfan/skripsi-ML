# =====================================================
# FILTER KAGGLE REDDIT - AUTO DOWNLOAD KAGGLEHUB
# =====================================================

import pandas as pd
import re
import random
import sys
import kagglehub

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("="*60)
print("DOWNLOAD & FILTER REDDIT INDONESIA")
print("="*60)

# Download dataset
print("\n[1/2] Downloading dataset...")
path = kagglehub.dataset_download('bwandowando/reddit-rindonesia-subreddit-dataset-2026')
print(f"Downloaded to: {path}")

# List files
import os
print(f"\nFiles in folder:")
files = []
for f in os.listdir(path):
    if f.endswith('.csv'):
        files.append(f)
        print(f"  - {f}")

if not files:
    print("[ERROR] Tidak ada file CSV!")
    exit()

# Load the first CSV
csv_file = files[0]
print(f"\n[2/2] Loading: {csv_file}")
df = pd.read_csv(f"{path}/{csv_file}", on_bad_lines='skip', low_memory=False)

print(f"Total: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Find text column
text_col = None
for col in df.columns:
    if 'text' in col.lower() or 'content' in col.lower() or 'body' in col.lower() or 'komen' in col.lower() or 'selftext' in col.lower():
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

def estimate_paragraphs(text):
    return max(1, str(text).count('\n\n') + 1)

# Filter by OR logic
print(f"\nFiltering (OR logic)...")
filtered = []
for idx, row in df.iterrows():
    text = str(row[text_col])
    word_count = len(text.split())
    char_count = len(text)
    sentence_count = count_sentences(text)
    paragraph_count = estimate_paragraphs(text)

    if ((150 <= word_count <= 500) or
        (1000 <= char_count <= 3000) or
        (8 <= sentence_count <= 20) or
        (2 <= paragraph_count <= 5)):
        filtered.append({
            'text': text,
            'word_count': word_count,
            'char_count': char_count
        })

print(f"Filtered: {len(filtered)} data")

# Calculate needed
needed = 750 - 250 - 162 - 6
print(f"Needed: {needed} more data")

if len(filtered) > needed:
    random.seed(42)
    filtered = random.sample(filtered, needed)
elif len(filtered) < needed:
    print(f"[WARNING] Hanya {len(filtered)} data tersedia!")

if len(filtered) > 0:
    df_final = pd.DataFrame([{
        'text': item['text'],
        'label': 'MANUSIA',
        'source': 'Kaggle_Reddit_Indonesia'
    } for item in filtered])

    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    output_file = 'data_manusia_kaggle_reddit.csv'
    df_final.to_csv(output_file, index=False)

    print(f"\n[OK] Saved: {output_file}")
    print(f"Total: {len(df_final)}")
else:
    print("\n[ERROR] Tidak ada data yang memenuhi kriteria!")

print("\n" + "="*60)
print("DONE!")
print("="*60)
