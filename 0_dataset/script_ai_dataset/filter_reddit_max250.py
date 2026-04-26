# =====================================================
# FILTER KAGGLE REDDIT - MAX 250
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
print("DOWNLOAD & FILTER REDDIT INDONESIA - MAX 250")
print("="*60)

print("\n[1/2] Downloading dataset...")
path = kagglehub.dataset_download('bwandowando/reddit-rindonesia-subreddit-dataset-2026')

import os
files = [f for f in os.listdir(path) if f.endswith('.csv')]
print(f"Files: {files}")

csv_file = 'indonesia_subreddit_comments.csv'
print(f"\n[2/2] Loading: {csv_file}")
df = pd.read_csv(f"{path}/{csv_file}", on_bad_lines='skip', low_memory=False)

print(f"Total: {len(df)}")

text_col = 'body'
print(f"Using column: {text_col}")

def count_sentences(text):
    sentences = re.split(r'[.!?]+', str(text))
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)

print(f"\nFiltering (OR logic, MAX 250)...")
filtered = []
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
            'char_count': char_count
        })

    if len(filtered) >= 250:
        break

print(f"Filtered: {len(filtered)} data")

if len(filtered) > 250:
    random.seed(42)
    filtered = filtered[:250]
    print(f"Taking first 250")

print(f"Selected: {len(filtered)} data")

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

print("\n" + "="*60)
print("DONE!")
print("="*60)
