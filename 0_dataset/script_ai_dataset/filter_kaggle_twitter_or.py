# =====================================================
# FILTER KAGGLE TWITTER PPKM - LOGIKA OR
# Salah satu: 150-500 kata OR 1000-3000 karakter OR 8-20 kalimat
# =====================================================

import pandas as pd
import re
import kagglehub
import random

# Download and load
print("Downloading Kaggle dataset...")
path = kagglehub.dataset_download('anggapurnama/twitter-dataset-ppkm')

print(f"Loading labeled data...")
df = pd.read_csv(f"{path}/INA_TweetsPPKM_Labeled_Pure.csv",
                 sep='\t',
                 on_bad_lines='skip')

print(f"Total: {len(df)}")
print(f"Columns: {list(df.columns)}")

def count_sentences(text):
    """Count sentences in text"""
    sentences = re.split(r'[.!?]+', str(text))
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)

# Filter by criteria - OR logic
print("\nFiltering by criteria (OR logic):")
print("  - 150-500 kata")
print("  - OR 1000-3000 karakter")
print("  - OR 8-20 kalimat")

filtered = []
for idx, row in df.iterrows():
    text = str(row['Tweet'])
    word_count = len(text.split())
    char_count = len(text)
    sentence_count = count_sentences(text)

    # OR logic - salah satu kriteria
    if ((150 <= word_count <= 500) or
        (1000 <= char_count <= 3000) or
        (8 <= sentence_count <= 20)):
        filtered.append({
            'text': text,
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count
        })

print(f"\nFiltered: {len(filtered)} data")

# Take max 250
if len(filtered) > 250:
    random.seed(42)
    filtered = random.sample(filtered, 250)
    print(f"Took 250 random from {len(filtered)}")

print(f"Selected: {len(filtered)} data")

if len(filtered) == 0:
    print("\n[WARNING] Tidak ada data yang memenuhi kriteria!")
    print("Twitter tweets terlalu pendek untuk kriteria ini.")
else:
    # Create final dataframe
    df_final = pd.DataFrame([{
        'text': item['text'],
        'label': 'MANUSIA',
        'source': 'Kaggle_Twitter_PPKM'
    } for item in filtered])

    # Shuffle
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    output_file = 'data_manusia_kaggle_twitter.csv'
    df_final.to_csv(output_file, index=False)

    print(f"\n[OK] Saved: {output_file}")
    print(f"Total: {len(df_final)}")

    # Statistics
    words = [len(t.split()) for t in df_final['text']]
    chars = [len(t) for t in df_final['text']]

    print(f"\nStatistics:")
    print(f"  Words: min={min(words)}, max={max(words)}, avg={sum(words)/len(words):.0f}")
    print(f"  Characters: min={min(chars)}, max={max(chars)}, avg={sum(chars)/len(chars):.0f}")

    print(f"\nSample:")
    for i in range(min(3, len(df_final))):
        print(f"\n{i+1}. {df_final.iloc[i]['text'][:150]}...")

print("\n" + "="*60)
print("DONE!")
print("="*60)
