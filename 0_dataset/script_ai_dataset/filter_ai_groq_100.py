# =====================================================
# FILTER AI DATA GROQ - MAX 100
# Kriteria sama kayak data manusia (OR logic)
# =====================================================

import pandas as pd
import re
import random
import sys

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("="*60)
print("FILTER AI DATA GROQ - MAX 100")
print("="*60)

print("\n[1/2] Loading dataset...")
df = pd.read_csv('dataset_ai_500.csv')

print(f"Total: {len(df)}")
print(f"Columns: {list(df.columns)}")

text_col = 'text'
print(f"Using column: {text_col}")

def count_sentences(text):
    sentences = re.split(r'[.!?]+', str(text))
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)

def estimate_paragraphs(text):
    return max(1, str(text).count('\n\n') + 1)

# Filter by OR logic
print(f"\n[2/2] Filtering (OR logic, MAX 100)...")
filtered = []
TARGET = 100

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
            'char_count': char_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count
        })

    if len(filtered) >= TARGET:
        break

print(f"Filtered: {len(filtered)} data")

if len(filtered) > TARGET:
    filtered = filtered[:TARGET]
    print(f"Taking first {TARGET}")

print(f"Selected: {len(filtered)} data")

if len(filtered) > 0:
    df_final = pd.DataFrame([{
        'text': item['text'],
        'label': 'AI',
        'source': 'Groq_Llama_3.3'
    } for item in filtered])

    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    output_file = 'data_ai_groq_100.csv'
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
    paras_ok = sum(1 for i in filtered if 2 <= i['paragraph_count'] <= 5)
    print(f"\nKriteria terpenuhi:")
    print(f"  - 150-500 kata: {words_ok}")
    print(f"  - 1000-3000 karakter: {chars_ok}")
    print(f"  - 8-20 kalimat: {sents_ok}")
    print(f"  - 2-5 paragraf: {paras_ok}")

else:
    print("\n[ERROR] Tidak ada data yang memenuhi kriteria!")

print("\n" + "="*60)
print("DONE!")
print("="*60)
