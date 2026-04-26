# =====================================================
# FILTER 250 DATA INDOSUM - LOGIKA OR
# Salah satu: 150-500 kata OR 1000-3000 karakter OR 8-20 kalimat OR 2-5 paragraf
# =====================================================

import pandas as pd
import ast
import re
import random

# Load IndoSum dev
print("Loading IndoSum dev data...")
df_indosum = pd.read_csv('indosum_dev.csv')

print(f"Total data: {len(df_indosum)}")

def flatten_indosum_paragraphs(para_str):
    """Flatten nested list structure from IndoSum paragraphs"""
    try:
        paras = ast.literal_eval(para_str)
        all_words = []
        for group in paras:
            if isinstance(group, list):
                for item in group:
                    if isinstance(item, list):
                        all_words.extend(item)
                    else:
                        all_words.append(str(item))
            else:
                all_words.append(str(group))
        return ' '.join(all_words)
    except:
        return str(para_str)

def count_sentences(text):
    """Count sentences in text"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)

def count_paragraphs(para_str):
    """Count paragraphs in original IndoSum data"""
    try:
        paras = ast.literal_eval(para_str)
        return len([p for p in paras if p and len(p) > 0])
    except:
        return 1

# Extract all data first
print("\nExtracting text...")
all_text = []
for idx, row in df_indosum.iterrows():
    text = flatten_indosum_paragraphs(row['paragraphs'])

    word_count = len(text.split())
    char_count = len(text)
    sentence_count = count_sentences(text)
    paragraph_count = count_paragraphs(row['paragraphs'])

    all_text.append({
        'text': text,
        'word_count': word_count,
        'char_count': char_count,
        'sentence_count': sentence_count,
        'paragraph_count': paragraph_count,
        'category': row['category']
    })

print(f"Total extracted: {len(all_text)}")

# Filter by criteria - OR logic
print("\nFiltering by criteria (OR logic):")
print("  - 150-500 kata")
print("  - OR 1000-3000 karakter")
print("  - OR 8-20 kalimat")
print("  - OR 2-5 paragraf")

filtered = []
for item in all_text:
    # Check if AT LEAST ONE criteria is met
    if ((150 <= item['word_count'] <= 500) or
        (1000 <= item['char_count'] <= 3000) or
        (8 <= item['sentence_count'] <= 20) or
        (2 <= item['paragraph_count'] <= 5)):
        filtered.append(item)

print(f"\nFiltered: {len(filtered)} data")

# Take max 250
if len(filtered) > 250:
    random.seed(42)
    selected = random.sample(filtered, 250)
    print(f"Took 250 random from {len(filtered)}")
else:
    selected = filtered

print(f"Selected: {len(selected)} data")

# Create final dataframe
df_final = pd.DataFrame([{
    'text': item['text'],
    'label': 'MANUSIA',
    'source': 'IndoSum',
    'category': item['category']
} for item in selected])

# Shuffle
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
df_final.to_csv('data_manusia_indosum_250.csv', index=False)

print(f"\n[OK] Saved: data_manusia_indosum_250.csv")

# Show which criteria each data met
print("\nKriteria yang terpenuhi:")
words_ok = sum(1 for i in selected if 150 <= i['word_count'] <= 500)
chars_ok = sum(1 for i in selected if 1000 <= i['char_count'] <= 3000)
sents_ok = sum(1 for i in selected if 8 <= i['sentence_count'] <= 20)
paras_ok = sum(1 for i in selected if 2 <= i['paragraph_count'] <= 5)
print(f"  - 150-500 kata: {words_ok}")
print(f"  - 1000-3000 karakter: {chars_ok}")
print(f"  - 8-20 kalimat: {sents_ok}")
print(f"  - 2-5 paragraf: {paras_ok}")

print(f"\nDistribusi Category:")
category_counts = {}
for item in selected:
    cat = item['category']
    category_counts[cat] = category_counts.get(cat, 0) + 1
for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  - {cat}: {count}")

print(f"\nSample data (3 pertama):")
for i in range(min(3, len(selected))):
    item = selected[i]
    met = []
    if 150 <= item['word_count'] <= 500:
        met.append(f"words={item['word_count']}")
    if 1000 <= item['char_count'] <= 3000:
        met.append(f"chars={item['char_count']}")
    if 8 <= item['sentence_count'] <= 20:
        met.append(f"sentences={item['sentence_count']}")
    if 2 <= item['paragraph_count'] <= 5:
        met.append(f"paragraphs={item['paragraph_count']}")
    print(f"\n{i+1}. Met: {', '.join(met)}")
    print(f"   {item['text'][:120]}...")

print("\n" + "="*60)
print("DONE!")
print("="*60)
