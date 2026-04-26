# Import library yang diperlukan
import pandas as pd
import numpy as np
import joblib
import re
from collections import Counter
import warnings
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

OUT = "hasil_phase5/4_linguistic_analysis"  # folder berbeda dari 4_error_analysis
os.makedirs(OUT, exist_ok=True)

print("ANALISIS LINGUISTIK - PERBEDAAN AI vs MANUSIA")

# Load dataset dulu
print("\nLoad dataset...")
df = pd.read_csv("dataset_clean_1500.csv", encoding='utf-8')
df = df.dropna()

# Pisah teks AI dan Manusia
ai_texts = df[df['label'] == 'AI']['text'].tolist()
human_texts = df[df['label'] == 'MANUSIA']['text'].tolist()

print(f"     AI texts: {len(ai_texts)}")
print(f"     Human texts: {len(human_texts)}")

# Analisis 1: Panjang kalimat rata-rata
print("\n" + "="*70)
print("1. ANALISIS PANJANG KALIMAT")

def avg_sentence_length(texts):
    """Hitung rata-rata panjang kalimat dalam kata"""
    total_words = 0
    total_sentences = 0

    for text in texts:
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]

        total_words += len(words)
        total_sentences += len(sentences)

    return total_words / max(total_sentences, 1)

# Hitung rata-rata panjang kalimat
ai_avg_sent_len = avg_sentence_length(ai_texts)
human_avg_sent_len = avg_sentence_length(human_texts)

print(f"\nRata-rata panjang kalimat (dalam kata):")
print(f"  AI:      {ai_avg_sent_len:.2f} kata/kalimat")
print(f"  Manusia: {human_avg_sent_len:.2f} kata/kalimat")
print(f"  Selisih: {abs(ai_avg_sent_len - human_avg_sent_len):.2f} kata")

# Analisis 2: Keragaman leksikal (Lexical Diversity)
print("\n" + "="*70)
print("2. KERAGAMAN LEKSIKAL (LEXICAL DIVERSITY)")

def lexical_diversity(texts):
    """Hitung TTR (Type-Token Ratio)"""
    all_words = []

    for text in texts:
        words = text.lower().split()
        all_words.extend(words)

    unique_words = set(all_words)
    ttr = len(unique_words) / len(all_words) if len(all_words) > 0 else 0

    return {
        'total_words': len(all_words),
        'unique_words': len(unique_words),
        'ttr': ttr
    }

# Hitung keragaman leksikal
ai_lex = lexical_diversity(ai_texts)
human_lex = lexical_diversity(human_texts)

print(f"\nStatistik Leksikal:")
print(f"  {'':<15} {'Total Kata':<12} {'Kata Unik':<12} {'TTR':<10}")
print(f"  {'-'*50}")
print(f"  {'AI':<15} {ai_lex['total_words']:<12} {ai_lex['unique_words']:<12} {ai_lex['ttr']:<10.4f}")
print(f"  {'Manusia':<15} {human_lex['total_words']:<12} {human_lex['unique_words']:<12} {human_lex['ttr']:<10.4f}")

print(f"\nInterpretasi TTR:")
print(f"  AI:      {ai_lex['ttr']*100:.2f}% (semakin tinggi = semakin beragam)")
print(f"  Manusia: {human_lex['ttr']*100:.2f}%")

# Analisis 3: Panjang kata rata-rata
print("\n" + "="*70)
print("3. PANJANG KATA RATA-RATA")

def avg_word_length(texts):
    """Hitung rata-rata panjang kata"""
    total_chars = 0
    total_words = 0

    for text in texts:
        words = text.split()
        total_words += len(words)
        total_chars += sum(len(word) for word in words)

    return total_chars / max(total_words, 1)

# Hitung rata-rata panjang kata
ai_avg_word_len = avg_word_length(ai_texts)
human_avg_word_len = avg_word_length(human_texts)

print(f"\nRata-rata panjang kata (karakter):")
print(f"  AI:      {ai_avg_word_len:.2f} karakter")
print(f"  Manusia: {human_avg_word_len:.2f} karakter")
print(f"  Selisih: {abs(ai_avg_word_len - human_avg_word_len):.2f} karakter")

# Analisis 4: Frekuensi tanda baca
print("\n" + "="*70)
print("4. FREKUENSI TANDA BACA")

def punctuation_analysis(texts):
    """Hitung frekuensi tanda baca"""
    punct_count = Counter()

    for text in texts:
        punct_count['.'] += text.count('.')
        punct_count[','] += text.count(',')
        punct_count['!'] += text.count('!')
        punct_count['?'] += text.count('?')
        punct_count[';'] += text.count(';')
        punct_count[':'] += text.count(':')

    return punct_count

# Hitung frekuensi tanda baca
ai_punct = punctuation_analysis(ai_texts)
human_punct = punctuation_analysis(human_texts)

print(f"\nFrekuensi tanda baca per 1000 kata:")
print(f"  {'Tanda':<10} {'AI':<12} {'Manusia':<12}")
print(f"  {'-'*35}")

for punct in ['.', ',', '!', '?', ';', ':']:
    ai_rate = (ai_punct[punct] / ai_lex['total_words']) * 1000
    human_rate = (human_punct[punct] / human_lex['total_words']) * 1000
    print(f"  {punct:<10} {ai_rate:<12.2f} {human_rate:<12.2f}")

# Ringkasan perbedaan
print("\n" + "="*70)
print("RINGKASAN PERBEDAAN LINGUISTIK")

# Buat ringkasan
summary = f"""
Perbedaan Utama Teks AI vs Manusia:

1. Panjang Kalimat:
   - AI:      {ai_avg_sent_len:.2f} kata/kalimat
   - Manusia: {human_avg_sent_len:.2f} kata/kalimat
   - {"AI lebih panjang" if ai_avg_sent_len > human_avg_sent_len else "Manusia lebih panjang"}

2. Keragaman Leksikal (TTR):
   - AI:      {ai_lex['ttr']*100:.2f}%
   - Manusia: {human_lex['ttr']*100:.2f}%
   - {"AI lebih beragam" if ai_lex['ttr'] > human_lex['ttr'] else "Manusia lebih beragam"}

3. Panjang Kata:
   - AI:      {ai_avg_word_len:.2f} karakter
   - Manusia: {human_avg_word_len:.2f} karakter
"""

print(summary)

# ======================================================
# VISUALISASI
# ======================================================
print("\nMembuat visualisasi analisis linguistik...")
plt.style.use('seaborn-v0_8-whitegrid')

# GRAFIK 1: Perbandingan Metrik Linguistik (4 panel)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Panjang Kalimat Rata-rata
categories = ['AI', 'Manusia']
sent_lengths = [ai_avg_sent_len, human_avg_sent_len]
bars = axes[0, 0].bar(categories, sent_lengths, color=['#e74c3c', '#3498db'],
                       alpha=0.85, edgecolor='white', width=0.5)
for bar, val in zip(bars, sent_lengths):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.2f}', ha='center', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Rata-rata Panjang Kalimat\n(kata per kalimat)', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Kata/Kalimat')
axes[0, 0].grid(axis='y', alpha=0.3)

# Panel 2: Keragaman Leksikal (TTR)
ttr_vals = [ai_lex['ttr'] * 100, human_lex['ttr'] * 100]
bars = axes[0, 1].bar(categories, ttr_vals, color=['#e74c3c', '#3498db'],
                       alpha=0.85, edgecolor='white', width=0.5)
for bar, val in zip(bars, ttr_vals):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.2f}%', ha='center', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Keragaman Leksikal (TTR)\n(Tipe-Token Ratio)', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('TTR (%)')
axes[0, 1].grid(axis='y', alpha=0.3)

# Panel 3: Panjang Kata Rata-rata
word_lengths = [ai_avg_word_len, human_avg_word_len]
bars = axes[1, 0].bar(categories, word_lengths, color=['#e74c3c', '#3498db'],
                       alpha=0.85, edgecolor='white', width=0.5)
for bar, val in zip(bars, word_lengths):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{val:.2f}', ha='center', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Rata-rata Panjang Kata\n(karakter per kata)', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Karakter/Kata')
axes[1, 0].grid(axis='y', alpha=0.3)

# Panel 4: Frekuensi Tanda Baca per 1000 kata
punct_labels = ['Titik (.)', 'Koma (,)', 'Seru (!)', 'Tanya (?)', 'Titik Koma (;)', 'Titik Dua (:)']
ai_punct_rates = [(ai_punct[p] / ai_lex['total_words']) * 1000 for p in ['.', ',', '!', '?', ';', ':']]
human_punct_rates = [(human_punct[p] / human_lex['total_words']) * 1000 for p in ['.', ',', '!', '?', ';', ':']]

x = np.arange(len(punct_labels))
width = 0.35
bars1 = axes[1, 1].bar(x - width/2, ai_punct_rates, width, label='AI', color='#e74c3c', alpha=0.85)
bars2 = axes[1, 1].bar(x + width/2, human_punct_rates, width, label='Manusia', color='#3498db', alpha=0.85)
axes[1, 1].set_title('Frekuensi Tanda Baca\n(per 1000 kata)', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Frekuensi / 1000 Kata')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(punct_labels, fontsize=8, rotation=15)
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.suptitle('Analisis Linguistik — AI vs Manusia', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/1_perbandingan_linguistik_4panel.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT}/1_perbandingan_linguistik_4panel.png")

# GRAFIK 2: Jumlah Kata & Kata Unik
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(2)
width = 0.3
vals_ai = [ai_lex['total_words'], ai_lex['unique_words']]
vals_human = [human_lex['total_words'], human_lex['unique_words']]
bars1 = ax.bar(x - width/2, vals_ai, width, label='AI', color='#e74c3c', alpha=0.85)
bars2 = ax.bar(x + width/2, vals_human, width, label='Manusia', color='#3498db', alpha=0.85)
for bar, val in zip(bars1, vals_ai):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'{val:,}', ha='center', fontsize=10, fontweight='bold', color='#e74c3c')
for bar, val in zip(bars2, vals_human):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'{val:,}', ha='center', fontsize=10, fontweight='bold', color='#3498db')
ax.set_xticks(x)
ax.set_xticklabels(['Total Kata', 'Kata Unik'], fontsize=11)
ax.set_ylabel('Jumlah', fontsize=12)
ax.set_title('Statistik Leksikal — Total Kata vs Kata Unik', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/2_statistik_leksikal.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT}/2_statistik_leksikal.png")

print(f"\nSemua grafik tersimpan di: {OUT}/")
print("Analisis Linguistik Selesai!")
