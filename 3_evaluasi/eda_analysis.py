"""
EVALUASI 2 - EDA (Exploratory Data Analysis)
Analisis dan visualisasi dataset — distribusi, panjang teks,
sumber data, wordcloud AI vs Manusia
Output: hasil_phase5/2_eda_dataset/
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False
    print("[WARNING] wordcloud tidak terinstall. Skip wordcloud chart.")
    print("          Install: pip install wordcloud")

OUT = "hasil_phase5/2_eda_dataset"
os.makedirs(OUT, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')

print("="*60)
print("EVALUASI 2 — EDA DATASET")
print(f"Output: {OUT}/")
print("="*60)

df = pd.read_csv("dataset_clean_1500.csv", encoding='utf-8')
df = df.dropna()
df['text_length'] = df['text'].apply(len)
df['word_count']  = df['text'].apply(lambda x: len(x.split()))
ai_texts  = df[df['label']=='AI']['text']
man_texts = df[df['label']=='MANUSIA']['text']

print(f"\nTotal data: {len(df)}")
print(f"AI: {(df['label']=='AI').sum()}, MANUSIA: {(df['label']=='MANUSIA').sum()}")

# ── GRAFIK 1: Distribusi Label ──
print("\n[1] Distribusi Label...")
fig, ax = plt.subplots(figsize=(7, 5))
counts = df['label'].value_counts()
bars = ax.bar(counts.index, counts.values, color=['#e74c3c','#3498db'], alpha=0.85, width=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
            f'{val}\n({val/len(df)*100:.1f}%)', ha='center', fontsize=12, fontweight='bold')
ax.set_title('Distribusi Label Dataset\n(AI vs Manusia)', fontsize=14, fontweight='bold')
ax.set_ylabel('Jumlah Teks', fontsize=12)
ax.set_ylim([0, max(counts.values)*1.2])
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/1_distribusi_label.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT}/1_distribusi_label.png")

# ── GRAFIK 2: Distribusi Panjang Teks ──
print("\n[2] Distribusi Panjang Teks (AI vs Manusia)...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, col, label in zip(axes, ['text_length','word_count'], ['Panjang Teks (karakter)','Jumlah Kata']):
    ai_vals  = df[df['label']=='AI'][col]
    man_vals = df[df['label']=='MANUSIA'][col]
    ax.hist(ai_vals,  bins=40, alpha=0.6, color='#e74c3c', label=f'AI  (mean={ai_vals.mean():.0f})')
    ax.hist(man_vals, bins=40, alpha=0.6, color='#3498db', label=f'Manusia (mean={man_vals.mean():.0f})')
    ax.axvline(ai_vals.mean(),  color='#e74c3c', linestyle='--', linewidth=2)
    ax.axvline(man_vals.mean(), color='#3498db', linestyle='--', linewidth=2)
    ax.set_xlabel(label, fontsize=11, fontweight='bold')
    ax.set_ylabel('Frekuensi', fontsize=11, fontweight='bold')
    ax.set_title(f'Distribusi {label}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

plt.suptitle('Distribusi Panjang Teks — AI vs Manusia', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/2_distribusi_panjang_teks.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT}/2_distribusi_panjang_teks.png")

# ── GRAFIK 3: Distribusi Sumber Data ──
print("\n[3] Distribusi Sumber Data...")
if 'source' in df.columns:
    src_counts = df.groupby(['source', 'label']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(14, 6))
    src_counts.plot(kind='bar', ax=ax, color=['#e74c3c','#3498db'], alpha=0.85)
    ax.set_title('Distribusi Teks per Sumber Data', fontsize=14, fontweight='bold')
    ax.set_xlabel('Sumber', fontsize=11)
    ax.set_ylabel('Jumlah Teks', fontsize=11)
    ax.legend(['AI','Manusia'], fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{OUT}/3_distribusi_sumber.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {OUT}/3_distribusi_sumber.png")
else:
    print("  [SKIP] Kolom 'source' tidak ada di dataset")

# ── GRAFIK 4: Statistik Deskriptif ──
print("\n[4] Statistik Deskriptif...")
stats = df.groupby('label')[['text_length','word_count']].agg(['mean','median','min','max'])
print(stats.to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col, title in zip(axes,
    ['text_length','word_count'],
    ['Panjang Teks (karakter)','Jumlah Kata']):
    data = [df[df['label']=='AI'][col].values,
            df[df['label']=='MANUSIA'][col].values]
    bp = ax.boxplot(data, tick_labels=['AI','Manusia'], patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor('#e74c3c')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#3498db')
    bp['boxes'][1].set_alpha(0.7)
    ax.set_title(f'Boxplot {title}', fontsize=12, fontweight='bold')
    ax.set_ylabel(title, fontsize=11)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Boxplot Panjang Teks — AI vs Manusia', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/4_boxplot_panjang_teks.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT}/4_boxplot_panjang_teks.png")

# ── GRAFIK 5 & 6: Wordcloud ──
if HAS_WORDCLOUD:
    print("\n[5] Wordcloud AI...")
    wc_ai = WordCloud(width=1200, height=600, background_color='white',
                      colormap='Reds', max_words=150, collocations=False)
    wc_ai.generate(' '.join(ai_texts))
    plt.figure(figsize=(14, 7))
    plt.imshow(wc_ai, interpolation='bilinear')
    plt.axis('off')
    plt.title('Wordcloud — Teks AI', fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(f'{OUT}/5_wordcloud_ai.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {OUT}/5_wordcloud_ai.png")

    print("\n[6] Wordcloud Manusia...")
    wc_man = WordCloud(width=1200, height=600, background_color='white',
                       colormap='Blues', max_words=150, collocations=False)
    wc_man.generate(' '.join(man_texts))
    plt.figure(figsize=(14, 7))
    plt.imshow(wc_man, interpolation='bilinear')
    plt.axis('off')
    plt.title('Wordcloud — Teks Manusia', fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(f'{OUT}/6_wordcloud_manusia.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {OUT}/6_wordcloud_manusia.png")

print("\n" + "="*60)
print("EVALUASI 2 SELESAI!")
print("="*60)
print(f"\nFile tersimpan di: {OUT}/")
