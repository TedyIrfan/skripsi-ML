"""
EVALUASI 5 - Per Source / Per Pola Teks
Karena dataset tidak punya kolom 'source', analisis dilakukan
berdasarkan POLA TEKS: panjang, formalitas, tanda sumber
untuk semua 4 model
Output: hasil_phase5/5_per_pola_teks/
"""
import pandas as pd
import numpy as np
import joblib
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

OUT = "hasil_phase5/5_per_pola_teks"
os.makedirs(OUT, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')

print("="*60)
print("EVALUASI 5 — ANALISIS PER POLA TEKS (4 MODEL)")
print(f"Output: {OUT}/")
print("="*60)
print("\nNote: Dataset tidak punya kolom 'source'.")
print("Analisis dilakukan berdasarkan POLA TEKS yang terdeteksi.")

# Load data
df = pd.read_csv("dataset_skripsi_manusia_ai_1510.csv", encoding='utf-8')
df = df.dropna()
df['label_num'] = df['label'].map({'MANUSIA': 0, 'AI': 1})
X = df['text']
y = df['label_num']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Load / train model
print("\nLoad & train model...")
pipe_lr = joblib.load('models_strict/best_pipeline_logistic_regression.pkl')
pipe_svm = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8, ngram_range=(1,2))),
    ('clf', SVC(C=1.0, kernel='rbf', probability=True, random_state=42))
])
pipe_rf = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8, ngram_range=(1,2))),
    ('clf', RandomForestClassifier(n_estimators=100, max_depth=20,
                                   min_samples_split=5, random_state=42, n_jobs=-1))
])
pipe_svm.fit(X_train, y_train)
pipe_rf.fit(X_train, y_train)
print("  [OK] LR, SVM, RF siap")

models = {'LR': pipe_lr, 'SVM': pipe_svm, 'RF': pipe_rf}

# Buat DataFrame test dengan fitur pola teks
test_df = pd.DataFrame({'text': X_test.values, 'true_label': y_test.values})
test_df['true_str']  = test_df['true_label'].map({0:'MANUSIA', 1:'AI'})
test_df['text_len']  = test_df['text'].str.len()
test_df['word_cnt']  = test_df['text'].str.split().str.len()

# Fitur pola
test_df['has_mention']   = test_df['text'].str.contains(r'@\w+', regex=True).astype(int)
test_df['has_hashtag']   = test_df['text'].str.contains(r'#\w+', regex=True).astype(int)
test_df['has_media_tag'] = test_df['text'].str.contains(
    r'CNN|Kompas|Detik|Tribun|liputan6|antara', case=False, regex=True).astype(int)
test_df['has_slang']     = test_df['text'].str.contains(
    r'\bwkwk\b|\bgue\b|\belo\b|\bbanget\b|\bsih\b|\bnih\b|\bdong\b|\bkan\b',
    case=False, regex=True).astype(int)
test_df['is_formal']     = (
    (test_df['text_len'] > 500) &
    (test_df['has_mention'] == 0) &
    (test_df['has_slang'] == 0)
).astype(int)

# Kategori panjang teks
def length_cat(n):
    if n < 200: return 'Sangat Pendek\n(<200)'
    if n < 500: return 'Pendek\n(200-500)'
    if n < 1000: return 'Sedang\n(500-1000)'
    if n < 2000: return 'Panjang\n(1000-2000)'
    return 'Sangat Panjang\n(>2000)'

test_df['len_cat'] = test_df['text_len'].apply(length_cat)

# Prediksi tiap model
print("\nPrediksi semua model...")
for name, pipe in models.items():
    preds = pipe.predict(X_test)
    test_df[f'pred_{name}'] = [({0:'MANUSIA',1:'AI'}[v]) for v in preds]
    test_df[f'ok_{name}']   = test_df['true_str'] == test_df[f'pred_{name}']

# ── GRAFIK 1: Akurasi per Panjang Teks (semua model) ──
print("\n[1] Akurasi per Kategori Panjang Teks...")
len_order = ['Sangat Pendek\n(<200)', 'Pendek\n(200-500)', 'Sedang\n(500-1000)',
             'Panjang\n(1000-2000)', 'Sangat Panjang\n(>2000)']

acc_len = {}
for cat in len_order:
    sub = test_df[test_df['len_cat'] == cat]
    if len(sub) == 0:
        continue
    acc_len[cat] = {
        'n': len(sub),
        'LR':  sub['ok_LR'].mean()*100,
        'SVM': sub['ok_SVM'].mean()*100,
        'RF':  sub['ok_RF'].mean()*100,
    }

cats   = list(acc_len.keys())
x      = np.arange(len(cats))
fig, ax = plt.subplots(figsize=(13, 7))
w = 0.25
b1 = ax.bar(x-w,   [acc_len[c]['LR']  for c in cats], w, label='LR',  color='#2196F3', alpha=0.85)
b2 = ax.bar(x,     [acc_len[c]['SVM'] for c in cats], w, label='SVM', color='#4CAF50', alpha=0.85)
b3 = ax.bar(x+w,   [acc_len[c]['RF']  for c in cats], w, label='RF',  color='#FF9800', alpha=0.85)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x()+bar.get_width()/2, h+0.2,
                    f'{h:.1f}', ha='center', fontsize=8, fontweight='bold')

for i, cat in enumerate(cats):
    ax.text(i, 2, f'n={acc_len[cat]["n"]}', ha='center', fontsize=9, color='gray')

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Akurasi Model per Kategori Panjang Teks\n(3 model klasik, test set 302 data)',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(cats, fontsize=10)
ax.set_ylim([0, 105])
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/1_akurasi_per_panjang_teks.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT}/1_akurasi_per_panjang_teks.png")

# ── GRAFIK 2: Akurasi pada teks ber-penanda sumber (mention, hashtag, media) ──
print("\n[2] Akurasi pada Teks dengan Penanda Sumber...")
patterns = {
    'Ada @mention': test_df['has_mention']==1,
    'Ada #hashtag':  test_df['has_hashtag']==1,
    'Ada Nama Media': test_df['has_media_tag']==1,
    'Ada Slang/Gaul': test_df['has_slang']==1,
    'Teks Formal\n(panjang+no @/slang)': test_df['is_formal']==1,
    'Tanpa Penanda\nKhusus': (
        (test_df['has_mention']==0) & (test_df['has_hashtag']==0) &
        (test_df['has_media_tag']==0) & (test_df['has_slang']==0)
    ),
}

pat_data = {}
for label, mask in patterns.items():
    sub = test_df[mask]
    if len(sub) > 2:
        pat_data[label] = {
            'n': len(sub),
            'LR':  sub['ok_LR'].mean()*100,
            'SVM': sub['ok_SVM'].mean()*100,
            'RF':  sub['ok_RF'].mean()*100,
        }

pats = list(pat_data.keys())
x    = np.arange(len(pats))
fig, ax = plt.subplots(figsize=(14, 7))
b1 = ax.bar(x-w, [pat_data[p]['LR']  for p in pats], w, label='LR',  color='#2196F3', alpha=0.85)
b2 = ax.bar(x,   [pat_data[p]['SVM'] for p in pats], w, label='SVM', color='#4CAF50', alpha=0.85)
b3 = ax.bar(x+w, [pat_data[p]['RF']  for p in pats], w, label='RF',  color='#FF9800', alpha=0.85)

for bars in [b1,b2,b3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x()+bar.get_width()/2, h+0.2,
                    f'{h:.0f}', ha='center', fontsize=8.5, fontweight='bold')

for i, p in enumerate(pats):
    ax.text(i, 1.5, f'n={pat_data[p]["n"]}', ha='center', fontsize=8.5, color='gray')

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Akurasi Model Berdasarkan Pola/Penanda dalam Teks\n(Bukti fitur non-linguistik memengaruhi prediksi)',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(pats, fontsize=9)
ax.set_ylim([0, 108])
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/2_akurasi_per_pola_teks.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT}/2_akurasi_per_pola_teks.png")

# ── Print ringkasan per pola ──
print("\n" + "="*60)
print("RINGKASAN AKURASI PER POLA TEKS")
print("="*60)
print(f"\n{'Pola':<30} {'n':>5} {'LR':>8} {'SVM':>8} {'RF':>8}")
print("-"*60)
for p in pats:
    d = pat_data[p]
    clean = p.replace('\n',' ')
    print(f"{clean:<30} {d['n']:>5} {d['LR']:>7.1f}% {d['SVM']:>7.1f}% {d['RF']:>7.1f}%")

print(f"\n  [OK] Selengkapnya lihat grafik di {OUT}/")
print("\n" + "="*60)
print("EVALUASI 5 SELESAI!")
print("="*60)
print(f"\nFile tersimpan di: {OUT}/")
print("  1_akurasi_per_panjang_teks.png   → Akurasi vs panjang teks")
print("  2_akurasi_per_pola_teks.png      → Akurasi vs penanda sumber")
