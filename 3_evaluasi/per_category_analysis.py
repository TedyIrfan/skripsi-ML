import pandas as pd
import numpy as np
import joblib
import glob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

OUT = "hasil_phase5/9_per_category_analysis"
os.makedirs(OUT, exist_ok=True)

print("ANALISIS PER KATEGORI/MODEL AI")

print("\nLoad dataset...")
df = pd.read_csv("dataset_clean_1500.csv", encoding='utf-8')
df = df.dropna()
print(f"     Total data: {len(df)}")

if 'ai_model' in df.columns or 'source' in df.columns:
    print("     [OK] Dataset memiliki kolom kategori")
else:
    print("     [!] Dataset tidak memiliki kolom ai_model/source")
    print("     Mencoba merge dengan file AI data asli...")

ai_files = glob.glob("dataset_ai_*.csv")
print(f"\nMencari file AI data...")
print(f"     Ditemukan {len(ai_files)} file AI:")

file_info = {}
for f in ai_files:
    try:
        temp_df = pd.read_csv(f, encoding='utf-8')
        print(f"     - {f}: {len(temp_df)} baris")
        if 'ai_model' in temp_df.columns:
            file_info[f] = temp_df
            print(f"       [OK] Memiliki kolom ai_model")
        elif 'model' in temp_df.columns:
            temp_df = temp_df.rename(columns={'model': 'ai_model'})
            file_info[f] = temp_df
            print(f"       [OK] Memiliki kolom model (di-rename)")
    except Exception as e:
        print(f"     - {f}: Error - {e}")

print("\n" + "="*70)
print("DISTRIBUSI DATA")

print("\nDistribusi Label:")
label_counts = df['label'].value_counts()
for label, count in label_counts.items():
    pct = count / len(df) * 100
    print(f"  {label}: {count} ({pct:.1f}%)")

print("\n" + "="*70)
print("LOAD MODEL UNTUK PREDIKSI")

print("\nTrain 3 model pipeline (LR, SVM, RF)...")
tfidf_params = dict(max_features=5000, min_df=2, max_df=0.8, ngram_range=(1, 2))
pipe_lr = Pipeline([('tfidf', TfidfVectorizer(**tfidf_params)),
                    ('clf', LogisticRegression(max_iter=1000, random_state=42, C=1.0))])
pipe_svm = Pipeline([('tfidf', TfidfVectorizer(**tfidf_params)),
                     ('clf', SVC(C=1.0, kernel='rbf', probability=True, random_state=42))])
pipe_rf  = Pipeline([('tfidf', TfidfVectorizer(**tfidf_params)),
                     ('clf', RandomForestClassifier(n_estimators=100, max_depth=20,
                                                    min_samples_split=5, random_state=42, n_jobs=-1))])

label_mapping = {'MANUSIA': 0, 'AI': 1}
reverse_label_mapping = {0: 'MANUSIA', 1: 'AI'}
df['label_num'] = df['label'].map(label_mapping)

X = df['text']
y = df['label_num']

pipe_lr.fit(X, y)
pipe_svm.fit(X, y)
pipe_rf.fit(X, y)
print("  [OK] LR, SVM, RF trained")

models_dict = {'LR': pipe_lr, 'SVM': pipe_svm, 'RF': pipe_rf}

# Gunakan LR sebagai model utama (untuk analisis per label)
pipeline = pipe_lr
y_pred = pipe_lr.predict(X)
y_proba = pipe_lr.predict_proba(X)

df['predicted_label'] = y_pred
df['predicted_label_str'] = df['predicted_label'].map(reverse_label_mapping)
df['proba_ai'] = y_proba[:, 1]
df['proba_manusia'] = y_proba[:, 0]

print("\n" + "="*70)
print("ANALISIS PER LABEL")

for label_val, label_name in [(0, 'MANUSIA'), (1, 'AI')]:
    subset = df[df['label_num'] == label_val]
    correct = subset[subset['predicted_label'] == label_val]
    wrong = subset[subset['predicted_label'] != label_val]

    accuracy = len(correct) / len(subset) * 100 if len(subset) > 0 else 0

    print(f"\n{label_name}:")
    print(f"  Total: {len(subset)}")
    print(f"  Benar: {len(correct)}")
    print(f"  Salah: {len(wrong)}")
    print(f"  Accuracy: {accuracy:.2f}%")

    if len(wrong) > 0:
        print(f"\n  Contoh yang salah prediksi:")
        for i, (idx, row) in enumerate(wrong.head(3).iterrows(), 1):
            print(f"\n  [{i}] {label_name} -> {row['predicted_label_str']}")
            print(f"      Confidence: {row['proba_ai' if row['predicted_label'] == 1 else 'proba_manusia']*100:.1f}%")
            print(f"      Text: {row['text'][:150]}...")

human_files = glob.glob("dataset_manusia*.csv") + glob.glob("*manusia*.csv")

print(f"\nMencari file data manusia...")
for f in human_files:
    try:
        temp_df = pd.read_csv(f, encoding='utf-8')
        source_col = None
        for col in ['source', 'category', 'kategori', 'sumber', 'type']:
            if col in temp_df.columns:
                source_col = col
                break

        if source_col:
            print(f"\n  File: {f}")
            print(f"  Kolom sumber: {source_col}")
            print(f"  Kategori: {temp_df[source_col].unique().tolist()}")
    except:
        pass

print("\n" + "="*70)
print("ANALISIS PANJANG TEKS PER LABEL")

df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()

for label_val, label_name in [(0, 'MANUSIA'), (1, 'AI')]:
    subset = df[df['label_num'] == label_val]
    print(f"\n{label_name}:")
    print(f"  Rata-rata panjang: {subset['text_length'].mean():.1f} karakter")
    print(f"  Median: {subset['text_length'].median():.1f}")
    print(f"  Min: {subset['text_length'].min()}, Max: {subset['text_length'].max()}")
    print(f"  Rata-rata kata: {subset['word_count'].mean():.1f}")

print("\n" + "="*70)
print("ANALISIS CONFIDENCE PREDIKSI PER LABEL")

for label_val, label_name in [(0, 'MANUSIA'), (1, 'AI')]:
    subset = df[df['label_num'] == label_val]
    print(f"\n{label_name}:")
    print(f"  Rata-rata confidence (prediksi benar): ", end="")
    correct = subset[subset['predicted_label'] == label_val]
    if len(correct) > 0:
        conf_col = 'proba_manusia' if label_val == 0 else 'proba_ai'
        print(f"{correct[conf_col].mean()*100:.2f}%")

    print(f"  Rata-rata confidence (prediksi salah): ", end="")
    wrong = subset[subset['predicted_label'] != label_val]
    if len(wrong) > 0:
        conf_col = 'proba_ai' if wrong.iloc[0]['predicted_label'] == 1 else 'proba_manusia'
        print(f"{wrong[conf_col].mean()*100:.2f}%")

print("\n" + "="*70)
print("EXPORT HASIL ANALISIS")

export_cols = ['text', 'label', 'predicted_label_str', 'proba_ai', 'proba_manusia', 'text_length', 'word_count']
df_export = df[export_cols].copy()
df_export.columns = ['text', 'true_label', 'predicted_label', 'proba_ai', 'proba_manusia', 'text_length', 'word_count']
df_export['is_correct'] = df_export['true_label'] == df_export['predicted_label']

df_export.to_csv(f'{OUT}/analysis_with_predictions.csv', index=False, encoding='utf-8')
print(f"\n[OK] Export: {OUT}/analysis_with_predictions.csv")

errors = df_export[df_export['is_correct'] == False]
errors.to_csv(f'{OUT}/analysis_errors.csv', index=False, encoding='utf-8')
print(f"[OK] Export: {OUT}/analysis_errors.csv ({len(errors)} errors)")

print("\n" + "="*70)
print("ANALISIS PER KATEGORI SELESAI!")

print("\n" + "="*70)
print("RINGKASAN")

total_correct = len(df[df['predicted_label'] == df['label_num']])
total_accuracy = total_correct / len(df) * 100

print(f"""
Total Data: {len(df)}
Total Benar: {total_correct}
Total Salah: {len(df) - total_correct}
Overall Accuracy: {total_accuracy:.2f}%

Per Label:
- MANUSIA: {len(df[df['label_num'] == 0])} data
- AI: {len(df[df['label_num'] == 1])} data
""")

# ======================================================
# VISUALISASI
# ======================================================
print("\nMembuat visualisasi per kategori...")
plt.style.use('seaborn-v0_8-whitegrid')

# GRAFIK 1: Akurasi per Label (LR, dengan boxplot confidence)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: Akurasi per label
label_colors = ['#3498db', '#e74c3c']
accs = []
for label_val, label_name in [(0, 'MANUSIA'), (1, 'AI')]:
    subset  = df[df['label_num'] == label_val]
    correct = subset[subset['predicted_label'] == label_val]
    accs.append(len(correct) / len(subset) * 100 if len(subset) > 0 else 0)

bars = axes[0].bar(['MANUSIA', 'AI'], accs, color=label_colors, alpha=0.85, edgecolor='white', width=0.5)
for bar, val in zip(bars, accs):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.2f}%', ha='center', fontsize=12, fontweight='bold')
axes[0].set_title('Akurasi per Label (LR)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Akurasi (%)', fontsize=12)
axes[0].set_ylim([90, 102])
axes[0].grid(axis='y', alpha=0.3)

# Panel 2: Confidence per label (boxplot — LR)
manusia_conf = df[df['label_num'] == 0]['proba_manusia'] * 100
ai_conf      = df[df['label_num'] == 1]['proba_ai'] * 100
bp = axes[1].boxplot([manusia_conf, ai_conf], labels=['MANUSIA', 'AI'],
                      patch_artist=True, widths=0.5)
for patch, color in zip(bp['boxes'], label_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[1].set_title('Distribusi Confidence per Label (LR)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Confidence (%)', fontsize=12)
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('Analisis Per Kategori — Logistic Regression', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/1_akurasi_dan_confidence_per_label.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT}/1_akurasi_dan_confidence_per_label.png")

# GRAFIK 2: Distribusi Panjang Teks per Label
fig, ax = plt.subplots(figsize=(12, 6))
manusia_len = df[df['label_num'] == 0]['word_count']
ai_len      = df[df['label_num'] == 1]['word_count']
ax.hist(manusia_len, bins=40, alpha=0.6, color='#3498db',
        label=f'Manusia (mean={manusia_len.mean():.0f})', edgecolor='white')
ax.hist(ai_len, bins=40, alpha=0.6, color='#e74c3c',
        label=f'AI (mean={ai_len.mean():.0f})', edgecolor='white')
ax.set_xlabel('Jumlah Kata', fontsize=12, fontweight='bold')
ax.set_ylabel('Frekuensi', fontsize=12, fontweight='bold')
ax.set_title('Distribusi Jumlah Kata — AI vs Manusia', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/2_distribusi_panjang_teks.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT}/2_distribusi_panjang_teks.png")

# GRAFIK 3: Perbandingan Akurasi per Label — LR vs SVM vs RF
print("\n[3] Perbandingan Akurasi per Label — 3 Model...")
model_names  = ['LR', 'SVM', 'RF']
model_colors = ['#2196F3', '#4CAF50', '#FF9800']
label_list   = [(0, 'MANUSIA'), (1, 'AI')]

# Hitung akurasi per label per model
acc_matrix = {}
for name, pipe in models_dict.items():
    preds_tmp = pipe.predict(X)
    row = {}
    for label_val, label_name in label_list:
        subset  = df[df['label_num'] == label_val]
        correct = (preds_tmp[df['label_num'] == label_val] == label_val).sum()
        row[label_name] = correct / len(subset) * 100
    acc_matrix[name] = row

x      = np.arange(len(label_list))
w      = 0.25
fig, ax = plt.subplots(figsize=(10, 6))

for i, (name, color) in enumerate(zip(model_names, model_colors)):
    vals = [acc_matrix[name][lname] for _, lname in label_list]
    bars = ax.bar(x + (i - 1) * w, vals, w, label=name, color=color, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')

ax.set_ylabel('Akurasi (%)', fontsize=12, fontweight='bold')
ax.set_title('Akurasi per Label — LR vs SVM vs RF\n(Full Dataset)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['MANUSIA', 'AI'], fontsize=12)
ax.set_ylim([80, 104])
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/3_akurasi_per_label_3model.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT}/3_akurasi_per_label_3model.png")

print(f"\nSemua grafik tersimpan di: {OUT}/")
print("  1_akurasi_dan_confidence_per_label.png -> Akurasi & confidence (LR)")
print("  2_distribusi_panjang_teks.png          -> Distribusi kata AI vs Manusia")
print("  3_akurasi_per_label_3model.png         -> Perbandingan LR, SVM, RF per label")
print("Analisis Per Kategori Selesai!")

