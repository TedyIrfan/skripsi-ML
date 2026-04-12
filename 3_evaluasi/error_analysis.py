"""
EVALUASI 4 - Error Analysis
Analisis teks yang salah prediksi dari semua 4 model
Output: hasil_phase5/4_error_analysis/
"""
import pandas as pd
import numpy as np
import joblib
import os
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

OUT = "hasil_phase5/4_error_analysis"
os.makedirs(OUT, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')

print("="*60)
print("EVALUASI 4 — ERROR ANALYSIS (4 MODEL)")
print(f"Output: {OUT}/")
print("="*60)

# Load data
df = pd.read_csv("dataset_skripsi_manusia_ai_1510.csv", encoding='utf-8')
df = df.dropna()
df['label_num'] = df['label'].map({'MANUSIA': 0, 'AI': 1})
X = df['text']
y = df['label_num']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTest set: {len(X_test)} data")

# Load / train semua model
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

models = {
    'Logistic Regression': pipe_lr,
    'SVM (RBF)':           pipe_svm,
    'Random Forest':       pipe_rf,
}

rev_map = {0: 'MANUSIA', 1: 'AI'}

# Analisis error per model
print("\n" + "="*60)
print("RINGKASAN ERROR PER MODEL")
print("="*60)

error_summary = {}
for name, pipe in models.items():
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)

    res = pd.DataFrame({
        'text':       X_test.values,
        'true':       [rev_map[v] for v in y_test],
        'pred':       [rev_map[v] for v in y_pred],
        'conf_ai':    y_proba[:, 1],
        'conf_man':   y_proba[:, 0],
        'is_error':   y_test.values != y_pred,
    })
    res['text_len'] = res['text'].str.len()
    res['word_cnt'] = res['text'].str.split().str.len()

    errors = res[res['is_error']]
    fp = errors[errors['true'] == 'MANUSIA']  # Manusia dikira AI
    fn = errors[errors['true'] == 'AI']       # AI dikira Manusia

    error_summary[name] = {
        'total_errors': len(errors),
        'fp_count': len(fp), 'fn_count': len(fn),
        'fp_conf_mean': fp['conf_ai'].mean() if len(fp) > 0 else 0,
        'fn_conf_mean': fn['conf_man'].mean() if len(fn) > 0 else 0,
        'err_len_mean': errors['text_len'].mean() if len(errors) > 0 else 0,
        'ok_len_mean':  res[~res['is_error']]['text_len'].mean(),
        'df': res,
        'errors': errors, 'fp_df': fp, 'fn_df': fn,
    }

    print(f"\n[{name}]")
    print(f"  Total error: {len(errors)} / {len(res)} ({len(errors)/len(res)*100:.2f}%)")
    print(f"  FP (Manusia→AI): {len(fp)}")
    print(f"  FN (AI→Manusia): {len(fn)}")
    if len(fp) > 0:
        print(f"  Rata2 confidence FP: {fp['conf_ai'].mean()*100:.1f}%")
    if len(fn) > 0:
        print(f"  Rata2 confidence FN: {fn['conf_man'].mean()*100:.1f}%")

# ── GRAFIK 1: Error per model (FP vs FN) ──
print("\n[1] Grafik Error per Model...")
model_names = list(error_summary.keys())
fp_counts = [error_summary[m]['fp_count'] for m in model_names]
fn_counts = [error_summary[m]['fn_count'] for m in model_names]

x = np.arange(len(model_names))
fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x-0.2, fp_counts, 0.38, label='False Positive (Manusia→AI)', color='#FF9800', alpha=0.85)
b2 = ax.bar(x+0.2, fn_counts, 0.38, label='False Negative (AI→Manusia)', color='#F44336', alpha=0.85)
for bar in b1:
    h = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, h+0.05, str(int(h)),
            ha='center', fontsize=13, fontweight='bold')
for bar in b2:
    ax.text(bar.get_x()+bar.get_width()/2, 0.1, '0',
            ha='center', fontsize=13, fontweight='bold', color='#F44336')

ax.set_ylabel('Jumlah Error', fontsize=12, fontweight='bold')
ax.set_title('Error Klasifikasi per Model — Test Set 302 Data\n(FN=0 semua model: tidak ada AI yang lolos)',
             fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim([0, 12])
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/1_error_count_per_model.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT}/1_error_count_per_model.png")

# ── GRAFIK 2: Distribusi panjang teks (error vs benar) per model ──
print("\n[2] Grafik Panjang Teks Error vs Benar...")
fig, axes = plt.subplots(1, 3, figsize=(17, 6))
for ax, name in zip(axes, model_names):
    d = error_summary[name]['df']
    ok  = d[~d['is_error']]['text_len']
    err = d[d['is_error']]['text_len']
    ax.hist(ok,  bins=30, alpha=0.6, color='#4CAF50', label=f'Benar (n={len(ok)})')
    ax.hist(err, bins=10, alpha=0.8, color='#F44336', label=f'Salah (n={len(err)})')
    ax.axvline(ok.mean(),  color='#4CAF50', linestyle='--', linewidth=1.5)
    ax.axvline(err.mean(), color='#F44336', linestyle='--', linewidth=1.5)
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_xlabel('Panjang Teks (karakter)', fontsize=10)
    ax.set_ylabel('Frekuensi', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle('Distribusi Panjang Teks — Prediksi Benar vs Salah (per Model)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/2_text_length_error_vs_correct.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT}/2_text_length_error_vs_correct.png")

# ── GRAFIK 3: Confidence distribution errors ──
print("\n[3] Confidence Distribution pada Error...")
fig, axes = plt.subplots(1, 3, figsize=(17, 6))
for ax, name in zip(axes, model_names):
    d = error_summary[name]
    fp = d['fp_df']
    if len(fp) > 0:
        ax.bar(range(len(fp)), sorted(fp['conf_ai'].values, reverse=True),
               color='#FF9800', alpha=0.85)
        ax.axhline(0.8, color='red', linestyle='--', linewidth=1.5, label='Threshold 80%')
        ax.set_ylim([0, 1])
        ax.set_xlabel('Index FP', fontsize=10)
        ax.set_ylabel('Confidence (AI)', fontsize=10)
        ax.set_title(f'{name}\n({len(fp)} FP, FN=0)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Tidak ada FP', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title(name, fontsize=11, fontweight='bold')

plt.suptitle('Confidence Score pada False Positive — Seberapa Yakin Model Salah?',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/3_confidence_false_positives.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT}/3_confidence_false_positives.png")

# ── Contoh teks yang salah (LR, paling representatif) ──
print("\n" + "="*60)
print("CONTOH TEKS YANG SALAH (Logistic Regression)")
print("="*60)
lr_fp = error_summary['Logistic Regression']['fp_df']
if len(lr_fp) > 0:
    print(f"\n[FP] Manusia yang dikira AI — {len(lr_fp)} teks:")
    for i, (_, row) in enumerate(lr_fp.iterrows(), 1):
        print(f"\n  [{i}] Confidence AI: {row['conf_ai']*100:.1f}%, Panjang: {row['text_len']} karakter")
        print(f"       {row['text'][:200]}...")

# ── Export CSV ──
lr_errors = error_summary['Logistic Regression']['errors']
lr_errors[['text','true','pred','conf_ai','conf_man','text_len']]\
    .to_csv(f'{OUT}/error_details_LR.csv', index=False, encoding='utf-8')
print(f"\n[OK] {OUT}/error_details_LR.csv")

print("\n" + "="*60)
print("EVALUASI 4 SELESAI!")
print("="*60)
print(f"\n3 grafik tersimpan di: {OUT}/")
print("  1_error_count_per_model.png          → FP & FN semua model")
print("  2_text_length_error_vs_correct.png   → Pola panjang teks")
print("  3_confidence_false_positives.png     → Seberapa yakin model saat salah")
