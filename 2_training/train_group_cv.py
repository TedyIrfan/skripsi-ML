# Import library yang diperlukan
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GroupKFold, StratifiedKFold, cross_val_score
)
from sklearn.metrics import accuracy_score
import warnings
import json
import os
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# Setup folder output
# ──────────────────────────────────────────────
OUT4      = "hasil_phase4"
OUT_MODEL = "models_group_cv"
os.makedirs(OUT4,      exist_ok=True)
os.makedirs(OUT_MODEL, exist_ok=True)

print("=" * 60)
print("  Group CV — Domain Gap Analysis")
print("=" * 60)

# ──────────────────────────────────────────────
# Load dataset
# ──────────────────────────────────────────────
print("\nLoad dataset...")
df = pd.read_csv("dataset_clean_1500.csv", encoding='utf-8')
df = df.dropna(subset=['text', 'label'])
print(f"    Total data: {len(df)}")
print(f"    - AI     : {(df['label'] == 'AI').sum()}")
print(f"    - MANUSIA: {(df['label'] == 'MANUSIA').sum()}")

# ──────────────────────────────────────────────
# Deteksi source group
# ──────────────────────────────────────────────
print("\nDetecting source groups...")

def detect_source(text):
    """Deteksi sumber teks berdasarkan pola konten"""
    text = str(text).lower()

    if any(p in text[:100] for p in ['cnn indonesia', 'kompas', 'antara', 'republika', 'tribun']):
        return 'indosum_news'

    if '@' in text or text.count('#') >= 2:
        return 'twitter'

    formal_indicators = [
        'implementasi', 'optimalisasi', 'transformasi', 'peningkatan',
        'pengembangan', 'berkelanjutan', 'komprehensif', 'strategis'
    ]
    if sum(1 for ind in formal_indicators if ind in text) >= 3:
        return 'ai_formal'

    if len(text) > 500:
        return 'long_form'
    elif len(text) > 200:
        return 'medium_form'
    else:
        return 'short_form'

df['source_group'] = df['text'].apply(detect_source)
df['source_group'] = df.apply(
    lambda row: f"{row['label'].lower()}_{row['source_group']}", axis=1
)

print("\nDistribusi Source Groups:")
group_dist = df['source_group'].value_counts().to_dict()
for group, count in sorted(group_dist.items()):
    print(f"  {group}: {count}")

# ──────────────────────────────────────────────
# Prepare data
# ──────────────────────────────────────────────
label_mapping = {'MANUSIA': 0, 'AI': 1}
df['label_num'] = df['label'].map(label_mapping)

X      = df['text'].values
y      = df['label_num'].values
groups = df['source_group'].values

# ──────────────────────────────────────────────
# Setup pipelines
# ──────────────────────────────────────────────
tfidf_params = {
    'max_features': 5000,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_df': 0.8
}

model_names  = ['Random Forest', 'Logistic Regression', 'SVM (RBF Kernel)']
model_labels = ['Random Forest', 'Logistic Reg.', 'SVM (RBF)']
model_colors = ['#e74c3c', '#3498db', '#2ecc71']

pipelines = {
    'Random Forest': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
        ))
    ]),
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', LogisticRegression(max_iter=1000, random_state=42, C=1.0))
    ]),
    'SVM (RBF Kernel)': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', SVC(kernel='rbf', probability=True, random_state=42, C=1.0))
    ])
}

# ──────────────────────────────────────────────
# Group CV
# ──────────────────────────────────────────────
n_groups = len(np.unique(groups))
n_splits = min(5, n_groups)

print(f"\nJumlah unique groups : {n_groups}")
print(f"Jumlah fold          : {n_splits}")
print("\nRunning Group CV (GroupKFold)...")

gkf = GroupKFold(n_splits=n_splits)
group_results = {}

for name, pipeline in pipelines.items():
    print(f"\nModel: {name}")
    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        fold_scores.append(acc)

        train_g = set(groups[train_idx])
        test_g  = set(groups[test_idx])
        bar = '#' * int(acc * 40)
        print(f"  Fold {fold}: {acc*100:5.2f}%  {bar}")
        print(f"    Train groups: {len(train_g)}, Test groups: {len(test_g)}, "
              f"Train size: {len(X_tr)}, Test size: {len(X_te)}")

    mean_score = float(np.mean(fold_scores))
    std_score  = float(np.std(fold_scores))
    print(f"\n  Group CV Mean : {mean_score*100:.2f}% (±{std_score*100:.2f}%)")
    print(f"  Min - Max     : {min(fold_scores)*100:.2f}% - {max(fold_scores)*100:.2f}%")

    group_results[name] = {
        'fold_scores': [float(s) for s in fold_scores],
        'group_cv_mean': mean_score,
        'group_cv_std': std_score,
        'group_cv_min': float(min(fold_scores)),
        'group_cv_max': float(max(fold_scores)),
    }

# ──────────────────────────────────────────────
# Standard CV (pembanding)
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Standard CV (Pembanding — StratifiedKFold 5-Fold)")
print("=" * 60)

standard_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n{'Model':<25} {'Group CV':>12} {'Standard CV':>14} {'Gap (Domain)':>14}")
print("-" * 68)

for name, pipeline in pipelines.items():
    std_scores = cross_val_score(pipeline, X, y, cv=standard_cv,
                                 scoring='accuracy', n_jobs=-1)
    std_mean = float(std_scores.mean())
    std_std  = float(std_scores.std())
    grp_mean = group_results[name]['group_cv_mean']
    grp_std  = group_results[name]['group_cv_std']
    diff = grp_mean - std_mean

    print(f"{name:<25}  {grp_mean*100:5.2f}% ±{grp_std*100:4.2f}%   "
          f"{std_mean*100:5.2f}% ±{std_std*100:4.2f}%   {diff*100:+6.2f}%")

    group_results[name]['standard_cv_mean'] = std_mean
    group_results[name]['standard_cv_std']  = std_std
    group_results[name]['domain_gap']       = float(diff)

print("""
Interpretasi:
  Gap < 0  (Group CV < Standard CV) : Indikasi domain gap — model sensitif terhadap sumber data
  Gap ~ 0  (Group CV ≈ Standard CV) : Model generalisasi baik antar domain
  Gap > 0  (Group CV > Standard CV) : Jarang, bisa karena distribusi group menguntungkan
""")

# ──────────────────────────────────────────────
# Simpan JSON
# ──────────────────────────────────────────────
output_data = {
    'group_cv_results': group_results,
    'n_groups': n_groups,
    'n_splits': n_splits,
    'group_distribution': group_dist,
    'dataset_size': int(len(df)),
    'tfidf_params': tfidf_params,
}

json_path = f"{OUT_MODEL}/group_cv_results.json"
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)
print(f"\n[OK] Hasil disimpan: {json_path}")

# ──────────────────────────────────────────────
# VISUALISASI
# ──────────────────────────────────────────────
print("\nMembuat visualisasi Group CV...")
plt.style.use('seaborn-v0_8-whitegrid')

# ── GRAFIK 6: Group CV Mean vs Standard CV Mean — Grouped Bar ──
print("  [6] Group CV vs Standard CV Mean per Model...")

x     = np.arange(len(model_names))
width = 0.35

grp_means = [group_results[m]['group_cv_mean'] * 100 for m in model_names]
std_means = [group_results[m]['standard_cv_mean'] * 100 for m in model_names]
grp_stds  = [group_results[m]['group_cv_std'] * 100 for m in model_names]
std_stds  = [group_results[m]['standard_cv_std'] * 100 for m in model_names]

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, grp_means, width, label='Group CV (Domain Gap)',
               color='#e67e22', alpha=0.88, edgecolor='white',
               yerr=grp_stds, capsize=5)
bars2 = ax.bar(x + width/2, std_means, width, label='Standard CV (Stratified 5-Fold)',
               color='#3498db', alpha=0.88, edgecolor='white',
               yerr=std_stds, capsize=5)

for bar, val in zip(bars1, grp_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#e67e22')
for bar, val in zip(bars2, std_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2980b9')

ax.set_ylabel('Akurasi (%)', fontsize=12, fontweight='bold')
ax.set_title('Group CV vs Standard CV — Analisis Domain Gap\n'
             '(Group CV memisahkan sumber data antar fold untuk uji generalisasi)',
             fontsize=12, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=11)
ax.set_ylim([70, 105])
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT4}/6_group_cv_vs_standard_cv.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT4}/6_group_cv_vs_standard_cv.png")

# ── GRAFIK 7: Group CV Score per Fold — Line Chart ──
print("  [7] Group CV Score per Fold per Model...")

folds = list(range(1, n_splits + 1))
fig, ax = plt.subplots(figsize=(12, 6))

for name, label, color in zip(model_names, model_labels, model_colors):
    scores = [s * 100 for s in group_results[name]['fold_scores']]
    ax.plot(folds, scores, marker='o', linewidth=2.5, markersize=8,
            label=f'{label} (Mean: {np.mean(scores):.2f}%)', color=color)
    for i, (fold, score) in enumerate(zip(folds, scores)):
        ax.annotate(f'{score:.1f}%',
                    xy=(fold, score),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=8, color=color, fontweight='bold')

ax.set_xlabel('Fold ke-', fontsize=12, fontweight='bold')
ax.set_ylabel('Akurasi (%)', fontsize=12, fontweight='bold')
ax.set_title(f'Akurasi per Fold — Group CV ({n_splits}-Fold GroupKFold)\n'
             'Setiap fold memisahkan sumber data yang berbeda (uji domain gap)',
             fontsize=12, fontweight='bold', pad=15)
ax.set_xticks(folds)
ax.set_ylim([60, 110])
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT4}/7_group_cv_score_per_fold.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT4}/7_group_cv_score_per_fold.png")

# ── GRAFIK 8: Domain Gap — Horizontal Bar Chart ──
print("  [8] Domain Gap (Selisih Group CV vs Standard CV)...")

gaps = [group_results[m]['domain_gap'] * 100 for m in model_names]
colors_gap = ['#e74c3c' if g < 0 else '#27ae60' for g in gaps]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(model_labels, gaps, color=colors_gap, alpha=0.85, edgecolor='white', height=0.5)

for bar, val in zip(bars, gaps):
    ax.text(val + (0.3 if val >= 0 else -0.3),
            bar.get_y() + bar.get_height()/2,
            f'{val:+.2f}%', va='center',
            ha='left' if val >= 0 else 'right',
            fontsize=11, fontweight='bold')

ax.axvline(x=0, color='black', linewidth=1.5, linestyle='--', alpha=0.6)
ax.set_xlabel('Domain Gap = Group CV − Standard CV (%)', fontsize=11, fontweight='bold')
ax.set_title('Domain Gap per Model\n'
             '(Negatif → penurunan performa ketika sumber data dipisah)',
             fontsize=12, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT4}/8_domain_gap_per_model.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT4}/8_domain_gap_per_model.png")

# ──────────────────────────────────────────────
# Ringkasan akhir
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RINGKASAN AKHIR GROUP CV")
print("=" * 60)
print(f"\n{'Model':<25} {'Group CV':>12} {'Standard CV':>14} {'Domain Gap':>12}")
print("-" * 65)
for name, label in zip(model_names, model_labels):
    r = group_results[name]
    print(f"{name:<25}  {r['group_cv_mean']*100:6.2f}%         "
          f"{r['standard_cv_mean']*100:6.2f}%       {r['domain_gap']*100:+6.2f}%")

print(f"\n[OK] JSON tersimpan  : {OUT_MODEL}/group_cv_results.json")
print(f"[OK] Grafik tersimpan: {OUT4}/6_group_cv_vs_standard_cv.png")
print(f"                     : {OUT4}/7_group_cv_score_per_fold.png")
print(f"                     : {OUT4}/8_domain_gap_per_model.png")
print("\n" + "=" * 60)
print("  GROUP CV SELESAI!")
print("=" * 60)
