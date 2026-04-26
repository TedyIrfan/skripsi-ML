"""
EVALUASI 1 - Visualisasi Utama
Confusion Matrix, ROC Curve, CV Scores, Group CV, TF-IDF Features
untuk SEMUA 4 model (LR, SVM, RF, IndoBERT)
Output: hasil_phase5/1_visualisasi_utama/
"""
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

OUT = "hasil_phase5/1_visualisasi_utama"
os.makedirs(OUT, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')

print("="*60)
print("EVALUASI 1 — VISUALISASI UTAMA (4 MODEL)")
print(f"Output: {OUT}/")
print("="*60)

# ──────────────────────────────────────────────
# DATA BERSAMA
# ──────────────────────────────────────────────
df = pd.read_csv("dataset_clean_1500.csv", encoding='utf-8')
df = df.dropna()
df['label_num'] = df['label'].map({'MANUSIA': 0, 'AI': 1})
X = df['text']
y = df['label_num']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nDataset: Train={len(X_train)}, Test={len(X_test)}")

# Retrain semua 3 model klasik dari data (tidak load pkl)
print("\nMelatih ulang 3 Model klasik...")
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

tfidf_params = dict(max_features=5000, min_df=2, max_df=0.8, ngram_range=(1,2))

pipe_lr = Pipeline([
    ('tfidf', TfidfVectorizer(**tfidf_params)),
    ('clf',   LogisticRegression(max_iter=1000, random_state=42, C=1.0))
])
pipe_svm = Pipeline([
    ('tfidf', TfidfVectorizer(**tfidf_params)),
    ('clf',   SVC(C=1.0, kernel='rbf', probability=True, random_state=42))
])
pipe_rf = Pipeline([
    ('tfidf', TfidfVectorizer(**tfidf_params)),
    ('clf',   RandomForestClassifier(n_estimators=100, max_depth=20,
                                     min_samples_split=5, random_state=42, n_jobs=-1))
])
pipe_lr.fit(X_train, y_train)
pipe_svm.fit(X_train, y_train)
pipe_rf.fit(X_train, y_train)
print("  [OK] Logistic Regression")
print("  [OK] SVM (RBF Kernel)")
print("  [OK] Random Forest")

y_pred_lr    = pipe_lr.predict(X_test)
y_proba_lr   = pipe_lr.predict_proba(X_test)
y_pred_svm   = pipe_svm.predict(X_test)
y_proba_svm  = pipe_svm.predict_proba(X_test)
y_pred_rf    = pipe_rf.predict(X_test)
y_proba_rf   = pipe_rf.predict_proba(X_test)
print("  [OK] Semua prediksi siap")


# ──────────────────────────────────────────────
# GRAFIK 1: 4 CONFUSION MATRIX
# ──────────────────────────────────────────────
print("\n[1] 4 Confusion Matrices...")
from sklearn.metrics import confusion_matrix

# Sumber: strict_cv_results.json (LR, SVM, RF) & indobert_results.json (IndoBERT)
# Test set = 300 data (150 AI + 150 MANUSIA)
cms = {
    "Logistic Regression\n(98.00%)": (np.array([[148,2],[4,146]]), "Blues"),
    "SVM (RBF Kernel)\n(98.00%)":    (np.array([[148,2],[4,146]]), "Greens"),
    "Random Forest\n(98.00%)":       (np.array([[146,4],[2,148]]), "Oranges"),
    "IndoBERT\n(97.67%)":            (np.array([[150,0],[7,143]]), "Purples"),
}

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
for ax, (title, (cm, cmap)) in zip(axes.flatten(), cms.items()):
    tn, fp, fn, tp = cm.ravel()
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax, cbar=False,
                xticklabels=['Manusia','AI'], yticklabels=['Manusia','AI'],
                annot_kws={'size': 22, 'weight': 'bold'})
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Prediksi', fontsize=10)
    ax.set_ylabel('Aktual', fontsize=10)
    acc  = (tp+tn)/(tp+tn+fp+fn)*100
    prec = tp/(tp+fp)*100 if (tp+fp) > 0 else 0
    rec  = tp/(tp+fn)*100 if (tp+fn) > 0 else 0
    fnr  = fn/(fn+tp)*100 if (fn+tp) > 0 else 0
    info = f"Acc={acc:.2f}%  Prec={prec:.2f}%\nRecall={rec:.2f}%  FNR={fnr:.2f}%"
    ax.text(0.5,-0.22, info, transform=ax.transAxes, ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Confusion Matrix — 4 Model (Test Set = 300 data, 150 AI + 150 MANUSIA)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout(h_pad=3)
plt.savefig(f'{OUT}/1_all_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT}/1_all_confusion_matrices.png")

# ──────────────────────────────────────────────
# GRAFIK 2: ROC CURVE — 3 model klasik
# ──────────────────────────────────────────────
print("\n[2] ROC Curves (3 model klasik)...")
fig, ax = plt.subplots(figsize=(9, 7))
colors = {'Logistic Regression': '#2196F3',
          'SVM (RBF Kernel)':    '#4CAF50',
          'Random Forest':       '#FF9800'}

for (name, proba), color in zip([
    ('Logistic Regression', y_proba_lr),
    ('SVM (RBF Kernel)',    y_proba_svm),
    ('Random Forest',       y_proba_rf)
], colors.values()):
    fpr, tpr, _ = roc_curve(y_test, proba[:,1])
    roc_auc     = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, linewidth=2.5,
            label=f'{name} (AUC={roc_auc:.4f})')

ax.plot([0,1],[0,1],'r--', linewidth=1.5, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate',  fontsize=12, fontweight='bold')
ax.set_title('ROC Curve — 3 Model Klasik', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)
ax.text(0.6, 0.1, '* IndoBERT tidak ditampilkan\nkarena menggunakan framework\nterpisah (PyTorch+Transformers)',
        transform=ax.transAxes, fontsize=9, color='gray',
        bbox=dict(facecolor='lightyellow', alpha=0.6))
plt.tight_layout()
plt.savefig(f'{OUT}/2_roc_curve_3models.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT}/2_roc_curve_3models.png")

# ──────────────────────────────────────────────
# GRAFIK 3: PERBANDINGAN 4 MODEL (Bar Chart)
# ──────────────────────────────────────────────
print("\n[3] Model Comparison Bar Chart (4 model)...")
models = ['Logistic\nRegression', 'SVM\n(RBF)', 'Random\nForest', 'IndoBERT']
# Sumber: strict_cv_results.json & indobert_results.json
accuracy  = [98.00, 98.00, 98.00, 97.67]
precision = [99.32, 99.32, 97.37, 100.00]
recall    = [96.67, 96.67, 98.67,  95.33]
f1_scores = [97.97, 97.97, 98.01,  97.61]

x = np.arange(len(models))
w = 0.2
fig, ax = plt.subplots(figsize=(13, 7))
b1 = ax.bar(x-1.5*w, accuracy,   w, label='Accuracy',  color='#2196F3', alpha=0.85)
b2 = ax.bar(x-0.5*w, precision,  w, label='Precision', color='#4CAF50', alpha=0.85)
b3 = ax.bar(x+0.5*w, recall,     w, label='Recall',    color='#FF9800', alpha=0.85)
b4 = ax.bar(x+1.5*w, f1_scores,  w, label='F1-Score',  color='#9C27B0', alpha=0.85)

for bars in [b1,b2,b3,b4]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.1,
                f'{h:.1f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Perbandingan 4 Model — Semua Metrik Evaluasi\n(Test Set = 300 data | Sumber: strict_cv_results.json & indobert_results.json)',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim([93, 103])
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/3_model_comparison_all.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT}/3_model_comparison_all.png")

# ──────────────────────────────────────────────
# GRAFIK 4: CV SCORES PER FOLD (3 model klasik)
# ──────────────────────────────────────────────
print("\n[4] CV Scores per Fold...")
# Sumber: strict_cv_results.json — 10-Fold StratifiedKFold pada train set 1.200 data
cv_lr  = [1.0, 0.9667, 0.9583, 1.0, 0.975, 0.9583, 0.9667, 0.9917, 1.0, 0.95]
cv_svm = [1.0, 0.975,  0.9667, 1.0, 0.9833, 0.9583, 0.9833, 0.9833, 1.0, 0.9667]
cv_rf  = [0.9917, 0.9667, 0.95, 1.0, 0.9667, 0.925, 0.975, 0.9667, 0.975, 0.9333]
folds  = list(range(1, 11))

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(folds, [s*100 for s in cv_lr],  'o-', linewidth=2, color='#2196F3', label=f'LR  (mean={np.mean(cv_lr)*100:.2f}%)')
ax.plot(folds, [s*100 for s in cv_svm], 's-', linewidth=2, color='#4CAF50', label=f'SVM (mean={np.mean(cv_svm)*100:.2f}%)')
ax.plot(folds, [s*100 for s in cv_rf],  '^-', linewidth=2, color='#FF9800', label=f'RF  (mean={np.mean(cv_rf)*100:.2f}%)')
ax.axhline(np.mean(cv_lr)*100,  color='#2196F3', linestyle='--', alpha=0.4)
ax.axhline(np.mean(cv_rf)*100,  color='#FF9800', linestyle='--', alpha=0.4)
ax.set_xlabel('Fold ke-', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('10-Fold Stratified CV — 3 Model Klasik\n* IndoBERT tidak menggunakan CV (pakai val set)', fontsize=13, fontweight='bold')
ax.set_xticks(folds)
ax.set_ylim([95, 101.5])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/4_cv_scores_all_models.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT}/4_cv_scores_all_models.png")

# ──────────────────────────────────────────────
# GRAFIK 5: GROUP CV vs STANDARD CV (3 model klasik)
# ──────────────────────────────────────────────
print("\n[5] Group CV vs Standard CV...")
import json

GROUP_JSON = 'models_group_cv/group_cv_results.json'
with open(GROUP_JSON, 'r') as f:
    gcv_data = json.load(f)['group_cv_results']

# Mapping: nama tampilan → key di JSON
_key_map = {
    'LR':  'Logistic Regression',
    'SVM': 'SVM (RBF Kernel)',
    'RF':  'Random Forest',
}
models_g = ['LR', 'SVM', 'RF']
std_cv   = [gcv_data[_key_map[m]]['standard_cv_mean'] * 100 for m in models_g]
group_cv = [gcv_data[_key_map[m]]['group_cv_mean']    * 100 for m in models_g]
print(f"  Dibaca dari {GROUP_JSON}")
for m in models_g:
    k = _key_map[m]
    print(f"    {m}: std_cv={gcv_data[k]['standard_cv_mean']*100:.2f}%  group_cv={gcv_data[k]['group_cv_mean']*100:.2f}%  gap={gcv_data[k]['domain_gap']*100:.2f}%")

# [Dynamic] Tambahan IndoBERT jika file-nya ada
INDO_GROUP_JSON = 'models_group_cv/indobert_group_cv_results.json'
INDO_STD_JSON = 'models_indobert/indobert_results.json'
if os.path.exists(INDO_GROUP_JSON) and os.path.exists(INDO_STD_JSON):
    with open(INDO_GROUP_JSON, 'r') as f:
        indo_gcv = json.load(f)['IndoBERT']['group_cv_mean'] * 100
    with open(INDO_STD_JSON, 'r') as f:
        indo_std = json.load(f)['test_accuracy'] * 100
    models_g.append('IndoBERT')
    std_cv.append(indo_std)
    group_cv.append(indo_gcv)
    print(f"    IndoBERT: std_cv={indo_std:.2f}%  group_cv={indo_gcv:.2f}%  gap={(indo_gcv - indo_std):.2f}%")

x = np.arange(len(models_g))
fig, ax = plt.subplots(figsize=(9, 6))
b1 = ax.bar(x-0.2, std_cv,   0.38, label='Standard CV',       color='#2196F3', alpha=0.85)
b2 = ax.bar(x+0.2, group_cv, 0.38, label='Group CV (per sumber)', color='#F44336', alpha=0.85)
for bars in [b1,b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.25,
                f'{h:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title(f'Standard CV vs Group CV — Bukti Domain Gap\n(Membandingkan Performa Lintas Sumber Data - {len(models_g)} Model)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_g, fontsize=12)
ax.set_ylim([70, 104])
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/5_group_cv_vs_standard.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT}/5_group_cv_vs_standard.png")

# ──────────────────────────────────────────────
# GRAFIK 6: TF-IDF FEATURE IMPORTANCE (LR & RF)
# ──────────────────────────────────────────────
print("\n[6] TF-IDF Feature Importance (LR & RF)...")
tfidf_lr  = pipe_lr.named_steps['tfidf']
clf_lr    = pipe_lr.named_steps['clf']
tfidf_rf  = pipe_rf.named_steps['tfidf']
clf_rf    = pipe_rf.named_steps['clf']
feat_lr   = tfidf_lr.get_feature_names_out()
feat_rf   = tfidf_rf.get_feature_names_out()
top_n = 15

# LR: pakai koefisien
coef      = clf_lr.coef_[0]
top_ai_lr = np.argsort(coef)[-top_n:][::-1]
top_hu_lr = np.argsort(coef)[:top_n]

# RF: pakai feature_importances_
imps     = clf_rf.feature_importances_
top_rf   = np.argsort(imps)[-top_n:][::-1]

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# LR → AI
axes[0].barh(range(top_n), coef[top_ai_lr], color=plt.cm.Reds(np.linspace(0.4,0.9,top_n)))
axes[0].set_yticks(range(top_n))
axes[0].set_yticklabels([feat_lr[i] for i in top_ai_lr], fontsize=9)
axes[0].set_title(f'LR: Top {top_n} Fitur → AI', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Koefisien', fontsize=10)
axes[0].grid(axis='x', alpha=0.3)

# LR → Manusia
axes[1].barh(range(top_n), np.abs(coef[top_hu_lr]), color=plt.cm.Blues(np.linspace(0.4,0.9,top_n)))
axes[1].set_yticks(range(top_n))
axes[1].set_yticklabels([feat_lr[i] for i in top_hu_lr], fontsize=9)
axes[1].set_title(f'LR: Top {top_n} Fitur → Manusia', fontsize=11, fontweight='bold')
axes[1].set_xlabel('|Koefisien|', fontsize=10)
axes[1].grid(axis='x', alpha=0.3)

# RF → Top features
axes[2].barh(range(top_n), imps[top_rf]*100, color=plt.cm.Oranges(np.linspace(0.4,0.9,top_n)))
axes[2].set_yticks(range(top_n))
axes[2].set_yticklabels([feat_rf[i] for i in top_rf], fontsize=9)
axes[2].set_title(f'RF: Top {top_n} Feature Importance', fontsize=11, fontweight='bold')
axes[2].set_xlabel('Importance (%)', fontsize=10)
axes[2].grid(axis='x', alpha=0.3)

plt.suptitle('TF-IDF Feature Importance — LR (kiri & tengah) dan RF (kanan)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/6_tfidf_feature_importance_LR_RF.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT}/6_tfidf_feature_importance_LR_RF.png")

print("\n" + "="*60)
print("EVALUASI 1 SELESAI!")
print("="*60)
print(f"\n6 grafik tersimpan di: {OUT}/")
print("  1_all_confusion_matrices.png    → Semua 4 model")
print("  2_roc_curve_3models.png         → LR, SVM, RF")
print("  3_model_comparison_all.png      → Semua 4 model")
print("  4_cv_scores_all_models.png      → LR, SVM, RF")
print("  5_group_cv_vs_standard.png      → LR, SVM, RF")
print("  6_tfidf_feature_importance.png  → LR & RF")
