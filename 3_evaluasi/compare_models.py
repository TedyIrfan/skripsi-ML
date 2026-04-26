"""
EVALUASI 3 - Perbandingan Detail 4 Model
Membaca hasil langsung dari JSON — tidak melatih ulang model.
  - models_strict/strict_cv_results.json     → LR, SVM, RF
  - models_indobert/indobert_results.json    → IndoBERT
  - models_group_cv/group_cv_results.json    → Group CV
Output: hasil_phase5/3_perbandingan_model/
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUT = "hasil_phase5/3_perbandingan_model"
os.makedirs(OUT, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')

print("="*60)
print("EVALUASI 3 — PERBANDINGAN DETAIL 4 MODEL")
print(f"Output: {OUT}/")
print("="*60)

# ──────────────────────────────────────────────────────────
# LOAD HASIL DARI JSON (tidak melatih ulang model)
# ──────────────────────────────────────────────────────────
STRICT_JSON   = 'models_strict/strict_cv_results.json'
INDOBERT_JSON = 'models_indobert/indobert_results.json'
GROUP_JSON    = 'models_group_cv/group_cv_results.json'

print(f"\nLoad: {STRICT_JSON}")
with open(STRICT_JSON, 'r') as f:
    strict = json.load(f)

print(f"Load: {INDOBERT_JSON}")
with open(INDOBERT_JSON, 'r') as f:
    indobert = json.load(f)

test_res   = strict['test_results']
cv_res     = strict['cv_results']
test_total = strict['test_size']       # 300
n_pos = n_neg = test_total // 2        # 150 AI + 150 MANUSIA (dataset balanced)

# Helper: hitung TN/FP/FN/TP dari precision & recall
def derive_cm(prec, rec, n_pos=150, n_neg=150):
    tp = round(rec  * n_pos)
    fn = n_pos - tp
    fp = round(tp / prec - tp) if prec > 0 else 0
    tn = n_neg - fp
    return int(tn), int(fp), int(fn), int(tp)

# ── ML klasik (LR, SVM, RF) ──
ml_order  = ['Logistic Regression', 'SVM (RBF Kernel)', 'Random Forest']
ml_labels = ['LR', 'SVM', 'RF']

acc_ml  = [round(test_res[m]['test_accuracy']  * 100, 2) for m in ml_order]
prec_ml = [round(test_res[m]['test_precision'] * 100, 2) for m in ml_order]
rec_ml  = [round(test_res[m]['test_recall']    * 100, 2) for m in ml_order]
f1_ml   = [round(test_res[m]['test_f1']        * 100, 2) for m in ml_order]
auc_ml  = [round(test_res[m]['test_auc']       * 100, 2) for m in ml_order]
cv_ml   = [round(cv_res[m]['cv_mean']          * 100, 2) for m in ml_order]

cms_ml = [derive_cm(test_res[m]['test_precision'], test_res[m]['test_recall']) for m in ml_order]
tn_ml  = [c[0] for c in cms_ml]
fp_ml  = [c[1] for c in cms_ml]
fn_ml  = [c[2] for c in cms_ml]
tp_ml  = [c[3] for c in cms_ml]
fnr_ml = [round(c[2] / (c[2] + c[3]) * 100, 2) if (c[2] + c[3]) > 0 else 0.0 for c in cms_ml]

# ── IndoBERT ──
ib_acc  = round(indobert['test_accuracy']        * 100, 2)
ib_prec = round(indobert['test_precision']       * 100, 2)
ib_rec  = round(indobert['test_recall']          * 100, 2)
ib_f1   = round(indobert['test_f1']              * 100, 2)
ib_fnr  = round(indobert['false_negative_rate']  * 100, 2)
ib_cm   = indobert['confusion_matrix']
ib_tn, ib_fp, ib_fn, ib_tp = ib_cm['tn'], ib_cm['fp'], ib_cm['fn'], ib_cm['tp']

# ── Group CV (dari group_cv_results.json) ──
print(f"Load: {GROUP_JSON}")
with open(GROUP_JSON, 'r') as f:
    grp = json.load(f)['group_cv_results']

group_cv_ml = [round(grp[m]['group_cv_mean'] * 100, 2) for m in ml_order]

# ──────────────────────────────────────────────────────────
# BUILD DATAFRAME
# ──────────────────────────────────────────────────────────
results = {
    'Model':           ml_order          + ['IndoBERT'],
    'Accuracy (%)':    acc_ml            + [ib_acc],
    'Precision (%)':   prec_ml           + [ib_prec],
    'Recall (%)':      rec_ml            + [ib_rec],
    'F1-Score (%)':    f1_ml             + [ib_f1],
    'AUC-ROC (%)':     auc_ml            + [None],
    'FNR (%)':         fnr_ml            + [ib_fnr],
    'Standard CV (%)': cv_ml             + [None],
    'Group CV (%)':    group_cv_ml       + [None],
    'TN':              tn_ml             + [ib_tn],
    'FP':              fp_ml             + [ib_fp],
    'FN':              fn_ml             + [ib_fn],
    'TP':              tp_ml             + [ib_tp],
}
df = pd.DataFrame(results)

print(f"\nTabel Perbandingan Lengkap (Test Set = {test_total} data):")
print(df[['Model','Accuracy (%)','Precision (%)','Recall (%)','F1-Score (%)','FNR (%)']].to_string(index=False))

# ──────────────────────────────────────────────────────────
# GRAFIK 1: Heatmap Standard CV vs Group CV
# ──────────────────────────────────────────────────────────
print("\n[1] Heatmap Standard CV vs Group CV...")

cols_std = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'FNR (%)']
data_std = df.set_index('Model')[cols_std].apply(pd.to_numeric, errors='coerce')
data_cv  = df.set_index('Model')[['Standard CV (%)', 'Group CV (%)']].apply(pd.to_numeric, errors='coerce')

fig, axes = plt.subplots(1, 2, figsize=(17, 5))

sns.heatmap(data_std, annot=True, fmt='.2f', cmap='YlGn', ax=axes[0],
            linewidths=0.5, cbar_kws={'shrink': 0.8},
            annot_kws={'size': 11, 'weight': 'bold'})
axes[0].set_title('Metrik Test Set (Internal)\n— Standard Evaluation',
                  fontsize=12, fontweight='bold')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=20, ha='right', fontsize=9)
axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0, fontsize=9)

sns.heatmap(data_cv, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[1],
            linewidths=0.5, cbar_kws={'shrink': 0.8},
            annot_kws={'size': 13, 'weight': 'bold'},
            vmin=40, vmax=101)
axes[1].set_title('Standard CV vs Group CV\n(NaN = IndoBERT tidak pakai CV klasik)',
                  fontsize=12, fontweight='bold')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0, fontsize=11, fontweight='bold')
axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0, fontsize=9)

plt.suptitle(f'Heatmap Perbandingan Metrik — 4 Model (Test Set = {test_total} data)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/1_heatmap_metrics_lengkap.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT}/1_heatmap_metrics_lengkap.png")
print(f"       Panel kiri : Test Set (Standard Evaluation)")
print(f"       Panel kanan: Standard CV vs Group CV (warna merah = domain gap besar)")

# ──────────────────────────────────────────────────────────
# GRAFIK 2: Radar Chart
# ──────────────────────────────────────────────────────────
print("\n[2] Radar Chart semua model...")
categories = ['Accuracy','Precision','Recall','F1-Score']
N      = len(categories)
angles = [n/N * 2 * np.pi for n in range(N)]
angles += angles[:1]

model_vals = {
    'LR':       [acc_ml[0],  prec_ml[0],  rec_ml[0],  f1_ml[0]],
    'SVM':      [acc_ml[1],  prec_ml[1],  rec_ml[1],  f1_ml[1]],
    'RF':       [acc_ml[2],  prec_ml[2],  rec_ml[2],  f1_ml[2]],
    'IndoBERT': [ib_acc,     ib_prec,     ib_rec,     ib_f1],
}
colors_r = ['#2196F3','#4CAF50','#FF9800','#9C27B0']

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for (name, vals), color in zip(model_vals.items(), colors_r):
    vals_plot = vals + vals[:1]
    ax.plot(angles, vals_plot, 'o-', linewidth=2, color=color, label=name)
    ax.fill(angles, vals_plot, alpha=0.06, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
ax.set_ylim([90, 101])
ax.set_yticks([92, 94, 96, 98, 100])
ax.set_yticklabels(['92','94','96','98','100'], fontsize=8)
ax.set_title('Radar Chart — 4 Model', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/2_radar_chart_4models.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT}/2_radar_chart_4models.png")

# ──────────────────────────────────────────────────────────
# GRAFIK 3: FP vs FN Comparison
# ──────────────────────────────────────────────────────────
print("\n[3] False Positive vs False Negative Comparison...")
models_fp = ml_labels + ['IndoBERT']
fp_vals   = fp_ml + [ib_fp]
fn_vals   = fn_ml + [ib_fn]

x = np.arange(len(models_fp))
fig, ax = plt.subplots(figsize=(9, 6))
b1 = ax.bar(x-0.2, fp_vals, 0.38, label='False Positive\n(Manusia dikira AI)', color='#FF9800', alpha=0.85)
b2 = ax.bar(x+0.2, fn_vals, 0.38, label='False Negative\n(AI dikira Manusia)', color='#F44336', alpha=0.85)

for bar in b1:
    h = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, h+0.05,
            str(int(h)), ha='center', fontsize=14, fontweight='bold')
for bar in b2:
    h = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, h+0.05,
            str(int(h)), ha='center', fontsize=14, fontweight='bold', color='#c0392b')

ax.set_ylabel('Jumlah Kesalahan', fontsize=12, fontweight='bold')
ax.set_title(f'Perbandingan Kesalahan Klasifikasi — 4 Model\n(dari {test_total} data test | FN = teks AI yang lolos sebagai Manusia)',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_fp, fontsize=12)
ax.set_ylim([0, max(max(fp_vals), max(fn_vals)) + 3])
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.axhline(0, color='black', linewidth=0.8)

best_fnr = min(fnr_ml)
best_model_idx = fnr_ml.index(best_fnr)
note = f'IndoBERT: FP=0 (Precision 100%) | {ml_labels[best_model_idx]}: FNR terkecil ML klasik ({best_fnr:.2f}%)'
ax.text(0.5, 0.92, note,
        transform=ax.transAxes, ha='center', fontsize=9.5,
        color='darkblue', fontweight='bold',
        bbox=dict(facecolor='lightblue', alpha=0.4, boxstyle='round'))
plt.tight_layout()
plt.savefig(f'{OUT}/3_kesalahan_klasifikasi.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT}/3_kesalahan_klasifikasi.png")

print("\n" + "="*60)
print("EVALUASI 3 SELESAI!")
print("="*60)
print(f"\nFile tersimpan di: {OUT}/")
print("  1_heatmap_metrics_lengkap.png → Tabel metrik visual (test set + CV)")
print("  2_radar_chart_4models.png     → Radar chart 4 model")
print("  3_kesalahan_klasifikasi.png   → FP vs FN semua model")
print(f"\nSumber data:")
print(f"  {STRICT_JSON}")
print(f"  {INDOBERT_JSON}")
print(f"  {GROUP_JSON}")
