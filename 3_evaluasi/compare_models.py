"""
EVALUASI 3 - Perbandingan Detail 4 Model
Tabel metrik lengkap + grafik perbandingan terperinci
Output: hasil_phase5/3_perbandingan_model/
"""
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

# Data lengkap semua 4 model
results = {
    'Model': ['Logistic Regression', 'SVM (RBF Kernel)', 'Random Forest', 'IndoBERT'],
    'Accuracy (%)':  [99.67, 99.67, 98.34, 97.35],
    'Precision (%)': [99.35, 99.35, 96.82, 95.00],
    'Recall (%)':    [100.00, 100.00, 100.00, 100.00],
    'F1-Score (%)':  [99.67, 99.67, 98.38, 97.44],
    'AUC-ROC (%)':   [99.99, 100.00, 99.86, None],
    'FNR (%)':       [0.00, 0.00, 0.00, 0.00],
    'Standard CV (%)': [99.75, 99.75, 98.59, None],
    'Group CV (%)':    [87.29, 96.74, 81.49, None],
    'TN': [149, 149, 145, 142],
    'FP': [1, 1, 5, 8],
    'FN': [0, 0, 0, 0],
    'TP': [152, 152, 152, 152],
}
df = pd.DataFrame(results)

print("\nTabel Perbandingan Lengkap:")
print(df.to_string(index=False))

# ── GRAFIK 1: Heatmap Standard CV vs Group CV ──
print("\n[1] Heatmap Standard CV vs Group CV...")

# Heatmap A: Test Set (Standard CV approach)
cols_std = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'FNR (%)']
data_std = df.set_index('Model')[cols_std].apply(pd.to_numeric, errors='coerce')

# Heatmap B: Standard CV mean vs Group CV mean (3 model klasik saja)
data_cv = df.set_index('Model')[['Standard CV (%)', 'Group CV (%)']].apply(pd.to_numeric, errors='coerce')

fig, axes = plt.subplots(1, 2, figsize=(17, 5))

# Panel kiri: Test Set Metrics
sns.heatmap(data_std, annot=True, fmt='.2f', cmap='YlGn', ax=axes[0],
            linewidths=0.5, cbar_kws={'shrink': 0.8},
            annot_kws={'size': 11, 'weight': 'bold'})
axes[0].set_title('Metrik Test Set (Internal)\n— Standard Evaluation',
                  fontsize=12, fontweight='bold')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=20, ha='right', fontsize=9)
axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0, fontsize=9)

# Panel kanan: Standard CV vs Group CV
sns.heatmap(data_cv, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[1],
            linewidths=0.5, cbar_kws={'shrink': 0.8},
            annot_kws={'size': 13, 'weight': 'bold'},
            vmin=75, vmax=101)
axes[1].set_title('Standard CV vs Group CV\n(NaN = IndoBERT tidak pakai CV klasik)',
                  fontsize=12, fontweight='bold')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0, fontsize=11, fontweight='bold')
axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0, fontsize=9)

plt.suptitle('Heatmap Perbandingan Metrik — 4 Model (Test Set = 302 data)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/1_heatmap_metrics_lengkap.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT}/1_heatmap_metrics_lengkap.png")
print(f"       Panel kiri : Test Set (Standard Evaluation)")
print(f"       Panel kanan: Standard CV vs Group CV (warna merah = domain gap)")


# ── GRAFIK 2: Radar Chart / Spider Chart ──
print("\n[2] Radar Chart semua model...")
categories = ['Accuracy','Precision','Recall','F1-Score']
N = len(categories)
angles = [n/N * 2 * np.pi for n in range(N)]
angles += angles[:1]

model_vals = {
    'LR':       [99.67, 99.35, 100.00, 99.67],
    'SVM':      [99.67, 99.35, 100.00, 99.67],
    'RF':       [98.34, 96.82, 100.00, 98.38],
    'IndoBERT': [97.35, 95.00, 100.00, 97.44],
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

# ── GRAFIK 3: FP Comparison (mana yang paling banyak salah) ──
print("\n[3] False Positive Comparison...")
models_fp = ['LR', 'SVM', 'RF', 'IndoBERT']
fp_vals   = [1, 1, 5, 8]
fn_vals   = [0, 0, 0, 0]

x = np.arange(len(models_fp))
fig, ax = plt.subplots(figsize=(9, 6))
b1 = ax.bar(x-0.2, fp_vals, 0.38, label='False Positive\n(Manusia dikira AI)', color='#FF9800', alpha=0.85)
b2 = ax.bar(x+0.2, fn_vals, 0.38, label='False Negative\n(AI dikira Manusia)', color='#F44336', alpha=0.85)

for bar in b1:
    h = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, h+0.05,
            str(int(h)), ha='center', fontsize=14, fontweight='bold')
for bar in b2:
    ax.text(bar.get_x()+bar.get_width()/2, 0.15,
            '0', ha='center', fontsize=14, fontweight='bold', color='#F44336')

ax.set_ylabel('Jumlah Kesalahan', fontsize=12, fontweight='bold')
ax.set_title('Perbandingan Kesalahan Klasifikasi — 4 Model\n(dari 302 data test)',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_fp, fontsize=12)
ax.set_ylim([0, 12])
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.axhline(0, color='black', linewidth=0.8)

# Catatan FNR
ax.text(0.5, 0.92, 'FNR = 0% untuk semua model (tidak ada AI yang lolos)',
        transform=ax.transAxes, ha='center', fontsize=10,
        color='darkgreen', fontweight='bold',
        bbox=dict(facecolor='lightgreen', alpha=0.4, boxstyle='round'))
plt.tight_layout()
plt.savefig(f'{OUT}/3_kesalahan_klasifikasi.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT}/3_kesalahan_klasifikasi.png")

print("\n" + "="*60)
print("EVALUASI 3 SELESAI!")
print("="*60)
print(f"\nFile tersimpan di: {OUT}/")
print("  1_heatmap_metrics.png       → Tabel metrik visual")
print("  2_radar_chart_4models.png   → Radar chart 4 model")
print("  3_kesalahan_klasifikasi.png → FP vs FN semua model")
