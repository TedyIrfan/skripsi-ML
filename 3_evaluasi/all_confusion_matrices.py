"""
all_confusion_matrices.py
Generate 4 confusion matrix (1 per model) dalam 1 gambar
Output: hasil_phase5/7_all_confusion_matrices.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("hasil_phase5", exist_ok=True)

# Data confusion matrix dari hasil training (sudah diverifikasi)
# Format: [[TN, FP], [FN, TP]]
models_data = {
    "Logistic Regression\n(Accuracy: 99.67%)": {
        "cm": np.array([[149, 1], [0, 152]]),
        "color": "Blues"
    },
    "SVM (RBF Kernel)\n(Accuracy: 99.67%)": {
        "cm": np.array([[149, 1], [0, 152]]),
        "color": "Greens"
    },
    "Random Forest\n(Accuracy: 98.34%)": {
        "cm": np.array([[145, 5], [0, 152]]),
        "color": "Oranges"
    },
    "IndoBERT\n(Accuracy: 97.35%)": {
        "cm": np.array([[142, 8], [0, 152]]),
        "color": "Purples"
    }
}

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

for idx, (title, data) in enumerate(models_data.items()):
    ax = axes[idx]
    cm = data["cm"]
    tn, fp, fn, tp = cm.ravel()

    acc  = (tp+tn)/(tp+tn+fp+fn)*100
    prec = tp/(tp+fp)*100 if (tp+fp) > 0 else 0
    rec  = tp/(tp+fn)*100 if (tp+fn) > 0 else 100
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    fnr  = fn/(fn+tp)*100

    sns.heatmap(cm, annot=True, fmt='d', cmap=data["color"],
                ax=ax, cbar=False,
                xticklabels=['Manusia', 'AI'],
                yticklabels=['Manusia', 'AI'],
                annot_kws={'size': 22, 'weight': 'bold'})

    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('Prediksi', fontsize=10)
    ax.set_ylabel('Aktual', fontsize=10)

    # Info box bawah
    info = f"Acc={acc:.2f}%  Prec={prec:.2f}%\nRecall={rec:.2f}%  F1={f1:.2f}%  FNR={fnr:.2f}%"
    ax.text(0.5, -0.22, info, transform=ax.transAxes,
            ha='center', fontsize=8.5,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Confusion Matrix — Perbandingan 4 Model\n(Test Set = 302 data)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout(h_pad=3)
plt.savefig("hasil_phase5/7_all_confusion_matrices.png", dpi=300, bbox_inches='tight')
plt.close()

print("[OK] hasil_phase5/7_all_confusion_matrices.png")
print("\nRingkasan:")
print(f"  LR  : TN=149, FP=1,  FN=0, TP=152  → FNR=0%")
print(f"  SVM : TN=149, FP=1,  FN=0, TP=152  → FNR=0%")
print(f"  RF  : TN=145, FP=5,  FN=0, TP=152  → FNR=0%")
print(f"  BERT: TN=142, FP=8,  FN=0, TP=152  → FNR=0%")
print("\nKesimpulan: Semua model FNR=0% (tidak ada AI yang lolos)")
print("IndoBERT punya FP paling banyak (8) = lebih konservatif")
