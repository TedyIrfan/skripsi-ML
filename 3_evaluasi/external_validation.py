"""
External Validation / Adversarial Testing Script
Menguji model tradisional ML melawan dataset adversarial (250 teks AI slang/typo).
"""
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

OUT = "hasil_phase5/7_adversarial_test"
os.makedirs(OUT, exist_ok=True)

print("="*70)
print("ADVERSARIAL TESTING — 3 MODEL ML vs 250 Teks AI Slang/Typo")
print("="*70)

# [1] Load & Latih Model dari Data Asli
print("\n[1] Melatih ulang 3 Model dari dataset_clean_1500.csv...")
df_train = pd.read_csv("dataset_clean_1500.csv", encoding='utf-8')
df_train = df_train.dropna(subset=['text', 'label'])
label_mapping = {'MANUSIA': 0, 'AI': 1}
X_train = df_train['text']
y_train = df_train['label'].map(label_mapping)

vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42)
}

trained_models = {}
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    trained_models[name] = model
print("    [OK] Semua model siap")

# [2] Load Adversarial Dataset
print("\n[2] Loading Adversarial Dataset (dataset_adversal.csv)...")
df_ext = pd.read_csv('dataset_adversal.csv', encoding='utf-8')
df_ext = df_ext.dropna(subset=['text', 'label'])
print(f"    [OK] Loaded {len(df_ext)} samples")
print(f"    Distribusi: AI={( df_ext['label']=='AI').sum()}, MANUSIA={(df_ext['label']=='MANUSIA').sum()}")

X_ext = df_ext['text']
y_ext = df_ext['label'].map(label_mapping)
X_ext_vec = vectorizer.transform(X_ext)

# [3] Evaluasi
print("\n[3] Menjalankan Adversarial Testing (3 ML + 1 Deep Learning)...")
results = []
cms = {}
for name, model in trained_models.items():
    y_pred = model.predict(X_ext_vec)
    acc = accuracy_score(y_ext, y_pred)
    cm = confusion_matrix(y_ext, y_pred, labels=[0, 1])
    cms[name] = cm
    tn, fp, fn, tp = cm.ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    results.append({'Model': name, 'Adversarial Accuracy': acc,
                    'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn, 'FNR': fnr})
    print(f"    [{name}]: Acc={acc*100:.2f}% | FN (AI lolos)={fn} | FNR={fnr*100:.1f}%")

# Inferensi IndoBERT
MODEL_PATH = "models_indobert/final_model"
if os.path.exists(MODEL_PATH):
    print("\n    [IndoBERT]: Menjalankan inferensi pada 250 teks (harap tunggu)...")
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    device = torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model_indo = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    model_indo.eval()
    
    texts = df_ext['text'].tolist()
    indobert_preds = []
    batch_size = 16
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt").to(device)
            outputs = model_indo(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            indobert_preds.extend(preds)
            
    cm_indo = confusion_matrix(y_ext, indobert_preds, labels=[0, 1])
    acc_indo = accuracy_score(y_ext, indobert_preds)
    cms['IndoBERT'] = cm_indo
    tn, fp, fn, tp = cm_indo.ravel()
    fnr_indo = fn / (fn + tp) if (fn + tp) > 0 else 0
    results.append({'Model': 'IndoBERT', 'Adversarial Accuracy': acc_indo,
                    'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn, 'FNR': fnr_indo})
    print(f"    [IndoBERT]: Acc={acc_indo*100:.2f}% | FN (AI lolos)={fn} | FNR={fnr_indo*100:.1f}%")

results_df = pd.DataFrame(results)

# [4] Tampilkan Tabel Ringkasan
print("\n" + "="*70)
print(f"{'Model':<25} {'Adv. Accuracy':<16} {'AI Lolos (FN)':<15} {'FNR':<10}")
print("-"*70)
for _, row in results_df.iterrows():
    print(f"{row['Model']:<25} {row['Adversarial Accuracy']*100:>14.2f}% {int(row['FN']):>10} teks {row['FNR']*100:>9.1f}%")
print("-"*70)

# Export CSV
results_df.to_csv(f'{OUT}/adversarial_results.csv', index=False)
print(f"\n[OK] adversarial_results.csv disimpan")

# ========== VISUALISASI ==========
print("\nMembuat visualisasi...")
plt.style.use('seaborn-v0_8-whitegrid')

# GRAFIK 1: Bar Chart Akurasi Adversarial vs Baseline
print("  [1] Bar Comparison Adversarial vs Baseline...")
baseline_acc = 98.0
model_names = results_df['Model'].tolist()
adv_accs = results_df['Adversarial Accuracy'].values * 100

x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, [baseline_acc]*len(model_names), width, label='Baseline (Standard Dataset)',
               color='#3498db', alpha=0.85, edgecolor='white')
bars2 = ax.bar(x + width/2, adv_accs, width, label='Adversarial (Slang/Typo)',
               color='#e74c3c', alpha=0.85, edgecolor='white')

for bar in bars1:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#c0392b')

ax.set_ylabel('Akurasi (%)', fontsize=12, fontweight='bold')
ax.set_title(f'Perbandingan Akurasi: Baseline vs Adversarial Testing\n({len(model_names)} Model Klasifikasi)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim([0, 105])
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/1_adversarial_vs_baseline.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT}/1_adversarial_vs_baseline.png")

# GRAFIK 2: Confusion Matrix untuk masing-masing model
print("  [2] Confusion Matrices...")
fig, axes = plt.subplots(1, len(cms), figsize=(4.5 * len(cms), 4))
if len(cms) == 1:
    axes = [axes]
    
for ax, (name, cm) in zip(axes, cms.items()):
    labels_str = ['MAN', 'AI']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels_str, yticklabels=labels_str,
                linewidths=0.5, cbar=False, annot_kws={"size": 14, "weight": "bold"})
    acc_val = results_df[results_df['Model'] == name]['Adversarial Accuracy'].values[0]
    ax.set_title(f'{name}\nAdv. Acc: {acc_val*100:.1f}%', fontsize=11, fontweight='bold')
    ax.set_xlabel('Prediksi', fontsize=10)
    ax.set_ylabel('Label Asli', fontsize=10)

plt.suptitle('Confusion Matrix — Adversarial Test (250 Teks AI Slang/Typo)',
             fontsize=13, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig(f'{OUT}/2_adversarial_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT}/2_adversarial_confusion_matrices.png")

print(f"\nSemua file tersimpan di: {OUT}/")
print("="*70)
print("ADVERSARIAL TESTING SELESAI!")
print("="*70)
