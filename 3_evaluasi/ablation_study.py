# Import library yang diperlukan
import pandas as pd
import numpy as np
import joblib
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
from sklearn.metrics import accuracy_score
from sklearn.base import clone
import time
import warnings
warnings.filterwarnings('ignore')

OUT = "hasil_phase5/6_ablation_study"
os.makedirs(OUT, exist_ok=True)

print("ABLATION STUDY - FEATURE IMPORTANCE ANALYSIS UNTUK 3 MODEL")

# Load dataset
print("\nLoad dataset...")
df = pd.read_csv("dataset_clean_1500.csv", encoding='utf-8')
df = df.dropna(subset=['text', 'label'])

# Mapping label biar jadi angka
label_mapping = {'MANUSIA': 0, 'AI': 1}
df['label_num'] = df['label'].map(label_mapping)

# Siapin X dan y
X = df['text']
y = df['label_num']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"     Training: {len(X_train)}, Testing: {len(X_test)}")

# Setup Baseline Models list
# Parameter diambil dari nilai baseline di train_strict_cv.py
baseline_models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=1.0),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42, C=1.0)
}

print("\nMelatih baseline TF-IDF secara terpisah...")
# Gunakan parameter default TF-IDF dari train_strict_cv.py
vectorizer_full = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2)
)
X_train_full = vectorizer_full.fit_transform(X_train)
X_test_full = vectorizer_full.transform(X_test)
feature_names_full = vectorizer_full.get_feature_names_out()

# Simpan skor baseline untuk tiap model
baseline_accs = {}
model_instances = {}

for name, model in baseline_models.items():
    model_clone = clone(model)
    model_clone.fit(X_train_full, y_train)
    y_pred = model_clone.predict(X_test_full)
    baseline_accs[name] = accuracy_score(y_test, y_pred)
    model_instances[name] = model_clone
    print(f"     Baseline Accuracy ({name}): {baseline_accs[name]*100:.2f}%")

# Analisis Feature Importance menggunakan Random Forest sebagai acuan (karena RBF SVM sulit diakses koefisiennya)
print("\nMenarik Feature Importance (menggunakan bobot Random Forest sebagai acuan)...")
feature_importance = model_instances['Random Forest'].feature_importances_
top_indices = np.argsort(feature_importance)[-20:][::-1]

print("\n     Top 20 Most Important Features (Words) via Random Forest:")
print("-"*70)
print(f"{'Rank':<5} {'Feature':<20} {'Importance':<12} {'Cumulative':<12}")
print("-"*70)
cumulative = 0
for rank, idx in enumerate(top_indices, 1):
    imp = feature_importance[idx]
    feat = feature_names_full[idx]
    cumulative += imp
    print(f"{rank:<5} {feat:<20} {imp*100:>11.4f}% {cumulative*100:>11.4f}%")

# Mulai Ablation Study
print("\n" + "="*70)
print("ABLATION STUDY - PENGHAPUSAN FITUR UNTUK 3 MODEL ML")

ablation_results = []

def evaluate_models(config_name, desc, X_tr, X_te, fit_time=0):
    """ Helper function to run the 3 models on the specified X_train and X_test"""
    for name, model in baseline_models.items():
        st = time.time()
        mod = clone(model)
        mod.fit(X_tr, y_train)
        tt = time.time() - st
        
        y_p = mod.predict(X_te)
        acc = accuracy_score(y_test, y_p)
        diff = baseline_accs[name] - acc
        
        ablation_results.append({
            'Model': name,
            'Configuration': config_name,
            'Description': desc,
            'Accuracy': acc,
            'Difference': diff,
            'Features': X_tr.shape[1],
            'Train_Time': tt + fit_time
        })
        print(f"       [{name}] {acc*100:.2f}% (diff: {diff*100:+.2f}%)")

# Skenario 1: Kurangi jumlah max_features
print("\nSkenario 1: Mengurangi jumlah fitur (max_features)...")
for max_feat in [1000, 2000, 3000, 4000]:
    print(f"   Test: max_features={max_feat}")
    vec_temp = TfidfVectorizer(max_features=max_feat, min_df=2, max_df=0.8, ngram_range=(1, 2))
    st = time.time()
    X_tr_tmp = vec_temp.fit_transform(X_train)
    X_te_tmp = vec_temp.transform(X_test)
    fit_time = time.time() - st
    
    evaluate_models(f'max_features={max_feat}', f'{max_feat} fitur teratas', X_tr_tmp, X_te_tmp, fit_time)

# Skenario 2: Hanya unigram (tanpa bigram)
print("\nSkenario 2: Menghapus bigram (unigram only)...")
print("   Test: ngram_range=(1,1)")
vec_uni = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8, ngram_range=(1, 1))
st = time.time()
X_tr_uni = vec_uni.fit_transform(X_train)
X_te_uni = vec_uni.transform(X_test)
evaluate_models('unigram_only', 'Hanya unigram (tanpa bigram)', X_tr_uni, X_te_uni, time.time() - st)

# Skenario 3: Hapus Top N fitur terpenting
print("\nSkenario 3: Menghapus top N fitur paling krusial / pembeda...")
for remove_top in [10, 20, 50, 100]:
    print(f"   Test: Hapus Top {remove_top} kata terpenting")
    # Cari index yang harus dihapus berdasarkan importance full
    top_n_idx = np.argsort(feature_importance)[-remove_top:]
    keep_indices = [i for i in range(len(feature_names_full)) if i not in top_n_idx]
    
    X_tr_red = X_train_full[:, keep_indices]
    X_te_red = X_test_full[:, keep_indices]
    
    evaluate_models(f'remove_top_{remove_top}', f'Hapus top {remove_top} kata kunci AI vs Manusia', X_tr_red, X_te_red, 0)

# Tampilkan ringkasan
print("\n" + "="*70)
print("RINGKASAN HASIL ABLATION STUDY UNTUK KE-3 MODEL")
print("="*70)

ablation_df = pd.DataFrame(ablation_results)

for name in baseline_models.keys():
    print(f"\n>> Untuk Model: {name}")
    print("-"*70)
    print(f"{'Configuration':<20} {'Accuracy':<12} {'Drop (Loss)':<15} {'Features':<10}")
    print("-"*70)
    
    sub_df = ablation_df[ablation_df['Model'] == name].sort_values('Difference', ascending=False)
    for _, row in sub_df.iterrows():
        # Karena kita melihat "jatuhnya" performa, difference positif berarti akurasi turun
        drop_performa = -row['Difference']  
        print(f"{row['Configuration']:<20} {row['Accuracy']*100:>11.2f}% {drop_performa*100:>14.2f}% {row['Features']:>10}")
    print("-"*70)
    print(f"{'BASELINE':<20} {baseline_accs[name]*100:>11.2f}% {'0.00%':>15} {5000:>10}")
    print("-"*70)

# Cek persentase jatuh paling besar
max_drop = ablation_df.loc[ablation_df['Difference'].idxmax()]
print("\nKesimpulan Cepat:")
print(f"Titik paling lemah ada ketika kita menerapkan '{max_drop['Configuration']}'.")
print(f"Model [{max_drop['Model']}] jatuh paling parah dari baseline sebesar {-max_drop['Difference']*100:.2f}%.")

# Export CSV
ablation_df.to_csv(f'{OUT}/ablation_study_results.csv', index=False, encoding='utf-8')
print("\n[OK] Data lengkap tersimpan")

fi_df = pd.DataFrame({'feature': feature_names_full, 'importance': feature_importance})
fi_df = fi_df.sort_values('importance', ascending=False)
fi_df.to_csv(f'{OUT}/feature_importance.csv', index=False, encoding='utf-8')

# ========== VISUALISASI ==========
print("\nMembuat visualisasi...")
plt.style.use('seaborn-v0_8-whitegrid')

MODEL_COLORS = {
    'Random Forest': '#e74c3c',
    'Logistic Regression': '#3498db',
    'SVM (RBF)': '#2ecc71'
}

# ── GRAFIK 1: Top 20 Feature Importance ──
print("  [1] Feature Importance Top 20...")
top_20 = fi_df.head(20).iloc[::-1]  # dibalik biar yang paling penting ada di atas
fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(top_20['feature'], top_20['importance'] * 100,
               color='#e74c3c', alpha=0.85, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, top_20['importance'] * 100):
    ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}%', va='center', fontsize=8, color='#333')
ax.set_xlabel('Feature Importance (%)', fontsize=12, fontweight='bold')
ax.set_title('Top 20 Kata Paling Berpengaruh (TF-IDF + Random Forest)\n'
             'Sebagai Pembeda Teks AI vs Manusia', fontsize=13, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/1_feature_importance_top20.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT}/1_feature_importance_top20.png")

# ── GRAFIK 2: Ablation Study - Akurasi per Skenario per Model ──
print("  [2] Ablation Accuracy per Skenario...")
configs_order = ['max_features=1000', 'max_features=2000', 'max_features=3000', 'max_features=4000',
                 'unigram_only', 'remove_top_10', 'remove_top_20', 'remove_top_50', 'remove_top_100']
model_names = list(baseline_models.keys())
x = np.arange(len(configs_order))
width = 0.25

fig, ax = plt.subplots(figsize=(16, 7))
for i, name in enumerate(model_names):
    sub = ablation_df[ablation_df['Model'] == name].set_index('Configuration')
    accs = [sub.loc[c, 'Accuracy'] * 100 if c in sub.index else baseline_accs[name] * 100
            for c in configs_order]
    offset = (i - 1) * width
    bars = ax.bar(x + offset, accs, width, label=name,
                  color=MODEL_COLORS[name], alpha=0.85, edgecolor='white')

# Garis baseline
ax.axhline(y=98.0, color='black', linestyle='--', linewidth=1.5, label='Baseline (98.00%)', alpha=0.7)
ax.set_ylabel('Akurasi (%)', fontsize=12, fontweight='bold')
ax.set_title('Ablation Study — Akurasi 3 Model di Berbagai Konfigurasi Fitur TF-IDF',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(configs_order, rotation=30, ha='right', fontsize=9)
ax.set_ylim([88, 101])
ax.legend(fontsize=10, loc='lower left')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/2_ablation_accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT}/2_ablation_accuracy_comparison.png")

# ── GRAFIK 3: Drop Akurasi Heatmap ──
print("  [3] Heatmap Drop Akurasi...")
pivot_data = ablation_df.pivot_table(values='Difference', index='Model', columns='Configuration')
# Reorder columns
cols_exist = [c for c in configs_order if c in pivot_data.columns]
pivot_data = pivot_data[cols_exist]
# Difference positif = akurasi turun dari baseline → didisplay sebagai drop (kita negate)
pivot_drop = -pivot_data * 100

fig, ax = plt.subplots(figsize=(14, 4))
sns.heatmap(pivot_drop, annot=True, fmt='.2f', cmap='RdYlGn_r',
            center=0, linewidths=0.5, ax=ax,
            annot_kws={"size": 9},
            cbar_kws={'label': 'Penurunan Akurasi dari Baseline (%)'})
ax.set_title('Heatmap Penurunan Akurasi (Drop %) — Ablation Study 3 Model\n'
             'Merah = Turun banyak | Hijau = Naik/Stabil dari Baseline',
             fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Konfigurasi Fitur', fontsize=11)
ax.set_ylabel('Model', fontsize=11)
plt.xticks(rotation=30, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUT}/3_ablation_heatmap_drop.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT}/3_ablation_heatmap_drop.png")

print(f"\nSemua grafik tersimpan di: {OUT}/")
print("\n" + "="*70)
print("ABLATION STUDY SELESAI!")
print("="*70)
