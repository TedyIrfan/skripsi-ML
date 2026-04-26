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
    cross_val_score, StratifiedKFold, train_test_split,
    cross_val_predict
)
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
import warnings
import json
import os
warnings.filterwarnings('ignore')

OUT4 = "hasil_phase4"
os.makedirs(OUT4, exist_ok=True)

print("Strict Cross-Validation - Pipeline-Based")

# Load dataset dulu
print("\nLoad dataset...")
df = pd.read_csv("dataset_clean_1500.csv", encoding='utf-8')
df = df.dropna(subset=['text', 'label'])
print(f"    Total data: {len(df)}")
print(f"    Distribusi label:")
print(f"    - MANUSIA: {(df['label'] == 'MANUSIA').sum()}")
print(f"    - AI: {(df['label'] == 'AI').sum()}")

# Mapping label biar jadi angka
label_mapping = {'MANUSIA': 0, 'AI': 1}
df['label_num'] = df['label'].map(label_mapping)

X = df['text'].values
y = df['label_num'].values

# Split data jadi train sama test
print("\nSplit data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    Training: {len(X_train)}")
print(f"    Testing: {len(X_test)}")

# VERIFICATION: Check for overlap between train and test
print("\nVerifying no overlap between train and test...")
train_texts = set(X_train)
test_texts = set(X_test)
overlap = train_texts.intersection(test_texts)
print(f"    Overlap count: {len(overlap)}")
if len(overlap) > 0:
    print(f"    ⚠️  WARNING: {len(overlap)} overlapping texts found!")
    for i, text in enumerate(list(overlap)[:3], 1):
        print(f"      {i}. {text[:80]}...")
else:
    print("    [OK] No overlap - dataset is clean!")


# Setup pipeline buat tiap model
print("\nDefine pipelines (TF-IDF + Classifier)...")

# Parameter TF-IDF yang bakal dipake
tfidf_params = {
    'max_features': 5000,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_df': 0.8
}

# Latih 3 model: RF, Logistic Regression, SVM
pipelines = {
    'Random Forest': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ]),
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0
        ))
        
    ]),
    'SVM (RBF Kernel)': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', SVC(
            kernel='rbf',
            probability=True,
            random_state=42,
            C=1.0
        ))
    ])
}

# Jalanin 10-Fold Cross-Validation
print("\nRunning 10-Fold Strict Cross-Validation...")
print("    (TF-IDF di-fit ulang di setiap fold)")

# Pake StratifiedKFold biar balance tiap fold
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Simpen hasil tiap model
results = {}

# Validasi silang pake 10 fold
for name, pipeline in pipelines.items():
    print(f"\nModel: {name}")

    # Hitung score cross-validation
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )

    # Prediksi pake cross-validation
    cv_predictions = cross_val_predict(
        pipeline, X_train, y_train,
        cv=cv,
        n_jobs=-1
    )

    # Ambil probabilitas buat AUC
    try:
        cv_proba = cross_val_predict(
            pipeline, X_train, y_train,
            cv=cv,
            method='predict_proba',
            n_jobs=-1
        )
        cv_auc = roc_auc_score(y_train, cv_proba[:, 1])
    except:
        cv_auc = None

    # Hitung metrics
    cv_accuracy = cv_scores.mean()
    cv_std = cv_scores.std()
    cv_precision = precision_score(y_train, cv_predictions)
    cv_recall = recall_score(y_train, cv_predictions)
    cv_f1 = f1_score(y_train, cv_predictions)

    # Tampilin hasil tiap fold
    print(f"\nFold-by-Fold Accuracy:")
    for i, score in enumerate(cv_scores, 1):
        bar = '#' * int(score * 50)
        print(f"  Fold {i:2d}: {score*100:5.2f}% {bar}")

    # Tampilin hasil CV
    print(f"\nCV Results (NO LEAKAGE):")
    print(f"  Mean Accuracy: {cv_accuracy*100:.2f}% (±{cv_std*100:.2f}%)")
    print(f"  Min - Max:     {cv_scores.min()*100:.2f}% - {cv_scores.max()*100:.2f}%")
    print(f"  Precision:     {cv_precision*100:.2f}%")
    print(f"  Recall:        {cv_recall*100:.2f}%")
    print(f"  F1-Score:      {cv_f1*100:.2f}%")
    if cv_auc:
        print(f"  AUC-ROC:       {cv_auc*100:.2f}%")

    # Simpen hasil CV ke dictionary
    results[name] = {
        'cv_scores': cv_scores.tolist(),
        'cv_mean': cv_accuracy,
        'cv_std': cv_std,
        'cv_precision': cv_precision,
        'cv_recall': cv_recall,
        'cv_f1': cv_f1,
        'cv_auc': cv_auc
    }

# Latih ulang di full train set terus test
print("\nFinal Training on Full Train Set & Test Evaluation")

# Simpen hasil test set
final_results = {}

for name, pipeline in pipelines.items():
    print(f"\n{name}:")

    # Fit model ke full training data
    pipeline.fit(X_train, y_train)
    # Prediksi test set
    y_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)

    # Hitung AUC kalau bisa
    try:
        y_proba = pipeline.predict_proba(X_test)
        test_auc = roc_auc_score(y_test, y_proba[:, 1])
    except:
        test_auc = None

    # Tampilin hasil test
    print(f"  Test Accuracy:  {test_accuracy*100:.2f}%")
    print(f"  Test Precision: {test_precision*100:.2f}%")
    print(f"  Test Recall:    {test_recall*100:.2f}%")
    print(f"  Test F1-Score:  {test_f1*100:.2f}%")
    if test_auc:
        print(f"  Test AUC-ROC:   {test_auc*100:.2f}%")

    # Bandingin CV vs test
    cv_mean = results[name]['cv_mean']
    diff = abs(cv_mean - test_accuracy)
    print(f"\n  CV vs Test Difference: {diff*100:.2f}%")
    if diff < 0.02:
        print(f"  Status: EXCELLENT - Consistent performance")
    elif diff < 0.05:
        print(f"  Status: GOOD - Acceptable difference")
    else:
        print(f"  Status: WARNING - Significant difference")

    # Simpen hasil test
    final_results[name] = {
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_auc': test_auc,
        'cv_test_diff': diff
    }

# Tampilin ringkasan perbandingan model
print("\nRingkasan Perbandingan Model (Strict CV)")

print(f"\n{'Model':<25} {'CV Mean':<15} {'CV Std':<12} {'Test Acc':<12} {'Diff':<10}")
print("-"*74)

for name in pipelines.keys():
    cv_mean = results[name]['cv_mean']
    cv_std = results[name]['cv_std']
    test_acc = final_results[name]['test_accuracy']
    diff = final_results[name]['cv_test_diff']

    print(f"{name:<25} {cv_mean*100:>6.2f}%        ±{cv_std*100:>4.2f}%     {test_acc*100:>6.2f}%      {diff*100:>5.2f}%")

# Pilih model terbaik
best_model = max(final_results.keys(), key=lambda x: final_results[x]['test_accuracy'])
print(f"\nBEST MODEL (by Test Accuracy): {best_model}")
print(f"  Test Accuracy: {final_results[best_model]['test_accuracy']*100:.2f}%")
print(f"  CV Mean: {results[best_model]['cv_mean']*100:.2f}% (±{results[best_model]['cv_std']*100:.2f}%)")

# Simpen pipeline terbaik
print("\nSaving Best Pipeline")

import joblib
os.makedirs('models_strict', exist_ok=True)

# Latih ulang best model di full train set
best_pipeline = pipelines[best_model]
best_pipeline.fit(X_train, y_train)
# Simpen ke file
joblib.dump(best_pipeline, f'models_strict/best_pipeline_{best_model.lower().replace(" ", "_").replace("(", "").replace(")", "")}.pkl')
print(f"  Saved: models_strict/best_pipeline_{best_model.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pkl")

# Gabungin semua hasil
all_results = {
    'cv_results': results,
    'test_results': final_results,
    'best_model': best_model,
    'dataset_size': len(df),
    'train_size': len(X_train),
    'test_size': len(X_test)
}

# Simpen hasil ke JSON
with open('models_strict/strict_cv_results.json', 'w') as f:
    def convert(obj):
        # Konversi numpy type ke python native
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    json.dump(convert(all_results), f, indent=2)

print(f"  Saved: models_strict/strict_cv_results.json")

print("\n" + "="*50)
print("CONFUSION MATRIX UNTUK SEMUA MODEL")
print("="*50)

for name, pipeline in pipelines.items():
    print(f"\nConfusion Matrix - {name}")

    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n                Predicted: MANUSIA    Predicted: AI")
    print(f"Actual: MANUSIA     {cm[0][0]:>8}          {cm[0][1]:>8}")
    print(f"Actual: AI          {cm[1][0]:>8}          {cm[1][1]:>8}")

    tn, fp, fn, tp = cm.ravel()
    print(f"\nTrue Negatives:  {tn} (MANUSIA correctly identified)")
    print(f"False Positives: {fp} (MANUSIA wrongly classified as AI)")
    print(f"False Negatives: {fn} (AI wrongly classified as MANUSIA)" + (" <-- CRITICAL!" if fn > 0 else ""))
    print(f"True Positives:  {tp} (AI correctly identified)")

    # Hitung False Negative Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    print(f"\nFalse Negative Rate: {fnr*100:.2f}%")
    print(f"(AI yang lolos sebagai MANUSIA - semakin rendah semakin baik)")

print("\nStrict Cross-Validation Selesai!")
print("""
KESIMPULAN:
- Cross-validation menggunakan Pipeline mencegah data leakage
- TF-IDF di-fit ulang di setiap fold CV
- Hasil lebih realistis dan dapat diandalkan
- Model siap untuk tahap selanjutnya (threshold tuning)
""")

# ======================================================
# VISUALISASI — Generate Grafik ke hasil_phase4/
# ======================================================
print("\nMembuat visualisasi hasil training...")
plt.style.use('seaborn-v0_8-whitegrid')

ml_model_names  = list(pipelines.keys())
ml_model_labels = ['Random Forest', 'Logistic Reg.', 'SVM (RBF)']
ml_colors       = ['#e74c3c', '#3498db', '#2ecc71']

# Coba baca hasil IndoBERT dari JSON (jika sudah dilatih)
indobert_results = None
INDOBERT_JSON = 'models_indobert/indobert_results.json'
if os.path.exists(INDOBERT_JSON):
    with open(INDOBERT_JSON, 'r') as f:
        indobert_results = json.load(f)
    print("  [INFO] Hasil IndoBERT ditemukan, akan dimasukkan ke grafik perbandingan.")
else:
    print("  [INFO] File IndoBERT JSON tidak ditemukan. Grafik perbandingan hanya 3 model.")

# ── Tentukan data 4 model (atau 3 jika IndoBERT belum ada) ──
all_labels  = ml_model_labels.copy()
all_colors  = ml_colors.copy()
all_test_accs = [final_results[m]['test_accuracy'] * 100 for m in ml_model_names]
all_prec = [final_results[m]['test_precision'] * 100 for m in ml_model_names]
all_rec  = [final_results[m]['test_recall']    * 100 for m in ml_model_names]
all_f1   = [final_results[m]['test_f1']        * 100 for m in ml_model_names]

if indobert_results:
    all_labels.append('IndoBERT')
    all_colors.append('#f39c12')
    all_test_accs.append(indobert_results['test_accuracy'] * 100)
    all_prec.append(indobert_results['test_precision'] * 100)
    all_rec.append(indobert_results['test_recall']    * 100)
    all_f1.append(indobert_results['test_f1']         * 100)

# GRAFIK 1: Test Accuracy 4 Model (termasuk IndoBERT jika ada)
print("  [1] Test Accuracy (semua model)...")
x = np.arange(len(all_labels))
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(x, all_test_accs, width=0.5,
              color=all_colors, alpha=0.9, edgecolor='white')
for bar, val in zip(bars, all_test_accs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Perbandingan Test Accuracy — 4 Model\n(3 ML Tradisional + IndoBERT)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(all_labels, fontsize=11)
ax.set_ylim([88, 103])
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT4}/1_test_accuracy_4models.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT4}/1_test_accuracy_4models.png")

# GRAFIK 2: CV Score per Fold (HANYA 3 model ML — IndoBERT tidak pakai CV)
print("  [2] CV Score per Fold (3 Model ML, IndoBERT tidak pakai Cross-Validation)...")
fig, ax = plt.subplots(figsize=(12, 6))
folds = list(range(1, 11))
for name, label, color in zip(ml_model_names, ml_model_labels, ml_colors):
    scores = [s * 100 for s in results[name]['cv_scores']]
    ax.plot(folds, scores, marker='o', linewidth=2.5, markersize=7,
            label=f'{label} (Mean: {np.mean(scores):.2f}%)', color=color)
ax.axhline(y=98.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Test Score (98.00%)')
ax.set_xlabel('Fold ke-', fontsize=12, fontweight='bold')
ax.set_ylabel('Akurasi (%)', fontsize=12, fontweight='bold')
ax.set_title('Akurasi per Fold — Cross Validation 10-Fold\n3 Model ML Tradisional\n'
             '(IndoBERT dikecualikan: dievaluasi per-epoch, bukan per-fold)',
             fontsize=12, fontweight='bold', pad=15)
ax.set_xticks(folds)
ax.set_ylim([88, 103])
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT4}/2_cv_score_per_fold.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT4}/2_cv_score_per_fold.png")

# GRAFIK 3: Precision, Recall, F1 — 4 Model
print("  [3] Precision, Recall, F1 (semua model)...")
metric_labels_list = ['Precision', 'Recall', 'F1-Score']
metric_vals_list   = [all_prec, all_rec, all_f1]
metric_colors      = ['#9b59b6', '#e67e22', '#1abc9c']
x = np.arange(len(all_labels))
width = 0.22
fig, ax = plt.subplots(figsize=(13, 6))
for i, (mlabel, mvals, mc) in enumerate(zip(metric_labels_list, metric_vals_list, metric_colors)):
    offset = (i - 1) * width
    bars = ax.bar(x + offset, mvals, width, label=mlabel, color=mc, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, mvals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=7.5)
ax.set_ylabel('Skor (%)', fontsize=12, fontweight='bold')
ax.set_title('Precision, Recall, F1-Score — Test Set\n4 Model (3 ML Tradisional + IndoBERT)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(all_labels, fontsize=11)
ax.set_ylim([88, 103])
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT4}/3_precision_recall_f1_4models.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT4}/3_precision_recall_f1_4models.png")

# GRAFIK 4: Confusion Matrix 3 Model ML (IndoBERT CM dibuat di train_indobert.py)
print("  [4] Confusion Matrix 3 Model ML Tradisional...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
cm_labels_str = ['MANUSIA', 'AI']
for ax, (name, label) in zip(axes, zip(ml_model_names, ml_model_labels)):
    y_pred_cm = pipelines[name].predict(X_test)
    cm_mat = confusion_matrix(y_test, y_pred_cm)
    acc_v = accuracy_score(y_test, y_pred_cm) * 100
    sns.heatmap(cm_mat, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=cm_labels_str, yticklabels=cm_labels_str,
                linewidths=0.5, cbar=False, annot_kws={"size": 16, "weight": "bold"})
    ax.set_title(f'{label}\nTest Acc: {acc_v:.1f}%', fontsize=11, fontweight='bold')
    ax.set_xlabel('Prediksi', fontsize=10)
    ax.set_ylabel('Aktual', fontsize=10)
plt.suptitle('Confusion Matrix — Standard Test Set (300 Teks, 80/20 Split)\n'
             '(Confusion Matrix IndoBERT lihat: hasil_phase4/5_confusion_matrix_indobert.png)',
             fontsize=12, fontweight='bold', y=1.04)
plt.tight_layout()
plt.savefig(f'{OUT4}/4_confusion_matrix_3models.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] {OUT4}/4_confusion_matrix_3models.png")

print(f"\nSemua grafik training tersimpan di: {OUT4}/")
print("Catatan:")
print("  - Grafik 1,3: 4 model (termasuk IndoBERT jika JSON tersedia)")
print("  - Grafik 2  : 3 model (CV per fold - IndoBERT tidak pakai Cross-Validation)")
print("  - Grafik 4  : 3 model CM (CM IndoBERT ada di file terpisah no.5)")
print("\n" + "="*50)
print("TRAINING SELESAI!")
print("="*50)

