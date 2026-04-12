"""
Hyperparameter Tuning untuk Random Forest
Mencari parameter terbaik dengan GridSearchCV + Pipeline (anti data leakage)
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("="*60)
print("HYPERPARAMETER TUNING - RANDOM FOREST")
print("="*60)

# Load data
print("\n[1] Load dataset...")
df = pd.read_csv("dataset_skripsi_manusia_ai_1510.csv", encoding='utf-8')
df = df.dropna()
print(f"     Total data: {len(df)}")

# Preprocessing
print("\n[2] Preprocessing...")
label_mapping = {'MANUSIA': 0, 'AI': 1}
df['label_num'] = df['label'].map(label_mapping)

X = df['text']
y = df['label_num']

# Split
print("\n[3] Split data (sama persis dengan train_strict_cv.py)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"     Training: {len(X_train)}, Testing: {len(X_test)}")

# Pipeline (TF-IDF + RF dalam satu Pipeline agar tidak ada leakage)
print("\n[4] Setup Pipeline TF-IDF + Random Forest...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2)
    )),
    ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
])

# Parameter Grid — prefix 'clf__' karena ada di dalam Pipeline
print("\n[5] Definisikan Parameter Grid...")
param_grid = {
    'clf__n_estimators':     [50, 100, 200],
    'clf__max_depth':        [10, 20, None],
    'clf__min_samples_split':[2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4]
}

total_combinations = 3 * 3 * 3 * 3
print(f"     Total kombinasi: {total_combinations}")
print(f"     Total model dilatih: {total_combinations} x 5-fold = {total_combinations*5} model RF")
print(f"     Estimasi waktu: 10-30 menit")

# Grid Search
print("\n[6] Grid Search CV (5-Fold)...")
print("     Mohon tunggu... (ini yang terlama)")

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print("\n     Grid Search selesai!")

# Best Parameters
print("\n" + "="*60)
print("HASIL GRID SEARCH")
print("="*60)

print("\nBest Parameters:")
for param, value in grid_search.best_params_.items():
    clean_param = param.replace('clf__', '')
    print(f"  {clean_param}: {value}")

print(f"\nBest CV Score: {grid_search.best_score_*100:.2f}%")

# Evaluasi dengan model terbaik pada test set
print("\n[7] Evaluasi Model Terbaik pada Test Set...")

best_pipeline = grid_search.best_estimator_
y_pred = best_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

print(f"\nTest Accuracy:  {accuracy*100:.2f}%")
print(f"False Negative Rate: {fnr*100:.2f}%")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['MANUSIA', 'AI']))

# Confusion Matrix
print("\nConfusion Matrix:")
print(f"                   Pred: MANUSIA   Pred: AI")
print(f"Aktual MANUSIA        {tn:<6}           {fp:<6}")
print(f"Aktual AI             {fn:<6}           {tp:<6}")

# Compare dengan baseline RF dari train_strict_cv.py
print("\n" + "="*60)
print("PERBANDINGAN: BASELINE vs TUNED")
print("="*60)

baseline_acc = 0.9834  # hasil RF dari train_strict_cv.py

print(f"\nBaseline RF (dari train_strict_cv.py):")
print(f"  Accuracy: {baseline_acc*100:.2f}%")
print(f"  Params  : n_estimators=100, max_depth=20, min_samples_split=5 (default)")

print(f"\nTuned RF (setelah GridSearch):")
print(f"  Accuracy: {accuracy*100:.2f}%")
best_params_clean = {k.replace('clf__',''): v for k, v in grid_search.best_params_.items()}
print(f"  Params  : {best_params_clean}")

diff = accuracy - baseline_acc
if diff > 0:
    print(f"\n  [+] Improvement: +{diff*100:.2f}% dari baseline")
elif diff < 0:
    print(f"\n  [-] Tuned lebih rendah: {diff*100:.2f}% dari baseline")
else:
    print(f"\n  [=] Tidak ada perubahan dari baseline")

# Simpan hasil ke file
print("\n[8] Menyimpan hasil...")
os.makedirs("models_strict", exist_ok=True)

results = {
    "best_params": best_params_clean,
    "best_cv_score": round(grid_search.best_score_ * 100, 2),
    "test_accuracy": round(accuracy * 100, 2),
    "fnr": round(fnr * 100, 2),
    "baseline_accuracy": round(baseline_acc * 100, 2),
    "improvement": round(diff * 100, 2),
    "confusion_matrix": {
        "TN": int(tn), "FP": int(fp),
        "FN": int(fn), "TP": int(tp)
    }
}

with open("models_strict/hyperparameter_tuning_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("     Saved: models_strict/hyperparameter_tuning_results.json")

print("\n" + "="*60)
print("HYPERPARAMETER TUNING SELESAI!")
print("="*60)
