"""
External Validation Script
Test model generalization on completely new data
"""
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import json

print("="*60)
print("EXTERNAL VALIDATION")
print("="*60)

# Load external data
print("\n[1] Loading external validation data...")
try:
    df_external = pd.read_csv('external_validation.csv', encoding='utf-8')
    print(f"    ✓ Loaded {len(df_external)} samples")
    print(f"    Label distribution:")
    print(f"      - MANUSIA: {(df_external['label'] == 'MANUSIA').sum()}")
    print(f"      - AI: {(df_external['label'] == 'AI').sum()}")
except FileNotFoundError:
    print("    ✗ File not found: external_validation.csv")
    print("    Please create external validation dataset first!")
    print("    Run: python merge_external_validation.py")
    exit()

# Prepare data
label_mapping = {'MANUSIA': 0, 'AI': 1}
X_external = df_external['text'].values
y_external = df_external['label'].map(label_mapping).values

# Load trained model
print("\n[2] Loading trained model...")
try:
    model = joblib.load('models_strict/best_pipeline_logistic_regression.pkl')
    print("    ✓ Model loaded: Logistic Regression")
except FileNotFoundError:
    print("    ✗ Model not found!")
    print("    Please train model first: python train_strict_cv.py")
    exit()

# Make predictions
print("\n[3] Making predictions...")
y_pred = model.predict(X_external)
y_pred_proba = model.predict_proba(X_external)
print("    ✓ Predictions completed")

# Calculate metrics
print("\n[4] Calculating metrics...")
accuracy = accuracy_score(y_external, y_pred)
precision = precision_score(y_external, y_pred)
recall = recall_score(y_external, y_pred)
f1 = f1_score(y_external, y_pred)
cm = confusion_matrix(y_external, y_pred)

# Print results
print("\n" + "="*60)
print("EXTERNAL VALIDATION RESULTS")
print("="*60)
print(f"\nOverall Metrics:")
print(f"  Accuracy:  {accuracy*100:.2f}%")
print(f"  Precision: {precision*100:.2f}%")
print(f"  Recall:    {recall*100:.2f}%")
print(f"  F1-Score:  {f1*100:.2f}%")

print(f"\nConfusion Matrix:")
print(f"  TN: {cm[0,0]:3d}  FP: {cm[0,1]:3d}")
print(f"  FN: {cm[1,0]:3d}  TP: {cm[1,1]:3d}")

# Per-source analysis
if 'source' in df_external.columns:
    print(f"\n" + "="*60)
    print("PER-SOURCE ANALYSIS")
    print("="*60)
    
    for source in df_external['source'].unique():
        df_source = df_external[df_external['source'] == source]
        X_source = df_source['text'].values
        y_source = df_source['label'].map(label_mapping).values
        y_pred_source = model.predict(X_source)
        
        acc_source = accuracy_score(y_source, y_pred_source)
        correct = (y_source == y_pred_source).sum()
        total = len(y_source)
        
        print(f"\n{source}:")
        print(f"  Samples: {total}")
        print(f"  Correct: {correct}/{total}")
        print(f"  Accuracy: {acc_source*100:.2f}%")

# Error analysis
print(f"\n" + "="*60)
print("ERROR ANALYSIS")
print("="*60)

errors = df_external.copy()
errors['predicted'] = ['MANUSIA' if p == 0 else 'AI' for p in y_pred]
errors['correct'] = errors['label'] == errors['predicted']
errors['confidence'] = [max(proba) for proba in y_pred_proba]

error_samples = errors[~errors['correct']]
print(f"\nTotal errors: {len(error_samples)}/{len(df_external)} ({len(error_samples)/len(df_external)*100:.1f}%)")

if len(error_samples) > 0:
    print("\nError breakdown:")
    false_positives = error_samples[error_samples['label'] == 'MANUSIA']
    false_negatives = error_samples[error_samples['label'] == 'AI']
    print(f"  False Positives (Human → AI): {len(false_positives)}")
    print(f"  False Negatives (AI → Human): {len(false_negatives)}")
    
    print("\nSample errors (first 5):")
    for idx, row in error_samples.head(5).iterrows():
        print(f"\n  [{idx+1}] Text: {row['text'][:100]}...")
        print(f"      True: {row['label']} | Predicted: {row['predicted']}")
        print(f"      Confidence: {row['confidence']*100:.1f}%")
        if 'source' in row:
            print(f"      Source: {row['source']}")
else:
    print("\n✓ Perfect predictions! No errors found.")

# Save results
print(f"\n[5] Saving results...")

results = {
    'dataset_info': {
        'total_samples': int(len(df_external)),
        'human_samples': int((df_external['label'] == 'MANUSIA').sum()),
        'ai_samples': int((df_external['label'] == 'AI').sum()),
    },
    'overall_metrics': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
    },
    'confusion_matrix': {
        'tn': int(cm[0,0]),
        'fp': int(cm[0,1]),
        'fn': int(cm[1,0]),
        'tp': int(cm[1,1]),
    },
    'error_rate': float(len(error_samples) / len(df_external)),
    'per_source': {}
}

if 'source' in df_external.columns:
    for source in df_external['source'].unique():
        df_source = df_external[df_external['source'] == source]
        X_source = df_source['text'].values
        y_source = df_source['label'].map(label_mapping).values
        y_pred_source = model.predict(X_source)
        acc_source = accuracy_score(y_source, y_pred_source)
        
        results['per_source'][source] = {
            'samples': int(len(df_source)),
            'accuracy': float(acc_source)
        }

with open('external_validation_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("    ✓ Saved to: external_validation_results.json")

# Save detailed results with predictions
errors.to_csv('external_validation_detailed.csv', index=False, encoding='utf-8')
print("    ✓ Saved to: external_validation_detailed.csv")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

if accuracy >= 0.90:
    print("\n✓ EXCELLENT: Model generalize sangat baik!")
    print("  Accuracy ≥ 90% menunjukkan model robust untuk real-world data.")
elif accuracy >= 0.85:
    print("\n✓ GOOD: Model generalize dengan baik!")
    print("  Accuracy 85-90% adalah hasil yang expected dan realistis.")
elif accuracy >= 0.80:
    print("\n⚠️  ACCEPTABLE: Model cukup baik.")
    print("  Ada room for improvement, tapi masih usable.")
else:
    print("\n⚠️  CONCERNING: Model mungkin overfit ke training data.")
    print("  Perlu review model atau tambah variasi training data.")

print("\n" + "="*60)
print("✓ External validation completed!")
print("="*60)
