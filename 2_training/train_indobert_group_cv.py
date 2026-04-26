# Import library yang diperlukan
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import json
import os
import shutil

OUT_MODEL = "models_group_cv"
os.makedirs(OUT_MODEL, exist_ok=True)

print("=" * 60)
print("  IndoBERT Group CV — Domain Gap Analysis")
print("=" * 60)

# 1. Load Dataset
print("\nLoad dataset...")
df = pd.read_csv("dataset_clean_1500.csv", encoding='utf-8')
df = df.dropna(subset=['text', 'label'])
print(f"Total data: {len(df)}")

# 2. Deteksi Source Group
print("\nDetecting source groups...")
def detect_source(text):
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
    if len(text) > 500: return 'long_form'
    elif len(text) > 200: return 'medium_form'
    else: return 'short_form'

df['source_group'] = df['text'].apply(detect_source)
df['source_group'] = df.apply(lambda row: f"{row['label'].lower()}_{row['source_group']}", axis=1)

label_mapping = {'MANUSIA': 0, 'AI': 1}
df['label_num'] = df['label'].map(label_mapping)

X = df['text'].values
y = df['label_num'].values
groups = df['source_group'].values

# 3. Setup IndoBERT 
MODEL_NAME = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

# 4. GroupKFold Loop
n_groups = len(np.unique(groups))
n_splits = min(5, n_groups)
gkf = GroupKFold(n_splits=n_splits)

fold_scores = []
print(f"\nMulai melatih IndoBERT untuk {n_splits} folds...")

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
    print(f"\n{'='*40}\n FOLD {fold}\n{'='*40}")
    
    # Siapkan Data
    train_df = pd.DataFrame({'text': X[train_idx], 'label': y[train_idx]})
    test_df = pd.DataFrame({'text': X[test_idx], 'label': y[test_idx]})
    
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    test_tokenized = test_dataset.map(tokenize_function, batched=True)
    
    # Inisialisasi Model Baru per fold (Wajib supaya tidak bocor bobotnya)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Folder temporary untuk fold ini
    fold_dir = f"./tmp_indobert_fold{fold}"
    
    training_args = TrainingArguments(
        output_dir=fold_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",  # Diperbarui untuk transformers versi terbaru
        save_strategy="no",        # Jangan simpan ckpt per fold biar hemat storage
        logging_dir=f"{fold_dir}/logs",
        logging_steps=50,
        load_best_model_at_end=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        compute_metrics=compute_metrics,
    )
    
    # Latih model
    trainer.train()
    
    # Evaluasi test set (fold ini)
    eval_results = trainer.evaluate()
    acc = eval_results['eval_accuracy']
    fold_scores.append(float(acc))
    
    print(f"--> Akurasi Fold {fold}: {acc*100:.2f}%")
    
    # Bersihkan memori + folder temp
    del model
    del trainer
    torch.cuda.empty_cache()
    if os.path.exists(fold_dir):
        shutil.rmtree(fold_dir)

mean_score = float(np.mean(fold_scores))
std_score = float(np.std(fold_scores))

print("\n" + "=" * 60)
print(f"HASIL AKHIR INDOBERT GROUP CV (Mean {n_splits}-Fold): {mean_score*100:.2f}% ±{std_score*100:.2f}%")
print("=" * 60)

# 5. Simpan JSON Hasil
output_data = {
    "IndoBERT": {
        "fold_scores": fold_scores,
        "group_cv_mean": mean_score,
        "group_cv_std": std_score,
        "group_cv_min": float(min(fold_scores)),
        "group_cv_max": float(max(fold_scores))
    }
}

json_path = f"{OUT_MODEL}/indobert_group_cv_results.json"
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)
    
print(f"\n[OK] Hasil CSV/JSON IndoBERT tersimpan di {json_path}")
print("Selesai. Kamu bisa ambil file json ini untuk digabungkan ke komputermu.")
