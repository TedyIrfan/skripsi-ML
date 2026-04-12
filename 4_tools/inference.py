"""
Script Inference - Prediksi Teks Baru
Run langsung di terminal (bukan background)
"""

import joblib
import sys

print("="*60)
print("INFERENCE - PREDIKSI TEKS AI vs MANUSIA")
print("="*60)

# Load model
print("\n[1] Load model...")
try:
    pipeline = joblib.load('models_strict/best_pipeline_logistic_regression.pkl')
    # Label mapping sesuai training script
    label_mapping = {'MANUSIA': 0, 'AI': 1}
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    print("     [OK] Model loaded! (Logistic Regression Pipeline)")
except Exception as e:
    print(f"     [X] Error loading model: {e}")
    print("     Pastikan sudah menjalankan: python train_strict_cv.py")
    sys.exit(1)

# Interactive inference
print("\n" + "="*60)
print("MODE INTERAKTIF")
print("="*60)
print("\nKetik teks untuk prediksi")
print("Ketik 'exit' untuk keluar\n")

prediction_count = 0

while True:
    try:
        user_input = input(">>> ")

        if user_input.lower().strip() == 'exit':
            print("\nKeluar dari program. Terima kasih!")
            break

        if not user_input.strip():
            print("[!] Teks tidak boleh kosong!")
            continue

        # Predict langsung pake pipeline (sudah include TF-IDF di dalamnya)
        pred_num = pipeline.predict([user_input])[0]
        pred_proba = pipeline.predict_proba([user_input])[0]
        pred_label = reverse_label_mapping[pred_num]

        prediction_count += 1

        print(f"\n  Hasil Prediksi: {pred_label}")
        print(f"  Confidence: {pred_proba[pred_num]*100:.1f}%")

        # Status confidence
        if pred_proba[pred_num] > 0.8:
            print(f"  Status: TINGGI (Yakin)")
        elif pred_proba[pred_num] > 0.5:
            print(f"  Status: SEDANG")
        else:
            print(f"  Status: RENDAH (Kurang yakin)")

        # Detail probabilitas
        print(f"  Detail: P(AI)={pred_proba[1]*100:.1f}%, P(Manusia)={pred_proba[0]*100:.1f}%")
        print(f"  Total predictions: {prediction_count}")

    except KeyboardInterrupt:
        print("\n\nProgram dihentukan user.")
        break
    except Exception as e:
        print(f"\n[X] Error: {e}")

print("\n" + "="*60)
