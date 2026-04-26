"""
Uji Inferensi Langsung — IndoBERT
Script ini menguji 3 teks nyata pada model IndoBERT yang sudah terlatih.
Hasil disimpan ke: indobert_inference_results.json
"""

import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ╔══════════════════════════════════════════════════════════════╗
# ║           EDIT 3 TEKS DI SINI SEBELUM DIJALANKAN            ║
# ╚══════════════════════════════════════════════════════════════╝

# Teks 1 — Manusia bergaya sangat formal/baku (edge case: berpotensi dikira AI)
TEKS_1 = """
Pada awalnya bangsa Israel hanya terdiri dari satu kelompok keluarga di antara banyak kelompok keluarga yang hidup di tanah Kan'an pada abad 18 SM.
"""

# Teks 2 — Teks deskriptif/ulasan natural khas tulisan manusia
TEKS_2 = """
warung ini dimiliki oleh pengusaha pabrik tahu yang sudah puluhan tahun terkenal membuat tahu putih di bandung. tahu berkualitas, dipadu keahlian memasak, dipadu kretivitas, jadilah warung yang menyajikan menu utama berbahan tahu, ditambah menu umum lain seperti ayam. semuanya selera indonesia. harga cukup terjangkau. jangan lewatkan tahu bletoka nya, tidak kalah dengan yang asli dari tegal!
"""

# Teks 3 — Manusia bergaya formal akademik/ensiklopedia (edge case)
TEKS_3 = """
Salah satu tekniknya adalah periplus, deskripsi pada pelabuhan dan daratan sepanjang garis pantai yang bisa dilihat pelaut di lepas pantai; contoh pertamanya adalah Hanno sang Navigator dari Carthaginia dan satu lagi dari Laut Erythraea, keduanya selamat di laut menggunakan teknik periplus dengan mengenali garis pantai laut Merah dan Teluk Persi.
"""

# Label yang benar untuk masing-masing teks
LABEL_BENAR_1 = "MANUSIA"
LABEL_BENAR_2 = "MANUSIA"
LABEL_BENAR_3 = "MANUSIA"

# Deskripsi singkat untuk tabel ringkasan
DESKRIPSI_1 = "Manusia formal/baku (edge case — berpotensi dikira AI)"
DESKRIPSI_2 = "Manusia deskriptif/ulasan natural (harusnya benar)"
DESKRIPSI_3 = "Manusia formal akademik/ensiklopedia (edge case)"

# ══════════════════════════════════════════════════════════════
# Mulai eksekusi — tidak perlu edit di bawah ini
# ══════════════════════════════════════════════════════════════

print("=" * 60)
print("UJI INFERENSI LANGSUNG — IndoBERT")
print("=" * 60)

# ── Cek dependensi ──────────────────────────────────────────
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
    print(f"\n[OK] PyTorch versi: {torch.__version__}")
    print(f"[OK] Transformers tersedia")
except ImportError as e:
    print(f"\n[ERROR] Dependensi tidak tersedia: {e}")
    print("Install: pip install torch transformers")
    sys.exit(1)

# ── Setup device ────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[OK] Device: {device}")

# ── Path model ──────────────────────────────────────────────
MODEL_PATH = "models_indobert/final_model"
MAX_LENGTH = 128

if not os.path.exists(MODEL_PATH):
    print(f"\n[ERROR] Model tidak ditemukan: {MODEL_PATH}")
    print("Pastikan sudah menjalankan: python 2_training/train_indobert.py")
    sys.exit(1)

# ── Load model & tokenizer ──────────────────────────────────
print(f"\n[1] Load model dari: {MODEL_PATH} ...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    print("[OK] Model dan tokenizer berhasil dimuat!")
    print(f"     Parameter: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"[ERROR] Gagal load model: {e}")
    sys.exit(1)

# ── Fungsi prediksi ─────────────────────────────────────────
def predict(text):
    """Prediksi teks dengan IndoBERT, kembalikan label + confidence."""
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    prob_manusia = probs[0][0].item()
    prob_ai      = probs[0][1].item()
    pred_label   = "AI" if prob_ai > 0.5 else "MANUSIA"
    confidence   = prob_ai if pred_label == "AI" else prob_manusia

    return {
        "label_prediksi": pred_label,
        "confidence_pct": round(confidence * 100, 1),
        "prob_ai_pct":    round(prob_ai * 100, 1),
        "prob_manusia_pct": round(prob_manusia * 100, 1),
    }

# ── 3 Teks Uji (diambil dari variabel di atas) ──────────────
test_cases = [
    {
        "id": 1,
        "deskripsi": DESKRIPSI_1,
        "label_benar": LABEL_BENAR_1,
        "teks": TEKS_1.strip(),
    },
    {
        "id": 2,
        "deskripsi": DESKRIPSI_2,
        "label_benar": LABEL_BENAR_2,
        "teks": TEKS_2.strip(),
    },
    {
        "id": 3,
        "deskripsi": DESKRIPSI_3,
        "label_benar": LABEL_BENAR_3,
        "teks": TEKS_3.strip(),
    },
]

# ── Jalankan semua prediksi ──────────────────────────────────
print("\n" + "=" * 60)
print("[2] MENJALANKAN PREDIKSI 3 TEKS ...")
print("=" * 60)

results = []

for case in test_cases:
    print(f"\n┌─ TES {case['id']}: {case['deskripsi']}")
    print(f"│  Label Benar  : {case['label_benar']}")
    print(f"│  Teks (50 chr): {case['teks'][:80]}...")

    pred = predict(case["teks"])

    is_correct = pred["label_prediksi"] == case["label_benar"]
    status = "✅ BENAR" if is_correct else "❌ SALAH"

    print(f"│")
    print(f"│  Prediksi     : {pred['label_prediksi']}  ({pred['confidence_pct']}%)")
    print(f"│  P(AI)        : {pred['prob_ai_pct']}%")
    print(f"│  P(MANUSIA)   : {pred['prob_manusia_pct']}%")
    print(f"│  Status       : {status}")
    print(f"└{'─' * 58}")

    results.append({
        **case,
        **pred,
        "is_correct": is_correct,
        "status": status,
    })

# ── Tabel ringkasan ─────────────────────────────────────────
print("\n" + "=" * 60)
print("TABEL RINGKASAN (untuk catatan_bab4.md)")
print("=" * 60)
print(f"\n{'Tes':<5} {'Deskripsi':<35} {'Label Benar':<14} {'Prediksi':<20} {'Status':<8}")
print("-" * 85)
for r in results:
    pred_str = f"{r['label_prediksi']} ({r['confidence_pct']}%)"
    print(f"{r['id']:<5} {r['deskripsi']:<35} {r['label_benar']:<14} {pred_str:<20} {r['status']}")

# ── Hitung akurasi inferensi ─────────────────────────────────
n_correct = sum(1 for r in results if r["is_correct"])
print(f"\nAkurasi 3 teks uji: {n_correct}/3 ({n_correct/3*100:.1f}%)")

# ── Simpan hasil ke JSON ─────────────────────────────────────
output = {
    "model": "IndoBERT (indobenchmark/indobert-base-p1)",
    "model_path": MODEL_PATH,
    "device": str(device),
    "max_length": MAX_LENGTH,
    "total_teks_uji": len(results),
    "akurasi_inferensi": f"{n_correct}/{len(results)}",
    "hasil": [
        {
            "id": r["id"],
            "deskripsi": r["deskripsi"],
            "teks": r["teks"],
            "label_benar": r["label_benar"],
            "prediksi": r["label_prediksi"],
            "confidence_pct": r["confidence_pct"],
            "prob_ai_pct": r["prob_ai_pct"],
            "prob_manusia_pct": r["prob_manusia_pct"],
            "is_correct": r["is_correct"],
        }
        for r in results
    ],
}

# ── Simpan JSON ke folder output ───────────────────────────────────────
OUT_IMG = "hasil_phase5/8_indobert_inference"
os.makedirs(OUT_IMG, exist_ok=True)

OUTPUT_FILE = f"{OUT_IMG}/indobert_inference_results.json"
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n[OK] Hasil disimpan ke: {OUTPUT_FILE}")

# ── Visualisasi ────────────────────────────────────────────────────────
print(f"\nMembuat visualisasi...")
plt.style.use('seaborn-v0_8-whitegrid')

labels_teks   = [f"Teks {r['id']}\n({r['deskripsi'][:28]}...)" for r in results]
prob_ai       = [r['prob_ai_pct'] for r in results]
prob_manusia  = [r['prob_manusia_pct'] for r in results]
is_correct    = [r['is_correct'] for r in results]
prediksi      = [r['label_prediksi'] for r in results]

x     = np.arange(len(results))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars_ai = ax.bar(x - width/2, prob_ai, width,
                 label='P(AI)', color='#e74c3c', alpha=0.85, edgecolor='white')
bars_mn = ax.bar(x + width/2, prob_manusia, width,
                 label='P(MANUSIA)', color='#3498db', alpha=0.85, edgecolor='white')

# Tambah angka di atas bar
for bar, val in zip(bars_ai, prob_ai):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9,
            fontweight='bold', color='#c0392b')
for bar, val in zip(bars_mn, prob_manusia):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9,
            fontweight='bold', color='#2980b9')

# Tambah label prediksi + status di bawah tiap grup
for i, (pred, correct) in enumerate(zip(prediksi, is_correct)):
    status_sym = '✅' if correct else '❌'
    ax.text(x[i], -8, f"Prediksi: {pred} {status_sym}",
            ha='center', va='top', fontsize=9, fontweight='bold',
            color='#27ae60' if correct else '#e74c3c')

ax.axhline(y=50, color='gray', linestyle='--', linewidth=1.2, alpha=0.6, label='Threshold 50%')
ax.set_ylabel('Probabilitas (%)', fontsize=12, fontweight='bold')
ax.set_title('Inferensi Langsung IndoBERT — 3 Teks Uji (Tulisan Manusia)\n'
             'P(AI) vs P(MANUSIA) per Teks | Label Benar: Semua MANUSIA',
             fontsize=12, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels_teks, fontsize=9)
ax.set_ylim([-15, 110])
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
out_path = f'{OUT_IMG}/1_indobert_inferensi_3teks.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Grafik disimpan: {out_path}")

print("\n" + "=" * 60)
print("SELESAI — Gunakan angka di tabel di atas untuk update catatan_bab4.md")
print("=" * 60)
