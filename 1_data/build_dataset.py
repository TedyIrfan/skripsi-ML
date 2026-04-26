"""
Script untuk menggabungkan data AI dan Manusia menjadi dataset final
Output: dataset_clean_1500.csv (1500 baris: 750 AI + 750 Manusia)
"""
import pandas as pd

print("=" * 55)
print("  BUILD DATASET FINAL")
print("=" * 55)

# ── 1. Load kedua file ──────────────────────────────────
print("\n[1/4] Load data...")
df_ai     = pd.read_csv('../0_dataset/data_ai_clean.csv', encoding='utf-8')
df_manusia = pd.read_csv('../0_dataset/data_manusia_clean.csv', encoding='utf-8')

print(f"  Data AI      : {len(df_ai)} teks")
print(f"  Data Manusia : {len(df_manusia)} teks")

# ── 2. Standardisasi: pastikan hanya kolom text & label ─
print("\n[2/4] Standardisasi kolom...")
df_ai     = df_ai[['text', 'label']].copy()
df_manusia = df_manusia[['text', 'label']].copy()

# ── 3. Gabungkan & shuffle ──────────────────────────────
print("\n[3/4] Gabungkan & shuffle...")
df_final = pd.concat([df_ai, df_manusia], ignore_index=True)
df_final = df_final.dropna(subset=['text', 'label'])
df_final = df_final.drop_duplicates(subset=['text'], keep='first')
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"  Total setelah digabung : {len(df_final)}")
print(f"  Distribusi label:")
print(f"    AI      : {(df_final['label'] == 'AI').sum()}")
print(f"    MANUSIA : {(df_final['label'] == 'MANUSIA').sum()}")

# ── 4. Verifikasi tidak ada overlap ─────────────────────
print("\n[4/4] Verifikasi...")
ai_texts    = set(df_final[df_final['label'] == 'AI']['text'].str.strip())
human_texts = set(df_final[df_final['label'] == 'MANUSIA']['text'].str.strip())
overlap     = ai_texts & human_texts
print(f"  Overlap AI-Manusia : {len(overlap)} (harus 0)")
if len(overlap) == 0:
    print("  OK - Tidak ada overlap!")
else:
    print(f"  WARNING: Ada {len(overlap)} teks yang overlap!")

# ── Simpan ──────────────────────────────────────────────
OUTPUT = 'dataset_clean_1500.csv'
df_final.to_csv(OUTPUT, index=False, encoding='utf-8')

print(f"\n{'=' * 55}")
print(f"  SELESAI! Tersimpan ke: {OUTPUT}")
print(f"  Total: {len(df_final)} teks")
print(f"{'=' * 55}")
