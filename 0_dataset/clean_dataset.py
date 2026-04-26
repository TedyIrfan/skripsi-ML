"""
clean_dataset.py — Membersihkan data AI dan Manusia (7 aturan + filter noise)
Output: data_ai_clean.csv, data_manusia_clean.csv
"""

import pandas as pd
import re
import os

INPUT_AI = "0_dataset/data_ai_all_clean.csv"
INPUT_MANUSIA = "0_dataset/data_manusia_all_clean.csv"
OUTPUT_AI = "0_dataset/data_ai_clean.csv"
OUTPUT_MANUSIA = "0_dataset/data_manusia_clean.csv"
MIN_WORDS = 15


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r'^.{0,120}?(?=:)', '', text, count=1)  # hapus teks pengantar
    text = re.sub(r'^.{0,120}?(?=https?://)', '', text, count=1)
    text = re.sub(r'https?://\S+', '', text)  # hapus URL
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)  # hapus mention
    text = re.sub(r'#\w+', '', text)  # hapus hashtag
    text = re.sub(r'<[^>]+>', '', text)  # hapus tag HTML
    text = text.encode('ascii', errors='ignore').decode('ascii')  # hapus emoji & non-ASCII
    text = re.sub(r"[^a-zA-Z0-9\s.,?!\"'\-]", '', text)  # hapus tanda baca tidak umum
    text = re.sub(r'[\n\t]+', ' ', text)  # rapikan newline
    text = re.sub(r'\s+', ' ', text)  # rapikan spasi berlebih
    return text.strip()


def filter_noise(df, min_words=MIN_WORDS):
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    before = len(df)
    df = df[df['word_count'] >= min_words].copy()
    print(f"  Filter noise: {before} → {len(df)} teks (dihapus: {before - len(df)})")
    return df


def main():
    print("=" * 60)
    print("  CLEANING DATASET — 7 Aturan + Filter Noise")
    print("=" * 60)

    print("\n[1/2] Cleaning Data AI...")
    if os.path.exists(INPUT_AI):
        df_ai = pd.read_csv(INPUT_AI, encoding='utf-8')
        print(f"  Input : {len(df_ai)} teks")
        df_ai['text'] = df_ai['text'].apply(clean_text)
        df_ai = df_ai[df_ai['text'].str.strip() != ''].copy()
        df_ai = filter_noise(df_ai)
        df_ai.to_csv(OUTPUT_AI, index=False, encoding='utf-8')
        print(f"  Output: {len(df_ai)} teks → {OUTPUT_AI}")
    else:
        print(f"  [SKIP] {INPUT_AI} tidak ditemukan.")

    print("\n[2/2] Cleaning Data Manusia...")
    if os.path.exists(INPUT_MANUSIA):
        df_manusia = pd.read_csv(INPUT_MANUSIA, encoding='utf-8')
        print(f"  Input : {len(df_manusia)} teks")
        df_manusia['text'] = df_manusia['text'].apply(clean_text)
        df_manusia = df_manusia[df_manusia['text'].str.strip() != ''].copy()
        df_manusia = filter_noise(df_manusia)
        df_manusia.to_csv(OUTPUT_MANUSIA, index=False, encoding='utf-8')
        print(f"  Output: {len(df_manusia)} teks → {OUTPUT_MANUSIA}")
    else:
        print(f"  [SKIP] {INPUT_MANUSIA} tidak ditemukan.")

    print("\n" + "=" * 60)
    print("  SELESAI!")
    print("=" * 60)


if __name__ == "__main__":
    main()
