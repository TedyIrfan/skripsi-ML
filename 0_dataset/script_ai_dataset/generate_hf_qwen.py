"""
Generate AI Text dari HuggingFace API
Target: 200 data (100 casual + 100 formal)
Model: Qwen/Qwen3-Coder-Next
"""

from huggingface_hub import InferenceClient
import pandas as pd
import time
import re
from datetime import datetime

# Config
TOKEN = "YOUR_HF_TOKEN_HERE"
MODEL = "moonshotai/Kimi-K2.5"

client = InferenceClient(token=TOKEN)

# Topics
TOPICS = ["Sosial", "Budaya", "Politik", "Ekonomi", "Teknologi",
          "Kesehatan", "Olahraga", "Pendidikan", "Lingkungan", "Entertainment"]

# Prompts
PROMPT_CASUAL = """Buatkan 10 teks berbeda dalam Bahasa Indonesia dengan spesifikasi berikut:

GAYA: CASUAL
TOPIK: Bervariasi dari list ini [Sosial, Budaya, Politik, Ekonomi, Teknologi, Kesehatan, Olahraga, Pendidikan, Lingkungan, Entertainment]
PANJANG: 150-400 kata per teks
FORMAT: Paragraf natural, bukan list atau poin-poin

Tulis dengan gaya santai seperti ngobrol dengan teman. Gunakan:
- Bahasa sehari-hari
- Kata ganti "kamu", "aku", "kita"
- Boleh ada kontraksi (gak, emang, dll)
- Tone ringan dan friendly
- Boleh ada pengalaman pribadi

OUTPUT FORMAT:
Berikan dalam format:

---TEKS 1---
TOPIK: [nama topik]
[isi teks]

---TEKS 2---
TOPIK: [nama topik]
[isi teks]

... dan seterusnya sampai 10 teks.

Pastikan setiap teks tentang topik yang BERBEDA dari list di atas."""

PROMPT_FORMAL = """Buatkan 10 teks berbeda dalam Bahasa Indonesia dengan spesifikasi berikut:

GAYA: FORMAL
TOPIK: Bervariasi dari list ini [Sosial, Budaya, Politik, Ekonomi, Teknologi, Kesehatan, Olahraga, Pendidikan, Lingkungan, Entertainment]
PANJANG: 150-400 kata per teks
FORMAT: Paragraf natural, bukan list atau poin-poin

Tulis dengan gaya formal dan informatif. Gunakan:
- Bahasa baku dan resmi
- Kata ganti "Anda", "kita"
- Struktur kalimat lengkap
- Tone profesional dan objektif
- Berbasis fakta dan analisis

OUTPUT FORMAT:
Berikan dalam format:

---TEKS 1---
TOPIK: [nama topik]
[isi teks]

---TEKS 2---
TOPIK: [nama topik]
[isi teks]

... dan seterusnya sampai 10 teks.

Pastikan setiap teks tentang topik yang BERBEDA dari list di atas."""


def parse_generated_text(raw_text):
    """Parse output format dari model menjadi list teks"""
    texts = []

    pattern = r'---TEKS\s*(\d+)---'
    parts = re.split(pattern, raw_text)

    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            text_block = parts[i + 1]

            topic_match = re.search(r'TOPIK:\s*(.+?)(?:\n|$)', text_block, re.IGNORECASE)
            topic = topic_match.group(1).strip() if topic_match else "Unknown"

            content = re.sub(r'TOPIK:\s*.+?\n', '', text_block, flags=re.IGNORECASE)
            content = content.strip()

            content = re.sub(r'^---+', '', content)
            content = content.strip()

            if len(content) > 50:
                texts.append({
                    'topic': topic,
                    'text': content
                })

    return texts


def generate_batch(style, batch_num):
    """Generate satu batch (10 teks)"""

    print(f"\n{'='*60}")
    print(f"Generating Batch {batch_num} - Style: {style.upper()}")
    print(f"{'='*60}")

    prompt = PROMPT_CASUAL if style == "casual" else PROMPT_FORMAL

    max_retry = 3
    for attempt in range(max_retry):
        try:
            print(f"  Sending request... (attempt {attempt + 1})")

            response = client.chat_completion(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
                temperature=0.8
            )

            raw_text = response.choices[0].message.content

            print(f"  Response received! Length: {len(raw_text)} chars")

            texts = parse_generated_text(raw_text)

            if len(texts) >= 8:
                print(f"  Successfully parsed {len(texts)} texts!")

                for i, t in enumerate(texts[:3], 1):
                    preview = t['text'][:100] + "..." if len(t['text']) > 100 else t['text']
                    print(f"    [{i}] {t['topic']}: {preview}")

                if len(texts) > 3:
                    print(f"    ... dan {len(texts) - 3} lainnya")

                return texts

            else:
                print(f"  Only got {len(texts)} texts, retrying...")
                time.sleep(3)

        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(3)

    print(f"  Failed after {max_retry} attempts")
    return []


def main():
    """Main function untuk generate semua data"""

    print("\n" + "="*60)
    print("HUGGINGFACE AI TEXT GENERATOR")
    print(f"Model: {MODEL} (Kimi K2.5)")
    print(f"Target: 100 texts (50 casual + 50 formal)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    all_data = []

    # Generate 5 batch casual (50 texts)
    print("\n" + "#"*60)
    print("# GENERATING CASUAL TEXTS (5 batches x 10 = 50 texts)")
    print("#"*60)

    for batch in range(1, 6):
        texts = generate_batch("casual", batch)

        for t in texts:
            all_data.append({
                'text': t['text'],
                'label': 'AI',
                'model': 'HF-Kimi-K2.5',
                'style': 'Casual',
                'topic': t['topic'],
                'source': 'HuggingFace API',
                'batch': f"C{batch}",
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

        print(f"  Total collected so far: {len(all_data)}")
        time.sleep(2)

    # Generate 5 batch formal (50 texts)
    print("\n" + "#"*60)
    print("# GENERATING FORMAL TEXTS (5 batches x 10 = 50 texts)")
    print("#"*60)

    for batch in range(1, 6):
        texts = generate_batch("formal", batch)

        for t in texts:
            all_data.append({
                'text': t['text'],
                'label': 'AI',
                'model': 'HF-Kimi-K2.5',
                'style': 'Formal',
                'topic': t['topic'],
                'source': 'HuggingFace API',
                'batch': f"F{batch}",
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

        print(f"  Total collected so far: {len(all_data)}")
        time.sleep(2)

    # Save to CSV
    if all_data:
        df = pd.DataFrame(all_data)

        filename = f"data_ai_hf_qwen_{len(all_data)}.csv"
        df.to_csv(filename, index=False, encoding='utf-8')

        print("\n" + "="*60)
        print("GENERATION COMPLETED!")
        print("="*60)
        print(f"Total texts generated: {len(all_data)}")
        print(f"Saved to: {filename}")
        print(f"\nBreakdown:")
        print(f"  - Casual: {len([d for d in all_data if d['style'] == 'Casual'])} texts")
        print(f"  - Formal: {len([d for d in all_data if d['style'] == 'Formal'])} texts")
        print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return df
    else:
        print("\nNo data generated!")
        return None


if __name__ == "__main__":
    df = main()

    if df is not None:
        print("\nFirst 3 rows preview:")
        print(df[['text', 'label', 'style', 'topic']].head(3))
