"""
Generate AI Text dari Groq API
Target: 200 data (100 casual + 100 formal) dalam 10 batch
"""

import requests
import pandas as pd
import time
import re
from datetime import datetime

# Groq API Configuration
API_KEY = "YOUR_GROQ_API_KEY_HERE"
API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Available Groq models (pilih salah satu untuk start)
MODELS = {
    "llama3-3": "llama-3.3-70b-versatile",
    "llama3-1": "llama-3.1-70b-versatile",
    "mixtral": "mixtral-8x7b-32768",
    "gemma2": "gemma2-9b-it",
}

# Pilih model (ganti ini untuk ganti model)
MODEL = MODELS["llama3-3"]  # Default: Llama 3.3 70B

# Headers
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Topics list
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

    # Split by ---TEKS
    pattern = r'---TEKS\s*(\d+)---'
    parts = re.split(pattern, raw_text)

    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            text_block = parts[i + 1]

            # Extract topic and content
            topic_match = re.search(r'TOPIK:\s*(.+?)(?:\n|$)', text_block, re.IGNORECASE)
            topic = topic_match.group(1).strip() if topic_match else "Unknown"

            # Get content (remove topic line and clean up)
            content = re.sub(r'TOPIK:\s*.+?\n', '', text_block, flags=re.IGNORECASE)
            content = content.strip()

            # Clean up common artifacts
            content = re.sub(r'^---+', '', content)
            content = content.strip()

            if len(content) > 50:  # Minimum length check
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

    # System prompt berbeda per style
    if style == "casual":
        system_msg = "Kamu adalah content creator yang suka nulis dengan gaya santai dan friendly. Tulis dalam bahasa Indonesia yang natural."
    else:
        system_msg = "Kamu adalah penulis artikel profesional yang menulis dengan gaya formal dan informatif. Tulis dalam bahasa Indonesia baku."

    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 3000,
        "temperature": 0.8
    }

    max_retry = 3
    for attempt in range(max_retry):
        try:
            print(f"  Sending request... (attempt {attempt + 1})")
            response = requests.post(API_URL, headers=HEADERS, json=data, timeout=120)
            response.raise_for_status()

            result = response.json()
            raw_text = result["choices"][0]["message"]["content"]

            print(f"  Response received! Length: {len(raw_text)} chars")

            # Parse the output
            texts = parse_generated_text(raw_text)

            if len(texts) >= 8:  # Minimal 8 dari 10
                print(f"  Successfully parsed {len(texts)} texts!")

                # Show preview
                for i, t in enumerate(texts[:3], 1):
                    preview = t['text'][:100] + "..." if len(t['text']) > 100 else t['text']
                    print(f"    [{i}] {t['topic']}: {preview}")

                if len(texts) > 3:
                    print(f"    ... dan {len(texts) - 3} lainnya")

                return texts

            else:
                print(f"  Only got {len(texts)} texts, retrying...")
                time.sleep(2)

        except requests.exceptions.RequestException as e:
            print(f"  Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"  Response: {e.response.text[:200]}")
            time.sleep(3)

    print(f"  Failed after {max_retry} attempts")
    return []


def main():
    """Main function untuk generate semua data"""

    print("\n" + "="*60)
    print("GROQ AI TEXT GENERATOR")
    print(f"Model: {MODEL}")
    print(f"Target: 70 texts (35 casual + 35 formal)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    all_data = []

    # Generate 4 batch casual (40 texts)
    print("\n" + "#"*60)
    print("# GENERATING CASUAL TEXTS (4 batches x 10 = 40 texts)")
    print("#"*60)

    for batch in range(1, 5):
        texts = generate_batch("casual", batch)

        for t in texts:
            all_data.append({
                'text': t['text'],
                'label': 'AI',
                'model': 'Groq-Llama3.3-70B',
                'style': 'Casual',
                'topic': t['topic'],
                'source': 'Groq API',
                'batch': f"C{batch}",
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

        print(f"  Total collected so far: {len(all_data)}")
        time.sleep(1)  # Delay antar batch

    # Generate 4 batch formal (40 texts)
    print("\n" + "#"*60)
    print("# GENERATING FORMAL TEXTS (4 batches x 10 = 40 texts)")
    print("#"*60)

    for batch in range(1, 5):
        texts = generate_batch("formal", batch)

        for t in texts:
            all_data.append({
                'text': t['text'],
                'label': 'AI',
                'model': 'Groq-Llama3.3-70B',
                'style': 'Formal',
                'topic': t['topic'],
                'source': 'Groq API',
                'batch': f"F{batch}",
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

        print(f"  Total collected so far: {len(all_data)}")
        time.sleep(1)  # Delay antar batch

    # Save to CSV
    if all_data:
        df = pd.DataFrame(all_data)

        # Save main file
        filename = f"data_ai_groq_{len(all_data)}.csv"
        df.to_csv(filename, index=False, encoding='utf-8')

        print("\n" + "="*60)
        print("GENERATION COMPLETED!")
        print("="*60)
        print(f"Total texts generated: {len(all_data)}")
        print(f"Saved to: {filename}")
        print(f"\nBreakdown:")
        print(f"  - Casual: {len([d for d in all_data if d['style'] == 'Casual'])} texts")
        print(f"  - Formal: {len([d for d in all_data if d['style'] == 'Formal'])} texts")
        print(f"\nTopics distribution:")
        topic_counts = df['topic'].value_counts()
        for topic, count in topic_counts.items():
            print(f"  - {topic}: {count} texts")

        # Show summary
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
