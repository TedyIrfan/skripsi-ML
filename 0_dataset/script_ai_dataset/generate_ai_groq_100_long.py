# =====================================================
# GENERATE 100 AI DATA (GROQ) - PANJANG (150-500 kata)
# =====================================================

import pandas as pd
import requests
import time
import random
import sys

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# GANTI DENGAN API KEY KAMU
API_KEY = "YOUR_GROQ_API_KEY_HERE"  # <- Masukkan API key Groq di sini

API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Topik beragam untuk generate text
TOPICS = [
    "Teknologi Artificial Intelligence dan dampaknya terhadap masa depan pekerjaan di Indonesia",
    "Perkembangan ekonomi digital dan startup di Indonesia dalam 5 tahun terakhir",
    "Pendidikan di era digital: tantangan dan peluang transformasi sistem pendidikan Indonesia",
    "Dampak perubahan iklim terhadap pertanian dan ketahanan pangan nasional",
    "Sejarah perjuangan kemerdekaan Indonesia dan relevansinya dengan generasi masa kini",
    "Budaya lokal Indonesia dan pelestarian warisan budaya di era globalisasi",
    "Sistem kesehatan nasional dan tantangan kesehatan masyarakat pasca pandemi",
    "Transportasi publik di Indonesia: masalah, solusi, dan rencana pembangunan",
    "Pariwisata Indonesia: potensi, tantangan, dan strategi pengembangan berkelanjutan",
    "Media sosial dan pengaruhnya terhadap perilaku generasi muda Indonesia",
    "Infrastruktur digital dan kesenjangan internet di daerah terpencil Indonesia",
    "Energi terbarukan dan transisi menuju Indonesia yang lebih hijau",
    "UMKM dan peran mereka dalam perekonomian nasional",
    "Sistem politik Indonesia: evaluasi demokrasi dan reformasi birokrasi",
    "Olahraga di Indonesia: prestasi, pembinaan atlet, dan infrastruktur",
    "Seni dan budaya kontemporer Indonesia: perkembangan dan apresiasi",
    "Urbanisasi dan permasalahan kota-kota besar di Indonesia",
    "Peran perempuan dalam pembangunan ekonomi dan sosial Indonesia",
    "Agrikultur modern dan teknologi pertanian di Indonesia",
    "Hubungan internasional Indonesia dan diplomasi di kancah global",
]

def generate_text(topic):
    """Generate text about topic with 150-500 words"""

    prompt = f"""Buatlah artikel bahasa Indonesia tentang "{topic}".

Syarat:
- Panjang teks: 150-500 kata
- Gaya penulisan formal dan informatif
- Struktur: pendahuluan, isi dengan 3-4 poin utama, kesimpulan
- Hindari pengulangan kalimat
- Berikan informasi yang relevan dan faktual

Tulis artikel sekarang:"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.8,
        "max_tokens": 2000,
    }

    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=60)
        response.raise_for_status()

        result = response.json()
        text = result['choices'][0]['message']['content'].strip()

        # Clean up common AI artifacts
        text = text.replace('**', '').replace('*', '')
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]

        return text

    except Exception as e:
        print(f"  [ERROR] {e}")
        return None

def count_words(text):
    return len(text.split())

print("="*60)
print("GENERATE 100 AI DATA (GROQ) - PANJANG")
print("="*60)

# Check API key
if API_KEY == "GROQ_API_KEY_ANDA_DISINI":
    print("\n[ERROR] Silakan masukkan API key Groq terlebih dahulu!")
    print("Edit file ini dan ganti GROQ_API_KEY_ANDA_DISINI dengan API key kamu")
    print("\nCara dapat API key:")
    print("1. Daftar di https://console.groq.com/")
    print("2. Buat API key di menu Keys")
    print("3. Copy paste ke file ini")
    exit()

# Check existing data
output_file = 'data_ai_groq_100.csv'
existing_count = 0
existing_data = []

try:
    df_existing = pd.read_csv(output_file)
    existing_count = len(df_existing)
    existing_data = df_existing.to_dict('records')
    print(f"\n[INFO] Existing data: {existing_count}")
except:
    print(f"\n[INFO] No existing data found")

TARGET = 100
needed = TARGET - existing_count

print(f"Target: {TARGET} AI data")
print(f"Panjang: 150-500 kata per teks")
print(f"Model: Llama 3.3 70B")

if needed <= 0:
    print(f"\n[OK] Sudah ada {existing_count} data. Target tercapai!")
    exit()

print(f"\nNeed: {needed} more data")
print("="*60)

results = existing_data  # Start with existing data

for i in range(needed):
    topic = random.choice(TOPICS)

    print(f"\n[{existing_count + i + 1}/{TARGET}] Generating: {topic[:50]}...")

    text = generate_text(topic)

    if text:
        word_count = count_words(text)

        # Check if meets criteria (150-500 words)
        if 150 <= word_count <= 500:
            results.append({
                'text': text,
                'label': 'AI',
                'source': 'Groq_Llama_3.3',
                'topic': topic,
                'word_count': word_count
            })
            print(f"  [OK] {word_count} words")
        else:
            print(f"  [SKIP] {word_count} words (need 150-500)")

    # Rate limiting - 1 request per second
    time.sleep(1)

print(f"\n{'='*60}")
print(f"GENERATION COMPLETE!")
print(f"{'='*60}")
new_generated = len(results) - existing_count
print(f"Existing: {existing_count}")
print(f"New: {new_generated}")
print(f"Total: {len(results)} data")

if len(results) > 0:
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    print(f"\n[OK] Saved: {output_file}")

    words = [r['word_count'] for r in results]
    print(f"\nStatistics:")
    print(f"  Words: min={min(words)}, max={max(words)}, avg={sum(words)/len(words):.0f}")

else:
    print("\n[WARNING] No data generated!")
