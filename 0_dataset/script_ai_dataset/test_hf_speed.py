"""
Test 1 batch HuggingFace Qwen3 - Speed Test
Bandingkan dengan OpenRouter dan Groq
"""

from huggingface_hub import InferenceClient
import time

TOKEN = "YOUR_HF_TOKEN_HERE"
MODEL = "Qwen/Qwen3-Coder-Next"

client = InferenceClient(token=TOKEN)

print("="*60)
print(f"SPEED TEST: {MODEL}")
print("="*60)

prompt = """Buatkan 10 teks berbeda dalam Bahasa Indonesia dengan spesifikasi berikut:

GAYA: CASUAL
TOPIK: Bervariasi dari list ini [Sosial, Budaya, Politik, Ekonomi, Teknologi, Kesehatan, Olahraga, Pendidikan, Lingkungan, Entertainment]
PANJANG: 150-400 kata per teks

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

print("\n[START] Sending request...")
print("-"*60)

start_time = time.time()

try:
    response = client.chat_completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0.8
    )

    end_time = time.time()
    elapsed = end_time - start_time

    raw_text = response.choices[0].message.content

    print(f"[DONE] Response received!")
    print(f"\nTime elapsed: {elapsed:.2f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Response length: {len(raw_text)} characters")
    print(f"Speed: {len(raw_text)/elapsed:.0f} chars/second")

    # Parse texts
    import re
    pattern = r'---TEKS\s*(\d+)---'
    parts = re.split(pattern, raw_text)

    texts_found = 0
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            text_block = parts[i + 1]
            content = re.sub(r'TOPIK:\s*.+?\n', '', text_block, flags=re.IGNORECASE).strip()
            if len(content) > 50:
                texts_found += 1

    print(f"\nTexts parsed: {texts_found}/10")

    # Show preview
    print(f"\n--- PREVIEW (first 500 chars) ---")
    print(raw_text[:500])
    print("..." if len(raw_text) > 500 else "")

except Exception as e:
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"[X] Error after {elapsed:.2f} seconds: {e}")

print("\n" + "="*60)
print("COMPARISON (estimated from previous tests):")
print("="*60)
print("OpenRouter (Step-3.5-Flash):  ~10-20 seconds per batch")
print("OpenRouter (TNG-R1T-Chimera): ~15-30 seconds per batch")
print("Groq (Llama-3.3-70B):        ~5-15 seconds per batch")
print(f"HuggingFace (Qwen3-Coder):   ~{elapsed:.0f} seconds per batch")
print("="*60)
