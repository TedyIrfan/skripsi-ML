"""
Test Kimi K2.5 dari HuggingFace
"""

from huggingface_hub import InferenceClient
import re
import time

TOKEN = "YOUR_HF_TOKEN_HERE"
MODEL = "moonshotai/Kimi-K2.5"

client = InferenceClient(token=TOKEN)

print("="*60)
print(f"TEST: {MODEL}")
print("="*60)

# Test 1: Simple test
print("\n[TEST 1] Simple test")
print("-"*60)

try:
    response = client.chat_completion(
        model=MODEL,
        messages=[{"role": "user", "content": "Halo! Siapa kamu? Jawab singkat dalam bahasa Indonesia."}],
        max_tokens=100
    )

    content = response.choices[0].message.content
    print(f"Response: {content}")
    print("[OK] Simple test berhasil!")

except Exception as e:
    print(f"[X] Error: {e}")

# Test 2: Indonesian text generation
print("\n[TEST 2] Indonesian text generation")
print("-"*60)

prompt = """Buatkan 3 teks berbeda dalam Bahasa Indonesia.

GAYA: CASUAL
TOPIK: Sosial, Budaya, Teknologi
PANJANG: 150-400 kata

OUTPUT FORMAT:
Berikan dalam format:

---TEKS 1---
TOPIK: [nama topik]
[isi teks]

---TEKS 2---
TOPIK: [nama topik]
[isi teks]

---TEKS 3---
TOPIK: [nama topik]
[isi teks]"""

start_time = time.time()

try:
    response = client.chat_completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.8
    )

    end_time = time.time()
    raw_text = response.choices[0].message.content

    print(f"Time: {end_time - start_time:.2f} seconds")
    print(f"Length: {len(raw_text)} chars")

    # Parse
    pattern = r'---TEKS\s*(\d+)---'
    parts = re.split(pattern, raw_text)

    texts_found = 0
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            text_block = parts[i + 1]
            topic_match = re.search(r'TOPIK:\s*(.+?)(?:\n|$)', text_block, re.IGNORECASE)
            topic = topic_match.group(1).strip() if topic_match else "Unknown"
            content = re.sub(r'TOPIK:\s*.+?\n', '', text_block, flags=re.IGNORECASE).strip()

            if len(content) > 50:
                texts_found += 1
                print(f"\n  [{texts_found}] Topic: {topic}")
                print(f"      Length: {len(content)} chars")
                print(f"      Preview: {content[:150]}...")

    print(f"\nTotal parsed: {texts_found}/3 texts")

    if texts_found == 3:
        print("\n[OK] Format looks good! Ready for generator.")
    else:
        print(f"\n[!] Only got {texts_found} texts")

except Exception as e:
    print(f"[X] Error: {e}")

print("\n" + "="*60)
