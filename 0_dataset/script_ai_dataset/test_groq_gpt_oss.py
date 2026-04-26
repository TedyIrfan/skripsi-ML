"""
Test Groq with openai/gpt-oss-120b
Using Groq library (not requests)
"""

try:
    from groq import Groq
except ImportError:
    print("Installing groq library...")
    import subprocess
    subprocess.check_call(["pip", "install", "groq", "-q"])
    from groq import Groq

import re
import time

API_KEY = "YOUR_GROQ_API_KEY_HERE"
MODEL = "openai/gpt-oss-120b"

print("="*60)
print(f"TEST: {MODEL} from Groq")
print("="*60)

client = Groq(api_key=API_KEY)

# Test 1: Simple test
print("\n[TEST 1] Simple test")
print("-"*60)

try:
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "Halo! Siapa kamu? Jawab singkat dalam bahasa Indonesia."}
        ],
        temperature=0.7,
        max_tokens=100
    )

    content = completion.choices[0].message.content
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
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,
        max_tokens=3000
    )

    end_time = time.time()
    raw_text = completion.choices[0].message.content

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
