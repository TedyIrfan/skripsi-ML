"""
Test kualitas output Groq Llama 3.3 untuk generate teks
"""

import requests
import re

API_KEY = "YOUR_GROQ_API_KEY_HERE"
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

print("="*60)
print(f"TEST KUALITAS: {MODEL}")
print("="*60)

# Test casual
data = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "Kamu adalah content creator yang suka nulis dengan gaya santai dan friendly. Tulis dalam bahasa Indonesia yang natural."},
        {"role": "user", "content": "Buatkan 3 teks berbeda dalam Bahasa Indonesia.\n\nGAYA: CASUAL\nTOPIK: Sosial, Budaya, Teknologi\nPANJANG: 150-400 kata\n\nOUTPUT FORMAT:\n---TEKS 1---\nTOPIK: [nama topik]\n[isi teks]\n\n---TEKS 2---\nTOPIK: [nama topik]\n[isi teks]\n\n---TEKS 3---\nTOPIK: [nama topik]\n[isi teks]"}
    ],
    "max_tokens": 2000,
    "temperature": 0.8
}

print("\n[TEST] Casual Style")
print("-"*60)

try:
    response = requests.post(API_URL, headers=HEADERS, json=data, timeout=60)
    result = response.json()
    raw_text = result["choices"][0]["message"]["content"]

    print(f"Length: {len(raw_text)} chars")
    print(f"\n--- RAW OUTPUT (first 800 chars) ---")
    print(raw_text[:800])
    print("..." if len(raw_text) > 800 else "")

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
