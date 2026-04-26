"""
Test 1 batch dari Qwen3 Coder untuk cek kualitas output
"""

import requests
import re

API_KEY = "YOUR_OPENROUTER_API_KEY_HERE"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "nousresearch/hermes-3-llama-3.1-405b:free"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/irfan-skripsi",
    "X-Title": "AI vs Human Text Detection"
}

print("="*60)
print(f"TEST MODEL: {MODEL}")
print("="*60)

# Test casual
data = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "Kamu adalah content creator yang suka nulis dengan gaya santai dan friendly. Tulis dalam bahasa Indonesia yang natural."},
        {"role": "user", "content": "Buatkan 3 teks berbeda dalam Bahasa Indonesia dengan spesifikasi berikut:\n\nGAYA: CASUAL\nTOPIK: Bervariasi dari list ini [Sosial, Budaya, Teknologi]\nPANJANG: 150-400 kata per teks\nFORMAT: Paragraf natural\n\nOUTPUT FORMAT:\n---TEKS 1---\nTOPIK: [nama topik]\n[isi teks]\n\n---TEKS 2---\nTOPIK: [nama topik]\n[isi teks]\n\n---TEKS 3---\nTOPIK: [nama topik]\n[isi teks]"}
    ],
    "max_tokens": 2000,
    "temperature": 0.8
}

print("\n[TEST 1] Casual Style")
print("-"*60)

try:
    response = requests.post(API_URL, headers=HEADERS, json=data, timeout=60)
    result = response.json()

    message = result["choices"][0]["message"]
    raw_text = message.get("content", "")

    # Check for reasoning
    if not raw_text and "reasoning" in message:
        raw_text = message["reasoning"]

    print(f"Length: {len(raw_text)} chars")
    print(f"\n--- RAW OUTPUT (first 1000 chars) ---")
    print(raw_text[:1000])
    print("..." if len(raw_text) > 1000 else "")

    # Parse
    pattern = r'---TEKS\s*(\d+)---'
    parts = re.split(pattern, raw_text)

    texts_found = 0
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            text_block = parts[i + 1]
            # Extract topic
            topic_match = re.search(r'TOPIK:\s*(.+?)(?:\n|$)', text_block, re.IGNORECASE)
            topic = topic_match.group(1).strip() if topic_match else "Unknown"
            # Get content
            content = re.sub(r'TOPIK:\s*.+?\n', '', text_block, flags=re.IGNORECASE).strip()

            if len(content) > 50:
                texts_found += 1
                print(f"\n  [{texts_found}] Topic: {topic}")
                print(f"      Length: {len(content)} chars")
                print(f"      Preview: {content[:150]}...")

    print(f"\nTotal parsed: {texts_found}/3 texts")

    if texts_found == 3:
        print("\n[OK] Format looks good!")
    else:
        print(f"\n[!] Only got {texts_found} texts, expected 3")

except Exception as e:
    print(f"[X] Error: {e}")

print("\n" + "="*60)
