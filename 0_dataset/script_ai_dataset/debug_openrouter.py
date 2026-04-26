"""
Debug script untuk OpenRouter API
Cek raw response dari model
"""

import requests
import json

API_KEY = "YOUR_OPENROUTER_API_KEY_HERE"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "z-ai/glm-4.5-air:free"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/irfan-skripsi",
    "X-Title": "AI vs Human Text Detection"
}

print("="*60)
print(f"DEBUG: {MODEL}")
print("="*60)

# Test dengan simple prompt
data = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "Halo! Siapa kamu?"}
    ],
    "max_tokens": 100
}

print("\n[TEST 1] Simple prompt: 'Halo! Siapa kamu?'")
print("-"*60)

try:
    response = requests.post(API_URL, headers=HEADERS, json=data, timeout=60)
    print(f"Status Code: {response.status_code}")

    print(f"\n--- RAW JSON RESPONSE ---")
    result = response.json()
    print(json.dumps(result, indent=2))

    if "choices" in result and len(result["choices"]) > 0:
        content = result["choices"][0]["message"]["content"]
        print(f"\n--- CONTENT ---")
        print(f"Length: {len(content)} chars")
        print(f"Content: {repr(content)}")
    else:
        print("\n[X] No 'choices' in response")

except Exception as e:
    print(f"[X] Error: {e}")

# Test 2: Dengan prompt yang lebih panjang
print("\n" + "="*60)
print("[TEST 2] Prompt panjang (seperti di generator)")
print("-"*60)

data2 = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "Kamu adalah content creator yang suka nulis dengan gaya santai."},
        {"role": "user", "content": "Buatkan 1 teks singkat tentang teknologi dalam bahasa Indonesia."}
    ],
    "max_tokens": 500
}

try:
    response = requests.post(API_URL, headers=HEADERS, json=data2, timeout=60)
    print(f"Status Code: {response.status_code}")

    result = response.json()

    if "choices" in result and len(result["choices"]) > 0:
        content = result["choices"][0]["message"]["content"]
        print(f"\n--- CONTENT ---")
        print(f"Length: {len(content)} chars")
        print(f"Content:\n{content}")
    else:
        print(f"\n--- RAW JSON ---")
        print(json.dumps(result, indent=2))

except Exception as e:
    print(f"[X] Error: {e}")

print("\n" + "="*60)
print("DEBUG SELESAI")
print("="*60)
