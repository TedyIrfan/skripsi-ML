"""
Test Groq API dengan berbagai model
"""

import requests
import json

API_KEY = "YOUR_GROQ_API_KEY_HERE"
API_URL = "https://api.groq.com/openai/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Test models
MODELS = {
    "Llama 3.3 70B": "llama-3.3-70b-versatile",
    "Llama 3.1 70B": "llama-3.1-70b-versatile",
    "Mixtral 8x7B": "mixtral-8x7b-32768",
    "Gemma 2 9B": "gemma2-9b-it",
}

print("="*60)
print("TEST GROQ API - AVAILABLE MODELS")
print("="*60)

for name, model_id in MODELS.items():
    print(f"\n[TEST] {name} ({model_id})")
    print("-"*60)

    data = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": "Halo! Siapa kamu? Jawaban singkat dalam bahasa Indonesia."}
        ],
        "max_tokens": 100
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, json=data, timeout=30)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print(f"Content: {content[:200]}")
            print(f"[OK] Model {name} berhasil!")
        else:
            print(f"[X] Error: {response.text[:300]}")

    except Exception as e:
        print(f"[X] Exception: {e}")

print("\n" + "="*60)
print("REKOMENDASI MODEL:")
print("="*60)
print("Llama 3.3 70B - Paling powerful")
print("Llama 3.1 70B - Alternatif good")
print("Mixtral 8x7B - Cepat dan bagus")
print("Gemma 2 9B - Ringan dan cepat")
