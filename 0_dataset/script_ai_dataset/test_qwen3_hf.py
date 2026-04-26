"""
Test Hugging Face - Qwen3-Coder-Next
"""

import requests
import json

TOKEN = "YOUR_HF_TOKEN_HERE"
MODEL = "Qwen/Qwen3-Coder-Next"

print("="*60)
print(f"TEST: {MODEL}")
print("="*60)

# Try different API formats
formats_to_try = [
    {
        "name": "Router (chat format)",
        "url": "https://router.huggingface.co/chat/" + MODEL,
        "payload": {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Halo! Siapa kamu?"}],
            "max_tokens": 100
        }
    },
    {
        "name": "Router (models)",
        "url": "https://router.huggingface.co/models/" + MODEL,
        "payload": {
            "inputs": "Halo! Siapa kamu?",
            "parameters": {"max_new_tokens": 100}
        }
    },
    {
        "name": "Inference (old)",
        "url": "https://api-inference.huggingface.co/models/" + MODEL,
        "payload": {
            "inputs": "Halo! Siapa kamu?",
            "parameters": {"max_new_tokens": 100}
        }
    },
]

for fmt in formats_to_try:
    print(f"\n[TEST] {fmt['name']}")
    print(f"URL: {fmt['url']}")
    print("-"*60)

    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(fmt['url'], headers=headers, json=fmt['payload'], timeout=60)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"[OK] BERHASIL!")
            print(f"Response: {json.dumps(result, indent=2)[:500]}")
            break
        elif response.status_code == 503:
            print(f"[!] Model is loading...")
        else:
            print(f"Error: {response.text[:300]}")

    except Exception as e:
        print(f"Exception: {e}")

print("\n" + "="*60)
