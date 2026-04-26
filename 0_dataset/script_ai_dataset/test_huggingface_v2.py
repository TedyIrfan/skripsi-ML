"""
Test Hugging Face Inference API format baru
"""

import requests
import json

TOKEN = "YOUR_HF_TOKEN_HERE"

# Test different API formats
API_FORMATS = [
    ("New Router (models)", "https://router.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct"),
    ("New Router (chat)", "https://router.huggingface.co/chat/Qwen/Qwen2.5-72B-Instruct"),
    ("Inference (old)", "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct"),
]

print("="*60)
print("TEST HUGGING FACE API FORMATS")
print("="*60)

for name, url in API_FORMATS:
    print(f"\n[TEST] {name}")
    print(f"URL: {url}")
    print("-"*60)

    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }

    # Simple payload
    payload = {
        "inputs": "Halo! Siapa kamu?",
        "parameters": {
            "max_new_tokens": 50
        }
    }

    # Chat format payload
    chat_payload = {
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "messages": [
            {"role": "user", "content": "Halo! Siapa kamu?"}
        ],
        "max_tokens": 50
    }

    try:
        # Try chat format
        response = requests.post(url, headers=headers, json=chat_payload, timeout=30)
        print(f"Status (chat): {response.status_code}")

        if response.status_code == 200:
            print(f"[OK] BERHASIL dengan chat format!")
            print(f"Response: {response.text[:300]}")
            break
        elif response.status_code == 404:
            # Try simple format
            response2 = requests.post(url, headers=headers, json=payload, timeout=30)
            print(f"Status (simple): {response2.status_code}")
            if response2.status_code == 200:
                print(f"[OK] BERHASIL dengan simple format!")
                print(f"Response: {response2.text[:300]}")
                break
            else:
                print(f"Error: {response2.text[:200]}")
        else:
            print(f"Error: {response.text[:200]}")

    except Exception as e:
        print(f"Exception: {e}")

print("\n" + "="*60)
