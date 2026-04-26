"""
Test Hugging Face API untuk text generation
Token: YOUR_HF_TOKEN_HERE
"""

import requests
import json

TOKEN = "YOUR_HF_TOKEN_HERE"

# Popular free models for text generation
MODELS = {
    "Qwen 2.5 72B": "Qwen/Qwen2.5-72B-Instruct",
    "Llama 3.1 70B": "meta-llama/Llama-3.1-70B-Instruct",
    "Mistral 7B": "mistralai/Mistral-7B-Instruct-v0.3",
    "Gemma 2 27B": "google/gemma-2-27b-it",
    "Qwen 2 7B": "Qwen/Qwen2-7B-Instruct",
}

print("="*60)
print("TEST HUGGING FACE API")
print("="*60)

for name, model_id in MODELS.items():
    print(f"\n[TEST] {name}")
    print(f"Model: {model_id}")
    print("-"*60)

    API_URL = f"https://router.huggingface.co/models/{model_id}"

    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }

    # Simple test prompt
    payload = {
        "inputs": f"<|im_start|>system\nKamu adalah asisten yang membantu. Jawab dalam bahasa Indonesia.<|im_end|>\n<|im_start|>user\nHalo! Siapa kamu? Jawab singkat.<|im_end|>\n<|im_start|>assistant\n",
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.7
        }
    }

    # For models that don't use chat template
    simple_payload = {
        "inputs": "Halo! Siapa kamu? Jawab singkat dalam bahasa Indonesia.",
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.7
        }
    }

    try:
        # Try chat format first
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        print(f"Status (chat format): {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                content = result[0].get("generated_text", "")
                # Extract just the assistant response
                if "<|im_start|>assistant" in content:
                    content = content.split("<|im_start|>assistant")[-1].strip()
                print(f"Response: {content[:200]}")
                print(f"[OK] {name} berhasil!")
            else:
                print(f"Response: {result}")
        elif response.status_code == 503:
            print(f"[!] Model is loading...")
        elif response.status_code == 401:
            print(f"[X] Unauthorized - check token")
        else:
            # Try simple format
            response2 = requests.post(API_URL, headers=headers, json=simple_payload, timeout=30)
            print(f"Status (simple format): {response2.status_code}")
            if response2.status_code == 200:
                result = response2.json()
                if isinstance(result, list) and len(result) > 0:
                    content = result[0].get("generated_text", "")
                    print(f"Response: {content[:200]}")
                    print(f"[OK] {name} berhasil with simple format!")
            else:
                print(f"[X] Error: {response2.text[:200]}")

    except Exception as e:
        print(f"[X] Exception: {e}")

    print()

print("="*60)
print("SUMMARY")
print("="*60)
print("Model yang berhasil akan digunakan untuk generator.")
