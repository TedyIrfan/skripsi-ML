"""
Test Hugging Face langsung ke model endpoint
"""

import requests

TOKEN = "YOUR_HF_TOKEN_HERE"

# Try smaller models yang lebih available
MODELS = [
    ("Qwen 2.5 3B", "Qwen/Qwen2.5-3B-Instruct"),
    ("Qwen 2 7B", "Qwen/Qwen2-7B-Instruct"),
    ("Mistral 7B", "mistralai/Mistral-7B-Instruct-v0.3"),
    ("Gemma 2 2B", "google/gemma-2-2b-it"),
]

print("="*60)
print("TEST HUGGING FACE (DIRECT ENDPOINT)")
print("="*60)

for name, model_id in MODELS:
    print(f"\n[TEST] {name}")
    print(f"Model: {model_id}")
    print("-"*60)

    # Try using the inference API format
    API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"

    headers = {
        "Authorization": f"Bearer {TOKEN}"
    }

    # Or use serverless inference
    SERVERLESS_URL = f"https://{model_id}.endpoint.huggingface.cloud"

    try:
        # Try simple completions endpoint
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{model_id}",
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type": "application/json"
            },
            json={"inputs": "Hello, who are you?"},
            timeout=30
        )

        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result}")
            print(f"[OK] {name} berhasil!")
            break
        elif response.status_code == 501:
            print(f"[!] Model loading or not configured")
        elif response.status_code == 401:
            print(f"[X] Token invalid")
        else:
            print(f"Error: {response.text[:200]}")

    except Exception as e:
        print(f"Exception: {e}")

print("\n" + "="*60)
print("CATATAN: Hugging Face free API mungkin butuh")
print("model deployment atau account verification.")
print("="*60)
