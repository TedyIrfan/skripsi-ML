"""
Debug Qwen3 Coder
"""

import requests
import json

API_KEY = "YOUR_OPENROUTER_API_KEY_HERE"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwen3-coder-next:free"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

data = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "Halo!"}
    ],
    "max_tokens": 100
}

print(f"Testing: {MODEL}")
print("-"*60)

try:
    response = requests.post(API_URL, headers=HEADERS, json=data, timeout=60)
    print(f"Status Code: {response.status_code}")

    result = response.json()
    print("\nRaw Response:")
    print(json.dumps(result, indent=2))

except Exception as e:
    print(f"Error: {e}")
