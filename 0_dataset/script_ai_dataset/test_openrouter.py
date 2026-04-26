"""
Test script untuk OpenRouter API
Digunakan untuk generate AI text untuk dataset skripsi
"""

import requests
import json

def test_openrouter():
    """Test koneksi ke OpenRouter API"""

    # API Key
    api_key = "YOUR_OPENROUTER_API_KEY_HERE"

    # OpenRouter API endpoint
    url = "https://openrouter.ai/api/v1/chat/completions"

    # Headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/irfan-skripsi",
        "X-Title": "AI vs Human Text Detection"
    }

    # Test prompts
    test_prompts = [
        "Jelaskan tentang teknologi artificial intelligence dalam bahasa Indonesia.",
        "Apa itu machine learning dan bagaimana cara kerjanya?",
        "Ceritakan sejarah perkembangan komputer modern."
    ]

    print("\n" + "="*50)
    print("TESTING OPENROUTER API")
    print("Model: arcee-ai/trinity-large-preview:free")
    print("="*50)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[Test {i}] Prompt: {prompt}")

        # Request body
        data = {
            "model": "arcee-ai/trinity-large-preview:free",
            "messages": [
                {
                    "role": "system",
                    "content": "Kamu adalah asisten yang membantu generate teks dalam bahasa Indonesia. Berikan jawaban yang detail dan informatif."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 500,
            "temperature": 0.7
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()

            result = response.json()

            # Extract generated text
            if "choices" in result and len(result["choices"]) > 0:
                generated_text = result["choices"][0]["message"]["content"]
                print(f"[OK] Success! Generated text ({len(generated_text)} chars):")
                print("-" * 40)
                print(generated_text[:400] + "..." if len(generated_text) > 400 else generated_text)
                print("-" * 40)

                # Print usage info if available
                if "usage" in result:
                    print(f"Tokens: {result['usage'].get('total_tokens', 'N/A')}")

            else:
                print("[X] No response content")
                print(json.dumps(result, indent=2))

        except requests.exceptions.RequestException as e:
            print(f"[X] Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return False

    print("\n" + "="*50)
    print("TEST SELESAI!")
    print("="*50)
    return True


if __name__ == "__main__":
    if test_openrouter():
        print("\n[OK] OpenRouter API berhasil digunakan!")
    else:
        print("\n[X] Gagal terkoneksi ke OpenRouter API")
