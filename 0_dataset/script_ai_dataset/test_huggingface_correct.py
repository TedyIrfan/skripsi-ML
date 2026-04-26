"""
test_huggingface_correct.py
Cara BENAR menggunakan HuggingFace API
"""

from huggingface_hub import InferenceClient
import time

# ============================================================
# CONFIG
# ============================================================
TOKEN = "YOUR_HF_TOKEN_HERE"

# Models to test
MODELS = [
    "Qwen/Qwen3-Coder-Next",           # Coding specialist
    "stepfun-ai/Step-3.5-Flash",       # General purpose
    "google/gemma-2-9b-it",            # Fast & efficient
    "microsoft/Phi-3-mini-4k-instruct", # Small but capable
    "deepseek-ai/DeepSeek-R1",         # Reasoning
]

# Test prompt
PROMPT = "Write a Python function to calculate fibonacci numbers with memoization"

# ============================================================
# MAIN TEST
# ============================================================

def test_model(client, model_name, prompt):
    """Test single model"""
    try:
        response = client.chat_completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )

        result = response.choices[0].message.content
        return True, result

    except Exception as e:
        return False, str(e)


def main():
    print("=" * 70)
    print("HUGGING FACE API TEST (CORRECT METHOD)")
    print("=" * 70)

    # Initialize client
    client = InferenceClient(token=TOKEN)

    results = []

    for model in MODELS:
        print(f"\n{'-' * 70}")
        print(f"Testing: {model}")
        print('-' * 70)

        success, output = test_model(client, model, PROMPT)

        if success:
            print("[OK] SUCCESS")
            print(f"\nResponse:\n{output}\n")
            results.append({"model": model, "status": "success"})
        else:
            print(f"[X] FAILED: {output}")
            results.append({"model": model, "status": "failed", "error": output})

        # Rate limit protection
        time.sleep(2)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    success_count = sum(1 for r in results if r["status"] == "success")
    total_count = len(results)

    print(f"\nTotal Models Tested: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_count - success_count}")

    print("\nDetailed Results:")
    for r in results:
        status = "[OK]" if r["status"] == "success" else "[X]"
        print(f"{status} {r['model']}")
        if r["status"] == "failed":
            print(f"   Error: {r['error']}")


if __name__ == "__main__":
    main()
