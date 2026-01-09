"""
Test script to verify that all RoBERTa joke rater models can be loaded correctly.

Usage:
    python test_roberta_models.py
"""
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, pipeline


MODELS = [
    "KonradBRG/joke-rater-roberta-en",
    "KonradBRG/joke-rater-roberta-zh",
    "KonradBRG/joke-rater-roberta-es",
    "KonradBRG/joke-rater-xlm-roberta",
]

TEST_JOKES = {
    "en": [
        "Why did the chicken cross the road? To get to the other side!",
        "I told my wife she was drawing her eyebrows too high. She looked surprised.",
        "This is not a joke at all.",
    ],
    "zh": [
        "为什么程序员总是分不清万圣节和圣诞节？因为 Oct 31 = Dec 25。",
        "我问我爸为什么要给我取这个名字，他说因为我出生的时候他正在看电视。",
    ],
    "es": [
        "¿Por qué los pájaros vuelan hacia el sur? Porque es muy lejos para ir caminando.",
        "Mi perro se llama 'Cinco Kilómetros' para poder decir que camino Cinco Kilómetros todos los días.",
    ],
}


def test_model(model_id: str):
    """Test loading and running inference on a single model."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_id}")
    print("=" * 60)

    try:
        # Load config first with trust_remote_code
        print("Loading config...")
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        print(f"  Config type: {type(config).__name__}")
        print(f"  Model type: {config.model_type}")

        # Load model
        print("Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            config=config,
            trust_remote_code=True,
        )
        print(f"  Model type: {type(model).__name__}")

        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Create pipeline
        print("Creating pipeline...")
        pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1,  # CPU for testing
        )

        # Determine language from model name
        if "-en" in model_id or "xlm-roberta" in model_id:
            jokes = TEST_JOKES["en"]
        elif "-zh" in model_id:
            jokes = TEST_JOKES["zh"]
        elif "-es" in model_id:
            jokes = TEST_JOKES["es"]
        else:
            jokes = TEST_JOKES["en"]

        # Run inference
        print("\nRunning inference...")
        for joke in jokes:
            result = pipe(joke[:200])  # Truncate long jokes
            score = result[0]["label"]
            confidence = result[0]["score"]
            print(f"  Score: {score:>2} (conf: {confidence:.3f}) | {joke[:50]}...")

        print(f"\n✓ {model_id} loaded and tested successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Error testing {model_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("Testing RoBERTa Joke Rater Models")
    print("=" * 60)

    results = {}
    for model_id in MODELS:
        results[model_id] = test_model(model_id)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model_id, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {model_id}")

    # Exit with error if any failed
    if not all(results.values()):
        print("\nSome models failed to load!")
        exit(1)
    else:
        print("\nAll models loaded successfully!")


if __name__ == "__main__":
    main()
