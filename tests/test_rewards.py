"""
Test script for GRPO reward functions.

Usage:
    python tests/test_rewards.py
    python tests/test_rewards.py --skip-roberta  # Skip slow roberta tests
    python tests/test_rewards.py --only-roberta  # Run only roberta tests
    python tests/test_rewards.py --data-dir ../data  # Test with parquet data
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from grpo.rewards import (
    # Validation functions
    is_valid_single_joke_en,
    is_valid_single_joke_zh,
    is_valid_single_joke_es,
    create_is_valid_single_joke_fn,
    # Structure extraction
    extract_joke_structure,
    extract_joke_structure_zh,
    extract_joke_structure_es,
    # Reward functions
    word_pair_prompt_adherence,
    create_formatting_fn,
    create_length_penalty_fn,
    create_headline_adherence_fn,
    create_coherence_penalty_fn,
    create_structure_diversity_reward_fn,
    create_roberta_score_fn,
    # Utils
    reset_recent_structures,
)


class TestResults:
    """Collect test results and report at the end."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.failures = []

    def check(self, condition, message, context=""):
        if condition:
            self.passed += 1
            return True
        else:
            self.failed += 1
            self.failures.append(f"{context}: {message}" if context else message)
            return False

    def summary(self):
        print("\n" + "="*60)
        print(f"SUMMARY: {self.passed} passed, {self.failed} failed")
        print("="*60)
        if self.failures:
            print("\nFailures:")
            for f in self.failures:
                print(f"  - {f}")
        return self.failed == 0


results = TestResults()


def test_word_pair_prompt_adherence():
    """Test word pair prompt adherence for all languages."""
    print("\n" + "="*60)
    print("Testing word_pair_prompt_adherence")
    print("="*60)

    # English tests
    en_prompts = [
        "Generate a funny joke using these two words: 'cat', 'pizza'.",
        "Generate a funny joke using these two words: 'dog', 'computer'.",
        "Generate a funny joke using these two words: 'robot', 'banana'.",
    ]
    en_completions = [
        "Why did the cat order pizza? Because it wanted a purr-fect meal!",  # Both words
        "The dog went to the store.",  # Only one word (dog)
        "I like ice cream.",  # Neither word
    ]

    scores = word_pair_prompt_adherence(en_completions, en_prompts)
    print("\nEnglish tests:")
    print(f"  Both words present: {scores[0]} (expected: 2.0)")
    print(f"  One word present:   {scores[1]} (expected: -1.0)")
    print(f"  No words present:   {scores[2]} (expected: -2.0)")

    results.check(scores[0] == 2.0, f"Expected 2.0 for both words, got {scores[0]}", "EN both words")
    results.check(scores[1] == -1.0, f"Expected -1.0 for one word, got {scores[1]}", "EN one word")
    results.check(scores[2] == -2.0, f"Expected -2.0 for no words, got {scores[2]}", "EN no words")

    # Chinese tests
    zh_prompts = [
        "用这两个词生成一个有趣的笑话：'猫'、'披萨'。",
        "用这两个词生成一个有趣的笑话：'狗'、'电脑'。",
    ]
    zh_completions = [
        "猫咪看着披萨说：这是我的！",  # Both words
        "今天天气真好。",  # Neither word
    ]

    scores = word_pair_prompt_adherence(zh_completions, zh_prompts)
    print("\nChinese tests:")
    print(f"  Both words present: {scores[0]} (expected: 2.0)")
    print(f"  No words present:   {scores[1]} (expected: -2.0)")

    results.check(scores[0] == 2.0, f"Expected 2.0 for both words, got {scores[0]}", "ZH both words")
    results.check(scores[1] == -2.0, f"Expected -2.0 for no words, got {scores[1]}", "ZH no words")

    # Spanish tests
    es_prompts = [
        "Genera un chiste gracioso usando estas dos palabras: 'gato', 'pizza'.",
        "Genera un chiste gracioso usando estas dos palabras: 'perro', 'computadora'.",
    ]
    es_completions = [
        "El gato pidio una pizza con extra queso.",  # Both words
        "Me gusta el helado.",  # Neither word
    ]

    scores = word_pair_prompt_adherence(es_completions, es_prompts)
    print("\nSpanish tests:")
    print(f"  Both words present: {scores[0]} (expected: 2.0)")
    print(f"  No words present:   {scores[1]} (expected: -2.0)")

    results.check(scores[0] == 2.0, f"Expected 2.0 for both words, got {scores[0]}", "ES both words")
    results.check(scores[1] == -2.0, f"Expected -2.0 for no words, got {scores[1]}", "ES no words")

    # Headline task (should return None)
    headline_prompts = ["Generate a funny joke related to this headline: 'Stock market crashes'."]
    headline_completions = ["The stock market crashed so hard it needed a helmet."]
    scores = word_pair_prompt_adherence(headline_completions, headline_prompts)
    print("\nHeadline task (should skip):")
    print(f"  Score: {scores[0]} (expected: None)")
    results.check(scores[0] is None, f"Expected None for headline task, got {scores[0]}", "Headline skip")

    print("\n[DONE] word_pair_prompt_adherence tests completed")


def test_is_valid_single_joke():
    """Test joke validation functions."""
    print("\n" + "="*60)
    print("Testing is_valid_single_joke")
    print("="*60)

    # English
    print("\nEnglish validation:")
    valid_en = "Why did the chicken cross the road? To get to the other side!"
    invalid_multiple_q = "Why? What? How?"
    invalid_qa_format = "Q: Why did the chicken cross? A: To get there."
    invalid_stacking = "Why did the cat? Why did the dog?"

    results.check(is_valid_single_joke_en(valid_en) == True, "Valid joke should pass", "EN valid")
    results.check(is_valid_single_joke_en(invalid_multiple_q) == False, "Multiple ? should fail", "EN multiple ?")
    results.check(is_valid_single_joke_en(invalid_qa_format) == False, "Q:/A: format should fail", "EN Q:/A:")
    results.check(is_valid_single_joke_en(invalid_stacking) == False, "Multiple 'why' should fail", "EN stacking")
    print("  English validation tests completed")

    # Chinese
    print("\nChinese validation:")
    valid_zh = "为什么猫喜欢披萨？因为它很好吃！"
    invalid_zh_multiple_q = "为什么？为什么？"
    invalid_zh_qa = "问：为什么？答：因为。"

    results.check(is_valid_single_joke_zh(valid_zh) == True, "Valid Chinese joke should pass", "ZH valid")
    results.check(is_valid_single_joke_zh(invalid_zh_multiple_q) == False, "Multiple 为什么 should fail", "ZH multiple")
    results.check(is_valid_single_joke_zh(invalid_zh_qa) == False, "问：/答： format should fail", "ZH Q/A")
    print("  Chinese validation tests completed")

    # Spanish
    print("\nSpanish validation:")
    valid_es = "¿Por qué el gato pidio pizza? Porque tenia hambre!"
    invalid_es_multiple = "¿Por qué? ¿Por qué otra vez?"

    results.check(is_valid_single_joke_es(valid_es) == True, "Valid Spanish joke should pass", "ES valid")
    results.check(is_valid_single_joke_es(invalid_es_multiple) == False, "Multiple por qué should fail", "ES multiple")
    print("  Spanish validation tests completed")

    print("\n[DONE] is_valid_single_joke tests completed")


def test_extract_joke_structure():
    """Test joke structure extraction."""
    print("\n" + "="*60)
    print("Testing extract_joke_structure")
    print("="*60)

    # English
    print("\nEnglish structures:")
    test_cases_en = [
        ("Why did the chicken cross the road?", "why-did"),
        ("What do you call a lazy kangaroo?", "what-do-you-call"),
        ("Knock knock! Who's there?", "knock-knock"),
        ("Life is like a box of chocolates.", "observation"),
        ("Why did the tomato turn red? Because it saw the salad dressing!", "qa-punchline"),
        ("I used to hate facial hair, but then it grew on me.", "one-liner"),
    ]

    for joke, expected in test_cases_en:
        result = extract_joke_structure(joke)
        passed = results.check(result == expected, f"Expected {expected}, got {result}", f"EN structure: {joke[:30]}")
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] '{joke[:40]}...' -> {result} (expected: {expected})")

    # Chinese
    print("\nChinese structures:")
    test_cases_zh = [
        ("为什么猫喜欢鱼？", "why-did"),
        ("什么叫真正的朋友？", "what-do-you-call"),
        ("生活就像一盒巧克力。", "observation"),
    ]

    for joke, expected in test_cases_zh:
        result = extract_joke_structure_zh(joke)
        passed = results.check(result == expected, f"Expected {expected}, got {result}", f"ZH structure: {joke[:20]}")
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] '{joke}' -> {result} (expected: {expected})")

    # Spanish
    print("\nSpanish structures:")
    test_cases_es = [
        ("¿Por qué el gato come pizza?", "why-did"),
        ("¿Cómo se llama un pez sin ojos?", "what-do-you-call"),
        ("Toc toc! Quien es?", "knock-knock"),
    ]

    for joke, expected in test_cases_es:
        result = extract_joke_structure_es(joke)
        passed = results.check(result == expected, f"Expected {expected}, got {result}", f"ES structure: {joke[:20]}")
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] '{joke}' -> {result} (expected: {expected})")

    print("\n[DONE] extract_joke_structure tests completed")


def test_formatting():
    """Test formatting reward function."""
    print("\n" + "="*60)
    print("Testing formatting reward")
    print("="*60)

    formatting_en = create_formatting_fn("en")

    good_jokes = [
        "Why did the programmer quit? Because he didn't get arrays!",
        "I told my cat a joke. It was not amused.",
    ]
    bad_jokes = [
        "How about: Here's a joke for you!",  # Bad pattern
        "This joke is funny. Let me know what you think!",  # Bad patterns
        "Why? Why? Why did this happen?",  # Invalid structure (multiple ?)
    ]

    good_scores = formatting_en(good_jokes)
    bad_scores = formatting_en(bad_jokes)

    print("\nGood jokes (expected: 1.0):")
    for joke, score in zip(good_jokes, good_scores):
        passed = results.check(score == 1.0, f"Expected 1.0, got {score}", f"Good joke: {joke[:30]}")
        print(f"  Score: {score} - '{joke[:50]}...'")

    print("\nBad jokes (expected: -1.0):")
    for joke, score in zip(bad_jokes, bad_scores):
        passed = results.check(score == -1.0, f"Expected -1.0, got {score}", f"Bad joke: {joke[:30]}")
        print(f"  Score: {score} - '{joke[:50]}...'")

    print("\n[DONE] formatting tests completed")


def test_length_penalty():
    """Test length penalty for different languages."""
    print("\n" + "="*60)
    print("Testing length_penalty")
    print("="*60)

    # English (word-based: min=5, max=24, optimal=16)
    length_en = create_length_penalty_fn("en")

    en_jokes = [
        "Short.",  # Too short (1 word)
        "This is a joke with exactly sixteen words in it for testing purposes here now.",  # 16 words (optimal)
        " ".join(["word"] * 30),  # Too long (30 words)
    ]

    scores = length_en(en_jokes)
    print("\nEnglish (word-based):")
    print(f"  Too short (1 word):  {scores[0]} (expected: -2.0)")
    print(f"  Optimal (16 words):  {scores[1]} (expected: 0.0)")
    print(f"  Too long (30 words): {scores[2]} (expected: -2.0)")

    results.check(scores[0] == -2.0, f"Expected -2.0 for too short, got {scores[0]}", "EN too short")
    results.check(scores[1] == 0.0, f"Expected 0.0 for optimal, got {scores[1]}", "EN optimal")
    results.check(scores[2] == -2.0, f"Expected -2.0 for too long, got {scores[2]}", "EN too long")

    # Chinese (character-based: min=10, max=60, optimal=35)
    length_zh = create_length_penalty_fn("zh")

    zh_jokes = [
        "短",  # Too short (1 char)
        "这是一个测试笑话，用来测试长度惩罚功能是否正常工作。",  # ~25 chars
        "这" * 70,  # Too long (70 chars)
    ]

    scores = length_zh(zh_jokes)
    print("\nChinese (character-based):")
    print(f"  Too short (1 char):  {scores[0]} (expected: -2.0)")
    print(f"  Medium (~25 chars):  {scores[1]} (expected: ~0.0 or small negative)")
    print(f"  Too long (70 chars): {scores[2]} (expected: -2.0)")

    results.check(scores[0] == -2.0, f"Expected -2.0 for too short, got {scores[0]}", "ZH too short")
    results.check(scores[2] == -2.0, f"Expected -2.0 for too long, got {scores[2]}", "ZH too long")

    print("\n[DONE] length_penalty tests completed")


def test_headline_adherence():
    """Test headline adherence function."""
    print("\n" + "="*60)
    print("Testing headline_adherence")
    print("="*60)

    headline_en = create_headline_adherence_fn("en")

    # Word pair prompts should return None
    word_pair_prompts = ["Generate a joke using these two words: 'cat', 'dog'."]
    word_pair_completions = ["The cat and dog became friends."]

    scores = headline_en(word_pair_completions, word_pair_prompts)
    print("\nWord pair task (should skip):")
    print(f"  Score: {scores[0]} (expected: None)")
    results.check(scores[0] is None, f"Expected None, got {scores[0]}", "Word pair skip")

    # Headline prompts
    headline_prompts = [
        "Generate a joke about: 'Stock market crashes'.",
        "Generate a joke about: 'New discovery'.",
        "Generate a joke about: 'Weather'.",
    ]
    headline_completions = [
        "The market crashed harder than my dating life.",  # Good, short
        "I need to generate a headline joke here.",  # Bad pattern "generate"
        " ".join(["word"] * 30),  # Too long
    ]

    scores = headline_en(headline_completions, headline_prompts)
    print("\nHeadline tasks:")
    print(f"  Good short joke:    {scores[0]} (expected: 1.0)")
    print(f"  Bad pattern:        {scores[1]} (expected: -1.0)")
    print(f"  Too long:           {scores[2]} (expected: -1.0)")

    results.check(scores[0] == 1.0, f"Expected 1.0, got {scores[0]}", "Good headline")
    results.check(scores[1] == -1.0, f"Expected -1.0, got {scores[1]}", "Bad pattern headline")
    results.check(scores[2] == -1.0, f"Expected -1.0, got {scores[2]}", "Too long headline")

    print("\n[DONE] headline_adherence tests completed")


def test_coherence_penalty():
    """Test coherence penalty function."""
    print("\n" + "="*60)
    print("Testing coherence_penalty")
    print("="*60)

    coherence_en = create_coherence_penalty_fn("en")

    jokes = [
        "why did the chicken cross the road to get to the other side",  # No caps, score ~0
        "The Cat And The Dog And The Bird And The Fish",  # Many caps, negative score
    ]

    scores = coherence_en(jokes)
    print("\nEnglish coherence:")
    print(f"  No caps:   {scores[0]} (expected: 0.0)")
    print(f"  Many caps: {scores[1]} (expected: negative)")

    results.check(scores[0] == 0.0, f"Expected 0.0, got {scores[0]}", "No caps")
    results.check(scores[1] < 0, f"Expected negative, got {scores[1]}", "Many caps")

    print("\n[DONE] coherence_penalty tests completed")


def test_roberta_score():
    """Test roberta scoring on sample jokes for all language models."""
    print("\n" + "="*60)
    print("Testing roberta_score for all language models")
    print("="*60)

    # Test configurations for each language
    test_configs = [
        {
            "model_id": "KonradBRG/joke-rater-roberta-en",
            "language": "en",
            "jokes": [
                "Why did the scarecrow win an award? Because he was outstanding in his field!",
                "I told my wife she was drawing her eyebrows too high. She looked surprised.",
                "What do you call a fake noodle? An impasta!",
                "This is not a joke at all, just a regular sentence.",
                "Why? Why? Why did this happen? Q: What? A: Nothing.",  # Invalid format
            ],
            "invalid_idx": 4,
        },
        {
            "model_id": "KonradBRG/joke-rater-roberta-zh",
            "language": "zh",
            "jokes": [
                "为什么程序员喜欢黑暗？因为光会产生bug！",
                "我告诉我的猫一个笑话，它一点都不觉得好笑。",
                "什么叫真正的朋友？借钱不还的那种！",
                "这不是一个笑话，只是一个普通的句子。",
                "为什么？为什么？问：什么？答：没有。",  # Invalid format
            ],
            "invalid_idx": 4,
        },
        {
            "model_id": "KonradBRG/joke-rater-roberta-es",
            "language": "es",
            "jokes": [
                "¿Por qué el libro de matemáticas estaba triste? Porque tenía muchos problemas!",
                "Le dije a mi esposa que estaba dibujando sus cejas muy altas. Se sorprendió.",
                "¿Cómo se llama un boomerang que no vuelve? Un palo!",
                "Esto no es un chiste, solo una oración normal.",
                "¿Por qué? ¿Por qué otra vez? P: ¿Qué? R: Nada.",  # Invalid format
            ],
            "invalid_idx": 4,
        },
    ]

    for config in test_configs:
        model_id = config["model_id"]
        lang = config["language"]
        jokes = config["jokes"]
        invalid_idx = config["invalid_idx"]

        print(f"\n--- Testing {lang.upper()}: {model_id} ---")

        try:
            roberta_fn = create_roberta_score_fn(model_id, language=lang)

            print("  Scoring jokes...")
            scores = roberta_fn(jokes)

            print("  Results:")
            for joke, score in zip(jokes, scores):
                display_joke = joke[:50] + "..." if len(joke) > 50 else joke
                print(f"    Score: {score:.3f} - '{display_joke}'")

            # The invalid joke should get 0.0 (filtered out)
            print(f"\n  Invalid joke score: {scores[invalid_idx]} (expected: 0.0)")
            results.check(
                scores[invalid_idx] == 0.0,
                f"Expected 0.0 for invalid joke, got {scores[invalid_idx]}",
                f"{lang} invalid joke filtered"
            )

            # Valid jokes should have non-zero scores
            results.check(
                scores[0] != 0.0,
                f"Expected non-zero for valid joke, got {scores[0]}",
                f"{lang} valid joke scored"
            )

            print(f"  [DONE] {lang.upper()} roberta tests completed")

        except Exception as e:
            print(f"  [ERROR] Failed to load {model_id}: {e}")
            results.check(False, f"Failed to load {model_id}: {e}", f"{lang} roberta load")


def test_with_parquet_data(data_dir: str):
    """Test reward functions with actual training/test data from parquet files."""
    print("\n" + "="*60)
    print(f"Testing with parquet data from {data_dir}")
    print("="*60)

    try:
        import pandas as pd
    except ImportError:
        print("[SKIP] pandas not installed, skipping parquet tests")
        return

    # Language-specific completion templates
    completion_templates = {
        "en": {
            "word_pair": "Why did the {0} meet the {1}? Because they wanted to be friends!",
            "headline": "This is a short headline joke.",
        },
        "zh": {
            "word_pair": "为什么{0}遇到了{1}？因为它们想成为朋友！",
            "headline": "这是一个简短的标题笑话。",
        },
        "es": {
            "word_pair": "¿Por qué el {0} conoció al {1}? Porque querían ser amigos!",
            "headline": "Este es un chiste corto sobre el titular.",
        },
    }

    # Test for each language
    for lang, suffix in [("en", ""), ("zh", "_zh"), ("es", "_es")]:
        train_path = os.path.join(data_dir, f"rl_df_train{suffix}.parquet")
        test_path = os.path.join(data_dir, f"rl_df_test{suffix}.parquet")

        if not os.path.exists(train_path):
            print(f"\n[SKIP] {train_path} not found")
            continue

        print(f"\n--- Testing {lang.upper()} data ---")

        # Reset structure history before testing each language
        reset_recent_structures()

        # Load data
        train_df = pd.read_parquet(train_path)
        print(f"  Loaded {len(train_df)} training prompts")

        # Sample some prompts
        sample_prompts = train_df["prompt"].head(10).tolist()

        # Test word_pair_prompt_adherence pattern matching
        print(f"\n  Testing word_pair pattern matching on {len(sample_prompts)} samples:")

        # Create fake completions in the target language
        templates = completion_templates[lang]
        good_completions = []
        for prompt in sample_prompts:
            # Extract words if it's a word pair prompt
            import re
            patterns = [
                r"'([^']+)'[、,]\s*'([^']+)'",  # Matches both , and 、
            ]
            words = None
            for pattern in patterns:
                match = re.search(pattern, prompt)
                if match:
                    words = match.groups()
                    break

            if words:
                # Create a completion with both words in target language
                good_completions.append(templates["word_pair"].format(words[0], words[1]))
            else:
                # Headline prompt
                good_completions.append(templates["headline"])

        scores = word_pair_prompt_adherence(good_completions, sample_prompts)

        word_pair_count = sum(1 for s in scores if s is not None)
        headline_count = sum(1 for s in scores if s is None)
        positive_scores = sum(1 for s in scores if s is not None and s > 0)

        print(f"    Word pair prompts: {word_pair_count}")
        print(f"    Headline prompts:  {headline_count}")
        print(f"    Positive scores:   {positive_scores}/{word_pair_count}")

        results.check(
            word_pair_count > 0 or headline_count > 0,
            f"Should detect prompts in {lang}",
            f"{lang} prompt detection"
        )

        # Test length penalty on actual format
        length_fn = create_length_penalty_fn(lang)
        if lang == "zh":
            test_completions = [
                "短",  # Too short
                "这是一个中等长度的笑话，用来测试长度惩罚功能。",  # Medium (~20 chars)
                "这" * 100,  # Too long
            ]
        else:
            test_completions = [
                "Short.",
                "This is a medium length joke that should be within the acceptable range for testing.",
                " ".join(["word"] * 40),
            ]
        length_scores = length_fn(test_completions)
        print(f"\n  Length penalty test:")
        print(f"    Short:  {length_scores[0]}")
        print(f"    Medium: {length_scores[1]}")
        print(f"    Long:   {length_scores[2]}")

        # Test structure diversity with language-appropriate completions
        reset_recent_structures()  # Reset again for clean diversity test
        structure_fn = create_structure_diversity_reward_fn(lang)
        structure_scores = structure_fn(good_completions[:5])
        print(f"\n  Structure diversity scores: {[round(s, 3) for s in structure_scores]}")

        # Verify first score is positive (novel structure)
        if structure_scores:
            results.check(
                structure_scores[0] > 0,
                f"First structure should be novel in {lang}",
                f"{lang} structure diversity"
            )

        if os.path.exists(test_path):
            test_df = pd.read_parquet(test_path)
            print(f"\n  Loaded {len(test_df)} test prompts")

    print("\n[DONE] parquet data tests completed")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test reward functions")
    parser.add_argument("--skip-roberta", action="store_true", help="Skip slow roberta tests")
    parser.add_argument("--only-roberta", action="store_true", help="Run only roberta tests")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory with parquet data files")
    args = parser.parse_args()

    print("="*60)
    print("REWARD FUNCTION TESTS")
    print("="*60)

    if args.only_roberta:
        # Run only roberta tests
        test_roberta_score()
    else:
        # Run all non-roberta tests
        test_word_pair_prompt_adherence()
        test_is_valid_single_joke()
        test_extract_joke_structure()
        test_formatting()
        test_length_penalty()
        test_headline_adherence()
        test_coherence_penalty()

        if not args.skip_roberta:
            test_roberta_score()
        else:
            print("\n[SKIPPED] roberta_score tests (use without --skip-roberta to run)")

        if args.data_dir:
            test_with_parquet_data(args.data_dir)
        else:
            # Try default location
            default_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
            if os.path.exists(default_data_dir):
                test_with_parquet_data(default_data_dir)
            else:
                print("\n[SKIPPED] parquet data tests (use --data-dir to specify location)")

    # Print summary
    success = results.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
