"""
Test script for GRPO reward functions.

Usage:
    python tests/test_rewards.py
    python tests/test_rewards.py --skip-roberta  # Skip slow roberta tests
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
)


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

    assert scores[0] == 2.0, f"Expected 2.0 for both words, got {scores[0]}"
    assert scores[1] == -1.0, f"Expected -1.0 for one word, got {scores[1]}"
    assert scores[2] == -2.0, f"Expected -2.0 for no words, got {scores[2]}"

    # Chinese tests
    zh_prompts = [
        "用这两个词生成一个有趣的笑话：'猫', '披萨'。",
        "用这两个词生成一个有趣的笑话：'狗', '电脑'。",
    ]
    zh_completions = [
        "猫咪看着披萨说：这是我的！",  # Both words
        "今天天气真好。",  # Neither word
    ]

    scores = word_pair_prompt_adherence(zh_completions, zh_prompts)
    print("\nChinese tests:")
    print(f"  Both words present: {scores[0]} (expected: 2.0)")
    print(f"  No words present:   {scores[1]} (expected: -2.0)")

    assert scores[0] == 2.0, f"Expected 2.0 for both words, got {scores[0]}"
    assert scores[1] == -2.0, f"Expected -2.0 for no words, got {scores[1]}"

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

    assert scores[0] == 2.0, f"Expected 2.0 for both words, got {scores[0]}"
    assert scores[1] == -2.0, f"Expected -2.0 for no words, got {scores[1]}"

    # Headline task (should return None)
    headline_prompts = ["Generate a funny joke related to this headline: 'Stock market crashes'."]
    headline_completions = ["The stock market crashed so hard it needed a helmet."]
    scores = word_pair_prompt_adherence(headline_completions, headline_prompts)
    print("\nHeadline task (should skip):")
    print(f"  Score: {scores[0]} (expected: None)")
    assert scores[0] is None, f"Expected None for headline task, got {scores[0]}"

    print("\n[PASS] word_pair_prompt_adherence tests passed!")


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

    assert is_valid_single_joke_en(valid_en) == True, "Valid joke should pass"
    assert is_valid_single_joke_en(invalid_multiple_q) == False, "Multiple ? should fail"
    assert is_valid_single_joke_en(invalid_qa_format) == False, "Q:/A: format should fail"
    assert is_valid_single_joke_en(invalid_stacking) == False, "Multiple 'why' should fail"
    print("  [PASS] English validation tests passed!")

    # Chinese
    print("\nChinese validation:")
    valid_zh = "为什么猫喜欢披萨？因为它很好吃！"
    invalid_zh_multiple_q = "为什么？为什么？"
    invalid_zh_qa = "问：为什么？答：因为。"

    assert is_valid_single_joke_zh(valid_zh) == True, "Valid Chinese joke should pass"
    assert is_valid_single_joke_zh(invalid_zh_multiple_q) == False, "Multiple 为什么 should fail"
    assert is_valid_single_joke_zh(invalid_zh_qa) == False, "问：/答： format should fail"
    print("  [PASS] Chinese validation tests passed!")

    # Spanish
    print("\nSpanish validation:")
    valid_es = "¿Por qué el gato pidio pizza? Porque tenia hambre!"
    invalid_es_multiple = "¿Por qué? ¿Por qué otra vez?"

    assert is_valid_single_joke_es(valid_es) == True, "Valid Spanish joke should pass"
    assert is_valid_single_joke_es(invalid_es_multiple) == False, "Multiple por qué should fail"
    print("  [PASS] Spanish validation tests passed!")

    print("\n[PASS] All is_valid_single_joke tests passed!")


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
        ("I told my wife she was drawing her eyebrows too high. She looked surprised.", "qa-punchline"),
    ]

    for joke, expected in test_cases_en:
        result = extract_joke_structure(joke)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] '{joke[:40]}...' -> {result} (expected: {expected})")
        assert result == expected, f"Expected {expected}, got {result}"

    # Chinese
    print("\nChinese structures:")
    test_cases_zh = [
        ("为什么猫喜欢鱼？", "why-did"),
        ("什么叫真正的朋友？", "what-do-you-call"),
        ("这就像生活一样。", "observation"),
    ]

    for joke, expected in test_cases_zh:
        result = extract_joke_structure_zh(joke)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] '{joke}' -> {result} (expected: {expected})")
        assert result == expected, f"Expected {expected}, got {result}"

    # Spanish
    print("\nSpanish structures:")
    test_cases_es = [
        ("¿Por qué el gato come pizza?", "why-did"),
        ("¿Cómo se llama un pez sin ojos?", "what-do-you-call"),
        ("Toc toc! Quien es?", "knock-knock"),
    ]

    for joke, expected in test_cases_es:
        result = extract_joke_structure_es(joke)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] '{joke}' -> {result} (expected: {expected})")
        assert result == expected, f"Expected {expected}, got {result}"

    print("\n[PASS] All extract_joke_structure tests passed!")


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
        print(f"  Score: {score} - '{joke[:50]}...'")
        assert score == 1.0, f"Expected 1.0, got {score}"

    print("\nBad jokes (expected: -1.0):")
    for joke, score in zip(bad_jokes, bad_scores):
        print(f"  Score: {score} - '{joke[:50]}...'")
        assert score == -1.0, f"Expected -1.0, got {score}"

    print("\n[PASS] formatting tests passed!")


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

    assert scores[0] == -2.0, f"Expected -2.0 for too short, got {scores[0]}"
    assert scores[1] == 0.0, f"Expected 0.0 for optimal, got {scores[1]}"
    assert scores[2] == -2.0, f"Expected -2.0 for too long, got {scores[2]}"

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

    assert scores[0] == -2.0, f"Expected -2.0 for too short, got {scores[0]}"
    assert scores[2] == -2.0, f"Expected -2.0 for too long, got {scores[2]}"

    print("\n[PASS] length_penalty tests passed!")


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
    assert scores[0] is None, f"Expected None, got {scores[0]}"

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

    assert scores[0] == 1.0, f"Expected 1.0, got {scores[0]}"
    assert scores[1] == -1.0, f"Expected -1.0, got {scores[1]}"
    assert scores[2] == -1.0, f"Expected -1.0, got {scores[2]}"

    print("\n[PASS] headline_adherence tests passed!")


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

    assert scores[0] == 0.0, f"Expected 0.0, got {scores[0]}"
    assert scores[1] < 0, f"Expected negative, got {scores[1]}"

    print("\n[PASS] coherence_penalty tests passed!")


def test_roberta_score(model_id: str = "KonradBRG/joke-rater-xlm-roberta"):
    """Test roberta scoring on sample jokes."""
    print("\n" + "="*60)
    print(f"Testing roberta_score with {model_id}")
    print("="*60)

    roberta_fn = create_roberta_score_fn(model_id, language="en")

    jokes = [
        "Why did the scarecrow win an award? Because he was outstanding in his field!",
        "I told my wife she was drawing her eyebrows too high. She looked surprised.",
        "What do you call a fake noodle? An impasta!",
        "This is not a joke at all, just a regular sentence.",
        "Why? Why? Why did this happen? Q: What? A: Nothing.",  # Invalid format
    ]

    print("\nScoring jokes...")
    scores = roberta_fn(jokes)

    print("\nResults:")
    for joke, score in zip(jokes, scores):
        print(f"  Score: {score:.3f} - '{joke[:60]}...'")

    # The invalid joke should get 0.0 (filtered out)
    print(f"\nInvalid joke score: {scores[4]} (expected: 0.0 due to validation)")

    print("\n[PASS] roberta_score test completed!")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test reward functions")
    parser.add_argument("--skip-roberta", action="store_true", help="Skip slow roberta tests")
    args = parser.parse_args()

    print("="*60)
    print("REWARD FUNCTION TESTS")
    print("="*60)

    # Run all tests
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

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    main()
