import os
import emoji
import re
from collections import deque
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer


def is_valid_single_joke_en(text):
    """Combat joke stacking (English)."""
    if text.count("?") > 1:
        return False
    if text.lower().count("why ") > 1:
        return False
    if "q:" in text.lower() or "a:" in text.lower():
        return False
    if len(text.strip().split("\n")) > 3:
        return False
    return True


def is_valid_single_joke_zh(text):
    """Combat joke stacking (Chinese)."""
    if text.count("？") > 1:
        return False
    if text.count("为什么") > 1:
        return False
    if "问：" in text or "答：" in text:
        return False
    if len(text.strip().split("\n")) > 3:
        return False
    return True


def is_valid_single_joke_es(text):
    """Combat joke stacking (Spanish)."""
    if text.count("?") + text.count("¿") > 2:  # ¿...? counts as one question
        return False
    if text.lower().count("por qué") > 1:
        return False
    if "p:" in text.lower() or "r:" in text.lower():  # Pregunta/Respuesta
        return False
    if len(text.strip().split("\n")) > 3:
        return False
    return True


def create_is_valid_single_joke_fn(language: str = "en"):
    """Factory function to get the appropriate validator for a language."""
    if language == "zh":
        return is_valid_single_joke_zh
    elif language == "es":
        return is_valid_single_joke_es
    return is_valid_single_joke_en


# Default for backwards compatibility
is_valid_single_joke = is_valid_single_joke_en


recent_structures = deque(maxlen=30)


def reset_recent_structures():
    """Reset the global recent_structures deque. Useful for testing."""
    global recent_structures
    recent_structures.clear()


def extract_joke_structure(joke: str) -> str:
    """Extracts common joke structures using regex (English)."""
    joke_lower = joke.lower()

    if re.search(
        r"why\s+(?:did|didn't|do|dont|don't|were|weren't|was|wasn't)\s+\w+", joke_lower
    ):
        return "why-did"
    elif re.search(
        r"where\s+(?:did|didn't|do|dont|don't|were|weren't|was|wasn't)\s+\w+",
        joke_lower,
    ):
        return "where-did"
    elif re.search(
        r"how\s+(?:did|didn't|do|dont|don't|were|weren't|was|wasn't)\s+\w+", joke_lower
    ):
        return "how-did"
    elif re.search(r"what\s+do\s+you\s+call", joke_lower):
        return "what-do-you-call"
    elif re.search(r"knock\s+knock", joke_lower):
        return "knock-knock"
    elif joke.count("?") == 1 and (joke.count("!") >= 1 or joke.count(".") >= 1):
        return "qa-punchline"
    elif any(phrase in joke_lower for phrase in [" is like ", " is when "]):
        return "observation"
    else:
        return "one-liner"


def extract_joke_structure_zh(joke: str) -> str:
    """Extracts common joke structures using regex (Chinese)."""
    # Chinese patterns for common joke structures
    if re.search(r"为什么", joke):
        return "why-did"
    elif re.search(r"(在哪|哪里|哪儿)", joke):
        return "where-did"
    elif re.search(r"怎么", joke):
        return "how-did"
    elif re.search(r"(什么叫|叫什么|算什么)", joke):
        return "what-do-you-call"
    elif re.search(r"(敲门|咚咚)", joke):
        return "knock-knock"
    elif "？" in joke and ("！" in joke or "。" in joke):
        return "qa-punchline"
    elif any(phrase in joke for phrase in ["就像", "好比", "仿佛"]):
        return "observation"
    else:
        return "one-liner"


def extract_joke_structure_es(joke: str) -> str:
    """Extracts common joke structures using regex (Spanish)."""
    joke_lower = joke.lower()

    if re.search(r"(¿)?por\s*qué", joke_lower):
        return "why-did"
    elif re.search(r"(¿)?dónde", joke_lower):
        return "where-did"
    elif re.search(r"(¿)?cómo", joke_lower):
        return "how-did"
    elif re.search(r"(¿)?cómo\s+se\s+llama", joke_lower) or re.search(r"(¿)?qué\s+es", joke_lower):
        return "what-do-you-call"
    elif re.search(r"toc\s*toc", joke_lower):
        return "knock-knock"
    elif ("?" in joke or "¿" in joke) and ("!" in joke or "¡" in joke or "." in joke):
        return "qa-punchline"
    elif any(phrase in joke_lower for phrase in [" es como ", " parece ", " cuando "]):
        return "observation"
    else:
        return "one-liner"


def create_structure_diversity_reward_fn(language: str = "en"):
    """
    Factory function that creates a structure_diversity_reward function for a specific language.

    Args:
        language: Language code ("en", "zh", or "es").

    Returns:
        A structure_diversity_reward function configured for the specified language.
    """
    # Select the appropriate structure extractor based on language
    if language == "zh":
        extractor = extract_joke_structure_zh
    elif language == "es":
        extractor = extract_joke_structure_es
    else:
        extractor = extract_joke_structure

    def structure_diversity_reward(completions, **kwargs):
        """Calculates reward based on structure novelty relative to recent history (cell 6)."""
        global recent_structures
        scores = []
        freq = {}

        # Calculate frequencies of existing structures in the history
        for s in recent_structures:
            freq[s] = freq.get(s, 0) + 1

        total = max(len(recent_structures), 1)
        num_structures = len(freq) if freq else 1
        # Target is equal distribution (e.g., if 5 structures, target is 1/5 = 0.2)
        target = 1 / num_structures

        for joke in completions:
            s = extractor(joke)
            actual = freq.get(s, 0) / total
            reward = target - actual
            scores.append(reward)

            # Update history and frequency for the NEXT joke/step
            recent_structures.append(s)
            freq[s] = freq.get(s, 0) + 1
            total += 1

        return scores

    return structure_diversity_reward


# Default structure_diversity_reward for backwards compatibility
structure_diversity_reward = create_structure_diversity_reward_fn("en")


def _get_current_device():
    """Get the current device for this process in distributed training."""
    if torch.cuda.is_available():
        # In distributed training, each process should use its assigned GPU
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return local_rank
    return -1


def create_roberta_score_fn(model_id: str = "KonradBRG/joke-rater-xlm-roberta", language: str = "en"):
    """
    Factory function that creates a roberta_score function configured with a specific model.

    Args:
        model_id: HuggingFace model ID for the joke rater model.
        language: Language code ("en", "zh", or "es") for validation.

    Returns:
        A roberta_score function that uses the specified model.
    """
    scoring_pipe = None
    _scoring_pipe_device = None
    validator = create_is_valid_single_joke_fn(language)

    def roberta_score(completions, **kwargs):
        """
        Scores humor using the joke rater model, but only for valid outputs.
        Invalid outputs get 0.0 regardless of what the model thinks.

        NOTE: In distributed training, this loads the model on each process's GPU.
        """
        nonlocal scoring_pipe, _scoring_pipe_device

        current_device = _get_current_device()

        # Load or reload pipeline if device changed (shouldn't happen, but safety check)
        if scoring_pipe is None or _scoring_pipe_device != current_device:
            print(f"Loading Joke Rater Pipeline ({model_id}) on device {current_device}...")
            # Use AutoModelForSequenceClassification to load custom model class from repo
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
            scoring_pipe = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=current_device,
                trust_remote_code=True,
            )
            _scoring_pipe_device = current_device
            print("Joke Rater Pipeline loaded.")

        # Only score completions that are valid jokes to reduce computation
        valid_completions = [c for c in completions if validator(c)]
        valid_indices = [i for i, c in enumerate(completions) if validator(c)]
        final_scores = [0.0] * len(completions)

        if valid_completions:
            # Get scores for valid jokes
            roberta_labels = scoring_pipe(valid_completions)
            for idx, roberta_label in zip(valid_indices, roberta_labels):
                score = float(roberta_label["label"])
                final_scores[idx] = score

        return final_scores

    return roberta_score


# Default roberta_score for backwards compatibility
roberta_score = create_roberta_score_fn()


def word_pair_prompt_adherence(completions, prompts, **kwargs):
    """Enforces the word pair constraint - works for EN/ZH/ES prompts."""
    scores = []
    # Patterns for different languages
    patterns = [
        # English: "using these two words: 'w1', 'w2'" or "contains these two words: 'w1', 'w2'"
        r"(?:using|contains)\s+these\s+two\s+words:\s*'([^']+)'\s*,\s*'([^']+)'",
        # Chinese: "用这两个词...：'w1'、'w2'"
        r"用这两个词[^：]*：\s*'([^']+)'[、,]\s*'([^']+)'",
        # Spanish: "usando estas dos palabras: 'w1', 'w2'"
        r"usando\s+estas\s+dos\s+palabras:\s*'([^']+)'\s*,\s*'([^']+)'",
    ]

    for i in range(len(completions)):
        p = prompts[i]

        # Try to find word pairs using any pattern
        w1, w2 = None, None
        for pattern in patterns:
            matches = re.findall(pattern, p, flags=re.IGNORECASE)
            if matches:
                w1, w2 = matches[0]
                break

        # Skip if no word pair found (headline task or unknown format)
        if w1 is None or w2 is None:
            scores.append(None)
            continue

        c = completions[i].lower()
        w1_lower = w1.lower().strip()
        w2_lower = w2.lower().strip()

        # Check if words are found in the completion
        w1_found = w1_lower in c
        w2_found = w2_lower in c

        # Reward/Penalty logic:
        if w1_found and w2_found:
            scores.append(2.0)  # High positive reward for both
        elif w1_found or w2_found:
            scores.append(-1.0)  # Penalty for only one
        else:
            scores.append(-2.0)  # High penalty for neither

    return scores


def create_headline_adherence_fn(language: str = "en"):
    """
    Factory function that creates a headline adherence function for a specific language.
    """
    # Patterns to detect word-pair tasks (should skip these)
    word_pair_patterns = {
        "en": ["two words", "these words"],
        "zh": ["两个词", "这两个词"],
        "es": ["dos palabras", "estas palabras"],
    }
    skip_patterns = word_pair_patterns.get(language, word_pair_patterns["en"])

    # Bad patterns in completions (conversational artifacts)
    bad_patterns_by_lang = {
        "en": ["headline", "generate"],
        "zh": ["标题", "生成"],
        "es": ["titular", "genera"],
    }
    bad_patterns = bad_patterns_by_lang.get(language, bad_patterns_by_lang["en"])

    # Max length (words for en/es, characters for zh)
    max_length = {"en": 25, "zh": 50, "es": 28}[language]
    use_chars = language == "zh"

    def headline_adherence(completions, prompts, **kwargs):
        """Simple check for headline tasks (cell 8)."""
        scores = []
        for i, completion in enumerate(completions):
            # Skip if it's a word pair task
            if any(pattern in prompts[i].lower() for pattern in skip_patterns):
                scores.append(None)
                continue

            # Calculate length
            if use_chars:
                length = len(re.sub(r'\s', '', completion))
            else:
                length = len(completion.split())

            # The prompt is a headline task.
            if length <= max_length:
                # Check for bad patterns (like repeating the prompt or being conversational)
                if any(pattern in completion.lower() for pattern in bad_patterns):
                    scores.append(-1.0)
                else:
                    scores.append(1.0)  # Small positive reward for conforming to length/format
            else:
                scores.append(-1.0)
        return scores

    return headline_adherence


# Default for backwards compatibility
headline_adherence = create_headline_adherence_fn("en")


def contains_emoji_func(text):
    """Helper to detect emojis."""
    return any(char in emoji.EMOJI_DATA for char in text)


def contains_forbidden_symbols(text):
    """Check for forbidden symbols: #, @, emojis."""
    if "#" in text or "@" in text:
        return True
    if contains_emoji_func(text):
        return True
    return False


def create_formatting_fn(language: str = "en"):
    """
    Factory function that creates a formatting reward function for a specific language.
    """
    validator = create_is_valid_single_joke_fn(language)

    # Language-specific bad patterns
    bad_patterns_by_lang = {
        "en": [
            "How about: ",
            "This joke",
            "Let me know",
            "Note: ",
            "Here's",
            "Here's a joke for you:",
            "Sure!",
            "You're welcome!",
            "That joke",
            "Joke:"
        ],
        "zh": [
            # original ZH
            "这个笑话",
            "怎么样",
            "注意：",
            "提示：",
            "让我",
            "好的，",
            # translated EN
            "怎么样：",
            "这个笑话",
            "告诉我",
            "注意：",
            "这是",
            "这是一个笑话：",
            "当然！",
            "不客气！",
            "那个笑话",
        ],
        "es": [
            # original ES
            "Qué tal: ",
            "Este chiste",
            "Avísame",
            "Nota: ",
            "Aquí tienes",
            # translated EN
            "Qué tal: ",
            "Este chiste",
            "Avísame",
            "Nota: ",
            "Aquí está",
            "Aquí tienes un chiste:",
            "¡Claro!",
            "¡De nada!",
            "Ese chiste",
            "Chiste:"
        ],
    }

    bad_patterns = bad_patterns_by_lang.get(language, bad_patterns_by_lang["en"])

    def formatting(completions, **kwargs):
        """Validates output formatting and penalizes hacking patterns (cell 9)."""
        scores = []
        for completion in completions:
            # Heavy penalty for forbidden symbols (emojis, #, @)
            # These should NEVER appear in generated jokes
            if contains_forbidden_symbols(completion):
                scores.append(-5.0)
                continue

            # Medium penalty for conversational artifacts/bad patterns
            if (
                any(pattern in completion for pattern in bad_patterns)
                or "   " in completion  # Multiple spaces
            ):
                scores.append(-1.0)
                continue

            # Penalty for joke stacking/invalid structure
            if not validator(completion):
                scores.append(-1.0)
                continue

            # Good formatting
            scores.append(1.0)

        return scores

    return formatting


# Default for backwards compatibility
formatting = create_formatting_fn("en")


def create_length_penalty_fn(language: str = "en"):
    """
    Factory function that creates a length penalty function for a specific language.
    Chinese uses character count; English/Spanish use word count.
    """
    # Language-specific length parameters
    # Chinese: character-based (roughly 2-3 chars per "word equivalent")
    # English/Spanish: word-based
    length_params = {
        "en": {"min": 5, "max": 24, "optimal": 16, "use_chars": False},
        "zh": {"min": 10, "max": 60, "optimal": 35, "use_chars": True},
        "es": {"min": 5, "max": 28, "optimal": 18, "use_chars": False},  # Spanish words are longer
    }
    params = length_params.get(language, length_params["en"])

    def length_penalty(completions, **kwargs):
        """Penalizes outputs outside the optimal length range (cell 9)."""
        scores = []
        for completion in completions:
            if params["use_chars"]:
                # For Chinese: count characters (excluding spaces/punctuation)
                length = len(re.sub(r'\s', '', completion))
            else:
                # For English/Spanish: count words
                length = len(completion.split())

            if length > params["max"] or length < params["min"]:
                scores.append(-2.0)  # Severe penalty
            else:
                # Smooth penalty for being over optimal length
                deviation = max(0, length - params["optimal"])
                penalty = -0.2 * deviation
                scores.append(penalty)

        return scores

    return length_penalty


# Default for backwards compatibility
length_penalty = create_length_penalty_fn("en")


def compute_coherence_penalty_latin(joke: str, penalty_weight: float = 0.5) -> float:
    """Penalize incoherent jokes with rare/technical terms (Latin alphabet languages)."""
    rare_word_pattern = r"\b[A-Z][a-z]*(?:-[a-z]+)*\b"
    rare_words = len(re.findall(rare_word_pattern, joke))

    words = joke.split()
    if len(words) > 0:
        rare_word_ratio = rare_words / len(words)
        # Penalize if capitalized words make up more than 20% of the joke
        if rare_word_ratio > 0.2:
            return -penalty_weight * (rare_word_ratio - 0.2)

    return 0.0


def compute_coherence_penalty_zh(joke: str, penalty_weight: float = 0.5) -> float:
    """Penalize incoherent jokes (Chinese) - checks for excessive punctuation or repetition."""
    # Check for excessive punctuation (more than 10% of chars are punctuation)
    punct_pattern = r'[，。！？、；：""''【】《》（）]'
    punct_count = len(re.findall(punct_pattern, joke))
    char_count = len(re.sub(r'\s', '', joke))

    if char_count > 0:
        punct_ratio = punct_count / char_count
        if punct_ratio > 0.15:
            return -penalty_weight * (punct_ratio - 0.15)

    return 0.0


def create_coherence_penalty_fn(language: str = "en"):
    """
    Factory function that creates a coherence penalty function for a specific language.
    """
    if language == "zh":
        compute_fn = compute_coherence_penalty_zh
    else:
        compute_fn = compute_coherence_penalty_latin

    def coherence_penalty(completions, **kwargs):
        """Wrapper for the coherence penalty"""
        return [compute_fn(c) for c in completions]

    return coherence_penalty


# Default for backwards compatibility
coherence_penalty = create_coherence_penalty_fn("en")
