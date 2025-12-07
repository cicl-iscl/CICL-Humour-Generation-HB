import emoji
import re
from collections import deque
import torch
from transformers import pipeline


def is_valid_single_joke(text):
    """
    Combat joke stacking.
    """
    if text.count("?") > 1:
        return False
    # Using 'why ' instead of 'why ' is crucial due to tokenization/word boundaries
    if text.lower().count("why ") > 1:
        return False
    if "Q:" in text.lower() or "A:" in text.lower():
        return False
    if '\"' in text or '-' in text: # This is a strong heuristic; consider refining it if it's too aggressive.
        return False
    if len(text.strip().split('\n')) > 3:
        return False
    
    return True

recent_structures = deque(maxlen=30) 

def extract_joke_structure(joke: str) -> str:
    """Extracts common joke structures using regex."""
    joke_lower = joke.lower()
    
    if re.search(r"why\s+(?:did|didn't|do|dont|don't|were|weren't|was|wasn't)\s+\w+", joke_lower):
        return "why-did"
    elif re.search(r"where\s+(?:did|didn't|do|dont|don't|were|weren't|was|wasn't)\s+\w+", joke_lower):
        return "where-did"
    elif re.search(r"how\s+(?:did|didn't|do|dont|don't|were|weren't|was|wasn't)\s+\w+", joke_lower):
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
        s = extract_joke_structure(joke)
        actual = freq.get(s, 0) / total
        reward = target - actual
        scores.append(reward)
        
        # Update history and frequency for the NEXT joke/step
        recent_structures.append(s)
        freq[s] = freq.get(s, 0) + 1
        total += 1
    
    return scores

scoring_pipe = None

def roberta_score(completions, **kwargs):
    """
    Scores humor using RoBERTA, but only for valid outputs (cell 7).
    Invalid outputs get 0.0 regardless of what the model thinks.
    """
    global scoring_pipe
    if scoring_pipe is None:
        # Load the pipeline on the first call
        print("Loading RoBERTA Joke Rater Pipeline...")
        scoring_pipe = pipeline(
            "text-classification", 
            model="KonradBRG/RoBERTA-Joke-Rater", 
            trust_remote_code=True,
            device=0 if torch.cuda.is_available() else -1
        )
        print("RoBERTA Pipeline loaded.")

    scores = []
    # Only score completions that are valid jokes to reduce computation
    valid_completions = [c for c in completions if is_valid_single_joke(c)]
    valid_indices = [i for i, c in enumerate(completions) if is_valid_single_joke(c)]
    
    # Initialize all scores to 0.0 (for invalid jokes)
    final_scores = [0.0] * len(completions)
    
    if valid_completions:
        # Get scores for valid jokes
        roberta_labels = scoring_pipe(valid_completions)
        for idx, roberta_label in zip(valid_indices, roberta_labels):
            roberta_score = float(roberta_label["label"])
            final_scores[idx] = roberta_score
            
    return final_scores


def word_pair_prompt_adherence(completions, prompts, **kwargs):
    """Enforces the word pair constraint (cell 8)."""
    scores = []
    pattern = r"contains\s+these\s+two\s+words:\s*'([^']+)'\s*,\s*'([^']+)'"
    for i in range(len(completions)):
        p = prompts[i]
        # Skip if not a word pair task
        if "two words" not in p:
            scores.append(None)
            continue
        
        c = completions[i].lower()
        try:
            # Extract the two required words from the prompt
            w1, w2 = re.findall(pattern, p, flags=re.IGNORECASE)[0]
        except IndexError:
            scores.append(None)
            continue
        
        w1_lower = w1.lower().strip()
        w2_lower = w2.lower().strip()
        
        # Check if words are found in the completion
        w1_found = w1_lower in c
        w2_found = w2_lower in c
        
        # Reward/Penalty logic:
        if w1_found and w2_found:
            scores.append(2.0) # High positive reward for both
        elif w1_found or w2_found:
            scores.append(-1.0) # Penalty for only one
        else:
            scores.append(-2.0) # High penalty for neither
    
    return scores

def headline_adherence(completions, prompts, **kwargs):
    """Simple check for headline tasks (cell 8)."""
    scores = []
    for i, completion in enumerate(completions):
        
        # Skip if it's a word pair task
        if "two words" in prompts[i]: 
            scores.append(None)
            continue
        
        # The prompt is a headline task.
        if len(completion.split()) <= 25: 
            # Check for bad patterns (like repeating the prompt or being conversational)
            if "headline" in completion.lower() or "generate" in completion.lower():
                scores.append(-1.0)
            else:
                scores.append(1.0) # Small positive reward for conforming to length/format
        else:
            scores.append(-1.0)
    return scores

def contains_emoji_func(text):
    """Helper to detect emojis."""
    return any(char in emoji.EMOJI_DATA for char in text)

def formatting(completions, **kwargs):
    """Validates output formatting and penalizes hacking patterns (cell 9)."""
    scores = []
    for completion in completions:
        is_penalized = False
        # Penalties for conversational artifacts/bad symbols
        if ("#" in completion
            or "How about: " in completion
            or "This joke" in completion
            or "Let me know" in completion
            or "Note: " in completion
            or "   " in completion # Multiple spaces
            or contains_emoji_func(completion)
        ):
            scores.append(-1.0)
            is_penalized = True
        # Penalty for joke stacking/invalid structure
        elif not is_valid_single_joke(completion):
            scores.append(-1.0)
            is_penalized = True
        
        if not is_penalized:
            scores.append(1.0)
    
    return scores

def length_penalty(completions, **kwargs):
    """Penalizes outputs outside the optimal 5-24 word range (cell 9)."""
    scores = []
    optimal_length = 16
    max_allowed = 24
    min_allowed = 5
    
    for completion in completions:
        word_count = len(completion.split())
        
        if word_count > max_allowed or word_count < min_allowed:
            scores.append(-2.0) # Severe penalty
        else:
            # Smooth penalty for being over optimal length (16 words)
            deviation = max(0, word_count - optimal_length)
            penalty = -0.2 * deviation
            scores.append(penalty)
    
    return scores

def compute_coherence_penalty(joke: str, penalty_weight: float = 0.5) -> float:
    """Penalize incoherent jokes with rare/technical terms (cell 9)."""
    rare_word_pattern = r'\b[A-Z][a-z]*(?:-[a-z]+)*\b'
    rare_words = len(re.findall(rare_word_pattern, joke))
    
    words = joke.split()
    if len(words) > 0:
        rare_word_ratio = rare_words / len(words)
        # Penalize if capitalized words make up more than 20% of the joke
        if rare_word_ratio > 0.2: 
            return -penalty_weight * (rare_word_ratio - 0.2)
    
    return 0.0

def coherence_penalty(completions, **kwargs):
    """Wrapper for the coherence penalty """
    return [compute_coherence_penalty(c) for c in completions]

