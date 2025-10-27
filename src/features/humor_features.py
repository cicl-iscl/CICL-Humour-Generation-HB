import nltk
import numpy as np
import re
import string
import torch

from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)


class HumourFeatureExtractor:
    """
    Extracts 10 different language-based humour features from text and converts them into a numerical vector.
    Based on https://arxiv.org/pdf/2402.01759
    """
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.intensifiers = {'very', 'so', 'extremely', 'incredibly', 'absolutely', 
                            'definitely', 'really', 'quite', 'rather', 'such'}
        self.self_deprecating_pronouns = {'i', 'me', 'my', 'we', 'us', 'our'}
    
    def extract_all_features(self, text):
        """Extract all 10 humour features from text"""
        features = {
            'ambiguity': self.extract_ambiguity(text),
            'phonetic_style': self.extract_phonetic_style(text),
            'exaggeration': self.extract_exaggeration(text),
            'pos_features': self.extract_pos_features(text),
            'ngram_features': self.extract_ngram_features(text),
            'punctuation_features': self.extract_punctuation_features(text),
            'tfidf_features': self.extract_tfidf_features(text),
            'semantic_similarity': self.extract_semantic_similarity(text),
            'contextual_information': self.extract_contextual_information(text),
            'incongruity': self.extract_incongruity(text)
        }
        return features
    
    def features_to_vector(self, text):
        """Convert all features into a single numerical vector"""
        features = self.extract_all_features(text)
        vector = []
        
        # Feature 1: Ambiguity (3 values)
        vector.extend([
            features['ambiguity']['ambiguous_words_count'],
            features['ambiguity']['ambiguity_score']
        ])
        
        # Feature 4: Phonetic Style (4 values)
        vector.extend([
            features['phonetic_style']['alliteration_count'],
            features['phonetic_style']['rhyme_count'],
            features['phonetic_style']['phonetic_score']
        ])
        
        # Feature 5: Exaggeration (4 values)
        vector.extend([
            features['exaggeration']['intensifier_count'],
            features['exaggeration']['caps_count'],
            features['exaggeration']['exaggeration_score']
        ])
        
        # Feature 6: POS Features (2 values)
        vector.extend([
            features['pos_features']['self_deprecating_count'],
            features['pos_features']['pronouns_score']
        ])
        
        # POS distribution (top 10 POS tags)
        pos_dist = features['pos_features']['pos_distribution']
        all_pos_tags = ['NN', 'VB', 'JJ', 'RB', 'PRP', 'DT', 'IN', 'CC', 'NNS', 'VBZ']
        pos_vector = [pos_dist.get(tag, 0) for tag in all_pos_tags]
        vector.extend(pos_vector)
        
        # Feature 7: N-gram Features (2 values)
        vector.extend([
            features['ngram_features']['bigram_count'],
            features['ngram_features']['trigram_count'],
            features['ngram_features']['ngram_diversity']
        ])
        
        # Feature 8: Punctuation Features (7 values)
        punc = features['punctuation_features']['punctuation_counts']
        vector.extend([
            punc['exclamation'],
            punc['question_marks'],
            punc['ellipsis'],
            punc['commas'],
            punc['periods'],
            punc['quotes'],
            features['punctuation_features']['punctuation_density']
        ])
        
        # Feature 9: TF-IDF (2 values)
        vector.extend([
            features['tfidf_features']['mean_tfidf'],
            features['tfidf_features']['max_tfidf']
        ])
        
        # Feature 10: Semantic Similarity (1 value)
        vector.append(features['semantic_similarity']['mean_similarity'])
        
        # Feature 2: Contextual Information (3 values)
        vector.extend([
            features['contextual_information']['avg_sentence_length'],
            features['contextual_information']['lexical_diversity']
        ])
        
        # Feature 3: Incongruity (1 value)
        vector.append(features['incongruity']['incongruity_score'])
        
        return torch.tensor(vector, dtype=torch.float32)
    
    def extract_ambiguity(self, text):
        """Feature 1: Ambiguity - multiple possible interpretations"""
        tokens = word_tokenize(text.lower())
        ambiguous_words = []
        
        for token in tokens:
            if token not in self.stop_words and token not in string.punctuation:
                synsets = wordnet.synsets(token)
                if len(synsets) > 1:
                    ambiguous_words.append({
                        'word': token,
                        'num_meanings': len(synsets),
                        'definitions': [s.definition() for s in synsets if s is not None]
                    })
        
        return {
            'ambiguous_words_count': len(ambiguous_words),
            'ambiguous_words': ambiguous_words,
            'ambiguity_score': len(ambiguous_words) / max(len(tokens), 1)
        }
    
    def extract_phonetic_style(self, text):
        """Feature 4: Phonetic Style - alliteration and rhyme"""
        tokens = word_tokenize(text.lower())
        
        # Alliteration detection
        alliterations = []
        for i in range(len(tokens) - 1):
            if tokens[i][0] == tokens[i + 1][0] and tokens[i] not in string.punctuation:
                alliterations.append(f"{tokens[i]}-{tokens[i + 1]}")
        
        # Rhyme detection (simple: same ending)
        rhymes = []
        for i in range(len(tokens) - 1):
            if len(tokens[i]) > 2 and len(tokens[i + 1]) > 2:
                if tokens[i][-2:] == tokens[i + 1][-2:] and tokens[i] != tokens[i + 1]:
                    rhymes.append(f"{tokens[i]}-{tokens[i + 1]}")
        
        return {
            'alliterations': alliterations,
            'alliteration_count': len(alliterations),
            'rhymes': rhymes,
            'rhyme_count': len(rhymes),
            'phonetic_score': (len(alliterations) + len(rhymes)) / max(len(tokens), 1)
        }
    
    def extract_exaggeration(self, text):
        """Feature 5: Exaggeration - intensifiers and emphasis"""
        tokens = word_tokenize(text.lower())
        pos_tags = pos_tag(tokens)
        
        intensifiers_found = []
        for token, pos in pos_tags:
            if token in self.intensifiers:
                intensifiers_found.append(token)
            # Check for adverbs, adjectives ending in -ly, -est
            elif pos in ['RB', 'RBR', 'RBS', 'JJ', 'JJR', 'JJS']:
                if token not in self.stop_words:
                    intensifiers_found.append(token)
        
        # Check for ALL CAPS words (emphasis)
        caps_words = [word for word in text.split() if word.isupper() and len(word) > 1]
        
        return {
            'intensifiers': intensifiers_found,
            'intensifier_count': len(intensifiers_found),
            'caps_emphasis': caps_words,
            'caps_count': len(caps_words),
            'exaggeration_score': (len(intensifiers_found) + len(caps_words)) / max(len(tokens), 1)
        }
    
    def extract_pos_features(self, text):
        """Feature 6: Part-of-Speech features"""
        tokens = word_tokenize(text.lower())
        pos_tags = pos_tag(tokens)
        
        # Count POS categories
        pos_counts = Counter([pos for _, pos in pos_tags])
        
        # Find self-deprecating pronouns
        self_depr_pronouns = [token for token, pos in pos_tags 
                             if token in self.self_deprecating_pronouns and pos == 'PRP']
        
        return {
            'pos_tags': pos_tags,
            'pos_distribution': dict(pos_counts),
            'self_deprecating_pronouns': self_depr_pronouns,
            'self_deprecating_count': len(self_depr_pronouns),
            'pronouns_score': len(self_depr_pronouns) / max(len(tokens), 1)
        }
    
    def extract_ngram_features(self, text):
        """Feature 7: N-gram based features"""
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t not in string.punctuation and t not in self.stop_words]
        
        # Bigrams (2-grams)
        bigrams = [f"{tokens[i]}-{tokens[i+1]}" for i in range(len(tokens) - 1)]
        bigram_freq = Counter(bigrams)
        
        # Trigrams (3-grams)
        trigrams = [f"{tokens[i]}-{tokens[i+1]}-{tokens[i+2]}" for i in range(len(tokens) - 2)]
        trigram_freq = Counter(trigrams)
        
        return {
            'bigrams': bigram_freq.most_common(5),
            'bigram_count': len(bigram_freq),
            'trigrams': trigram_freq.most_common(5),
            'trigram_count': len(trigram_freq),
            'ngram_diversity': (len(bigram_freq) + len(trigram_freq)) / max(len(tokens), 1)
        }
    
    def extract_punctuation_features(self, text):
        """Feature 8: Punctuation features"""
        punctuation_marks = {
            'exclamation': text.count('!'),
            'question_marks': text.count('?'),
            'ellipsis': text.count('...'),
            'commas': text.count(','),
            'periods': text.count('.'),
            'quotes': text.count('"') + text.count("'")
        }
        
        return {
            'punctuation_counts': punctuation_marks,
            'total_punctuation': sum(punctuation_marks.values()),
            'punctuation_density': sum(punctuation_marks.values()) / max(len(text), 1)
        }
    
    def extract_tfidf_features(self, text):
        """Feature 9: TF-IDF features"""
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        
        try:
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top TF-IDF scores
            tfidf_scores = tfidf_matrix.toarray()[0]
            top_indices = tfidf_scores.argsort()[-10:][::-1]
            
            top_terms = [(feature_names[i], tfidf_scores[i]) for i in top_indices if tfidf_scores[i] > 0]
            
            return {
                'top_tfidf_terms': top_terms,
                'mean_tfidf': tfidf_scores[tfidf_scores > 0].mean() if any(tfidf_scores > 0) else 0,
                'max_tfidf': tfidf_scores.max()
            }
        except:
            return {'top_tfidf_terms': [], 'mean_tfidf': 0, 'max_tfidf': 0}
    
    def extract_semantic_similarity(self, text):
        """Feature 10: Semantic similarity - basic word relatedness"""
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t not in string.punctuation and t not in self.stop_words]
        
        similarity_pairs = []
        for i in range(len(tokens) - 1):
            synsets1 = wordnet.synsets(tokens[i])
            synsets2 = wordnet.synsets(tokens[i + 1])
            
            if synsets1 and synsets2:
                max_sim = max(synsets1[0].path_similarity(s) for s in synsets2 
                             if synsets1[0].path_similarity(s) is not None)
                similarity_pairs.append({
                    'pair': f"{tokens[i]}-{tokens[i+1]}",
                    'similarity': max_sim
                })
        
        return {
            'similar_word_pairs': similarity_pairs,
            'mean_similarity': sum([p['similarity'] for p in similarity_pairs]) / max(len(similarity_pairs), 1) 
                              if similarity_pairs else 0
        }
    
    def extract_contextual_information(self, text):
        """Feature 2: Contextual information"""
        sentences = sent_tokenize(text)
        tokens = word_tokenize(text.lower())
        
        return {
            'sentence_count': len(sentences),
            'token_count': len(tokens),
            'avg_sentence_length': len(tokens) / max(len(sentences), 1),
            'unique_tokens': len(set(tokens)),
            'lexical_diversity': len(set(tokens)) / max(len(tokens), 1)
        }
    
    def extract_incongruity(self, text):
        """Feature 3: Incongruity - unexpected contradictions"""
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t not in string.punctuation and t not in self.stop_words]
        
        # Simple incongruity: detect contrasts and contradictions
        contrasts = {'but', 'however', 'yet', 'though', 'although', 'while', 'whereas'}
        contrast_count = sum(1 for t in tokens if t in contrasts)
        
        # Repeated words (unusual repetition can signal incongruity)
        word_freq = Counter(tokens)
        repeated_words = [word for word, count in word_freq.items() if count > 2]
        
        return {
            'contrast_markers': contrast_count,
            'repeated_words': repeated_words,
            'incongruity_score': (contrast_count + len(repeated_words)) / max(len(tokens), 1)
        }


# Example usage
if __name__ == "__main__":
    extractor = HumourFeatureExtractor()
    
    # Example text
    sample_text = "I'm not saying I'm Batman, but have you ever seen me and Batman in the same room? Absolutely NOT!"
    
    features = extractor.features_to_vector(sample_text)
    print("Feature Vector:", features)
    print("Shape:", features.shape)