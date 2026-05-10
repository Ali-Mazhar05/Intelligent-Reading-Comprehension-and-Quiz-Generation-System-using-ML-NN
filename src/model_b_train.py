import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import string
from preprocessing import FeatureEngineer, load_data
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api

class HintGenerator:
    def __init__(self, feature_engineer):
        # store the feature engineer for vectorization
        self.feature_engineer = feature_engineer
        
    def generate_hints(self, article, question, top_k=3):
        # split article into sentences using simple regex
        sentences = re.split(r'(?<=[.!?]) +', str(article))
        
        # filter out very short sentences
        sentences = [s for s in sentences if len(s.split()) > 3]
        
        # if article has no valid sentences return empty
        if not sentences:
            return []
            
        # vectorize the question
        q_features, _ = self.feature_engineer.transform_corpus([question])
        
        # vectorize all sentences
        s_features, _ = self.feature_engineer.transform_corpus(sentences)
        
        # compute cosine similarity between question and sentences
        similarities = cosine_similarity(q_features, s_features)[0]
        
        # get indices of top k most similar sentences
        # we want graduated hints: General (3rd best) -> Specific (2nd best) -> Explicit (1st best)
        top_indices = similarities.argsort()[-top_k:]
        
        # extract the sentences as hints in increasing order of similarity
        hints = [sentences[i] for i in top_indices]
        
        return hints

class DistractorGenerator:
    def __init__(self, feature_engineer, ranker_model=None):
        # store feature engineer and optional ML ranker
        self.feature_engineer = feature_engineer
        self.ranker_model = ranker_model
        self.w2v_model = None
        
    def _load_w2v(self):
        """Try to load a pre-trained w2v model or train a small local one."""
        if self.w2v_model is not None:
            return
            
        try:
            # try to load a very small pre-trained model for speed
            print("Loading pre-trained Word2Vec (glove-twitter-25)...")
            self.w2v_model = api.load('glove-twitter-25')
        except Exception as e:
            print(f"Pre-trained model loading failed: {e}. Falling back to local training.")
            # local fallback is handled per-generation if needed
        
    def _extract_features(self, article, candidate, correct_answer, word_freqs):
        ans_lower = str(correct_answer).lower()
        cand_lower = str(candidate).lower()
        
        # 1. Cosine Similarity (OHE)
        ans_feat, _ = self.feature_engineer.transform_corpus([ans_lower])
        cand_feat, _ = self.feature_engineer.transform_corpus([cand_lower])
        sim = cosine_similarity(ans_feat, cand_feat)[0][0]
        
        # 2. Frequency Feature
        ans_freq = word_freqs.get(ans_lower, 1)
        cand_freq = word_freqs.get(cand_lower, 0)
        freq_score = 1.0 / (1.0 + abs(ans_freq - cand_freq))
        
        # 3. Length Features
        len_diff = abs(len(ans_lower) - len(cand_lower))
        
        return [sim, freq_score, len_diff, len(cand_lower)]

    def generate_distractors(self, article, correct_answer, num_distractors=3):
        blacklist = {
            'there', 'their', 'these', 'those', 'where', 'which', 'whose', 'while',
            'this', 'that', 'they', 'them', 'from', 'with', 'under', 'after',
            'before', 'into', 'each', 'every', 'some', 'many', 'much', 'very'
        }
        
        words = str(article).lower().split()
        from collections import Counter
        word_freqs = Counter([w.strip(string.punctuation) for w in words])
        
        candidates = list(set([w.strip(string.punctuation) for w in words 
                              if len(w) > 4 and w.lower() not in blacklist]))
        
        if len(candidates) < num_distractors:
            return candidates
            
        ans_lower = str(correct_answer).lower()
        distractor_scores = []
        
        # Limit candidates for speed if ranking with ML
        for cand in candidates[:100]:
            cand_lower = cand.lower()
            if cand_lower == ans_lower or cand_lower in ans_lower or ans_lower in cand_lower:
                continue
                
            features = self._extract_features(article, cand, correct_answer, word_freqs)
            
            if self.ranker_model:
                # Use ML Ranker (Random Forest)
                score = self.ranker_model.predict_proba([features])[0][1]
            else:
                # Fallback to Heuristic
                score = (0.7 * features[0]) + (0.3 * features[1])
                
            distractor_scores.append((score, cand))
            
        distractor_scores.sort(reverse=True, key=lambda x: x[0])
        distractors = [cand for score, cand in distractor_scores][:num_distractors]
        return distractors
        
    def generate_w2v_distractors(self, article, correct_answer, num_distractors=3):
        """Retrieve distractors using Word2Vec nearest neighbors (Rubric Requirement)."""
        self._load_w2v()
        ans_lower = str(correct_answer).lower().strip()
        
        # If model is loaded and answer is in vocab
        if self.w2v_model and ans_lower in self.w2v_model:
            # get nearest neighbors
            sims = self.w2v_model.most_similar(ans_lower, topn=10)
            # filter neighbors: not the answer, and not explicitly in the article (to be plausible but wrong)
            distractors = []
            article_words = set(str(article).lower().split())
            for word, score in sims:
                if word.lower() != ans_lower and word.lower() not in article_words:
                    distractors.append(word)
                if len(distractors) >= num_distractors:
                    break
            
            if len(distractors) >= num_distractors:
                return distractors
        
        # Fallback to standard generation if w2v fails
        return self.generate_distractors(article, correct_answer, num_distractors)

    def generate_fitb_distractors(self, article, target_word, num_distractors=3):
        return self.generate_distractors(article, target_word, num_distractors)

def train_and_save_model_b():
    print("loading feature engineer...")
    feature_engineer = FeatureEngineer(use_tf_idf=False)
    feature_engineer.load_models()
    
    print("loading training data for distractor ranker...")
    train_df = load_data('data/processed/train_clean.csv')
    
    # Prepare training set for the distractor ranker
    # We sample a subset for faster training
    subset_df = train_df.dropna(subset=['article', 'answer', 'clean_a', 'clean_b', 'clean_c', 'clean_d']).head(2000)
    
    X = []
    y = []
    
    temp_gen = DistractorGenerator(feature_engineer)
    
    print("engineering features for distractor ranker...")
    for _, row in subset_df.iterrows():
        article = row['article']
        correct_char = row['answer']
        ans_text = row[correct_char]
        
        # word frequencies in article
        words = str(article).lower().split()
        from collections import Counter
        word_freqs = Counter([w.strip(string.punctuation) for w in words])
        
        # Positive samples: actual distractors from the dataset
        dist_chars = [c for c in ['A', 'B', 'C', 'D'] if c != correct_char]
        for dc in dist_chars:
            dist_text = row[dc]
            if isinstance(dist_text, str) and len(dist_text) > 2:
                feats = temp_gen._extract_features(article, dist_text, ans_text, word_freqs)
                X.append(feats)
                y.append(1)
        
        # Negative samples: random words from the article that aren't distractors
        potential_negatives = [w.strip(string.punctuation) for w in words if len(w) > 5]
        for neg in potential_negatives[:3]:
            if neg.lower() != ans_text.lower():
                feats = temp_gen._extract_features(article, neg, ans_text, word_freqs)
                X.append(feats)
                y.append(0)
                
    print(f"training random forest ranker on {len(X)} samples...")
    ranker = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    ranker.fit(X, y)
    
    print("initializing distractor and hint generators...")
    hint_gen = HintGenerator(feature_engineer)
    dist_gen = DistractorGenerator(feature_engineer, ranker_model=ranker)
    
    os.makedirs('models/model_b/traditional', exist_ok=True)
    
    print("saving model b components...")
    with open('models/model_b/traditional/hint_generator.pkl', 'wb') as file:
        pickle.dump(hint_gen, file)
        
    with open('models/model_b/traditional/distractor_generator.pkl', 'wb') as file:
        pickle.dump(dist_gen, file)
        
    print("model b setup complete with supervised ranker.")

if __name__ == '__main__':
    train_and_save_model_b()

if __name__ == '__main__':
    train_and_save_model_b()
