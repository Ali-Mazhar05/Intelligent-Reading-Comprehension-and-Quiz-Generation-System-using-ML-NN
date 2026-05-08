import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from preprocessing import FeatureEngineer, load_data

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
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # extract the sentences as hints
        hints = [sentences[i] for i in top_indices]
        
        return hints

class DistractorGenerator:
    def __init__(self, feature_engineer):
        # store feature engineer
        self.feature_engineer = feature_engineer
        
    def generate_distractors(self, article, correct_answer, num_distractors=3):
        # split article into candidate words
        words = str(article).lower().split()
        
        # remove short and common words
        candidates = list(set([w for w in words if len(w) > 4]))
        
        # return empty if not enough candidates
        if len(candidates) < num_distractors:
            return candidates
            
        # vectorize correct answer
        ans_features, _ = self.feature_engineer.transform_corpus([correct_answer])
        
        # vectorize candidates
        cand_features, _ = self.feature_engineer.transform_corpus(candidates)
        
        # compute cosine similarity
        similarities = cosine_similarity(ans_features, cand_features)[0]
        
        # to find distractors we want related but not identical words
        # so we pick high similarity but not exactly 1.0
        distractor_scores = []
        for i, cand in enumerate(candidates):
            sim = similarities[i]
            # penalize if it is exactly the correct answer
            if cand in str(correct_answer).lower():
                sim = -1
            distractor_scores.append((sim, cand))
            
        # sort candidates by adjusted similarity
        distractor_scores.sort(reverse=True, key=lambda x: x[0])
        
        # extract top distractors
        distractors = [cand for score, cand in distractor_scores[:num_distractors]]
        
        return distractors

def train_and_save_model_b():
    # initialize and load feature engineer
    print("loading feature engineer...")
    feature_engineer = FeatureEngineer(use_tf_idf=False)
    feature_engineer.load_models()
    
    # model b primarily uses heuristic ranking based on vector representations
    # we initialize the generators here and save them as components if needed
    print("initializing distractor and hint generators...")
    hint_gen = HintGenerator(feature_engineer)
    dist_gen = DistractorGenerator(feature_engineer)
    
    # create save directory
    os.makedirs('models/model_b/traditional', exist_ok=True)
    
    # save generators
    print("saving model b components...")
    with open('models/model_b/traditional/hint_generator.pkl', 'wb') as file:
        pickle.dump(hint_gen, file)
        
    with open('models/model_b/traditional/distractor_generator.pkl', 'wb') as file:
        pickle.dump(dist_gen, file)
        
    print("model b setup complete.")

if __name__ == '__main__':
    train_and_save_model_b()
