import pandas as pd
import numpy as np
import os
import pickle
import re
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
from preprocessing import load_data, FeatureEngineer

# ensure nltk resources are available
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

class QuestionGenerator:
    def __init__(self, ranker_model, feature_engineer):
        # store the trained ranker (e.g. svm) and feature engineer
        self.ranker_model = ranker_model
        self.feature_engineer = feature_engineer
        
    def generate_question(self, article, correct_answer):
        # step 1: extract candidate sentences from passage using keyword overlap
        sentences = re.split(r'(?<=[.!?]) +', str(article))
        ans_words = set(str(correct_answer).lower().split())
        
        best_sentence = ""
        max_overlap = -1
        
        # find sentence with maximum overlap with the correct answer
        for sentence in sentences:
            sent_words = set(sentence.lower().split())
            overlap = len(ans_words.intersection(sent_words))
            if overlap > max_overlap and len(sent_words) > 5:
                max_overlap = overlap
                best_sentence = sentence
                
        # fallback if no overlap found
        if not best_sentence and sentences:
            best_sentence = sentences[0]
            
        # step 2: apply simple wh-word template transformation
        # a very basic heuristic: if it contains a person, use who, else what
        generated_q = f"What is the significance of {best_sentence.lower().replace('.', '')}?"
        
        return generated_q

def prepare_verification_features(df, feature_engineer):
    # drop rows with missing values
    df = df.dropna(subset=['clean_article', 'clean_question', 'clean_a', 'clean_b', 'clean_c', 'clean_d', 'answer'])
    
    corpus = []
    labels = []
    answer_map = {'A': 'clean_a', 'B': 'clean_b', 'C': 'clean_c', 'D': 'clean_d'}
    
    print(f"preparing verification features for {len(df)} rows...")
    
    for _, row in df.iterrows():
        correct_answer = str(row['answer']).strip().upper()
        if correct_answer not in answer_map:
            continue
            
        article_text = row['clean_article']
        question_text = row['clean_question']
        
        for option_char, col_name in answer_map.items():
            option_text = row[col_name]
            combined_text = f"{article_text} {question_text} {option_text}"
            corpus.append(combined_text)
            
            if option_char == correct_answer:
                labels.append(1)
            else:
                labels.append(0)
                
    # transform corpus
    ohe_features, _ = feature_engineer.transform_corpus(corpus)
    return ohe_features, np.array(labels)

def train_and_evaluate_models():
    print("loading vectorizer models...")
    feature_engineer = FeatureEngineer(use_tf_idf=False)
    feature_engineer.load_models()
    
    print("loading clean datasets...")
    train_df = load_data('data/processed/train_clean.csv').head(2000)
    dev_df = load_data('data/processed/dev_clean.csv').head(500)
    
    # --- unsupervised component: k-means clustering ---
    print("running unsupervised k-means clustering on qa pairs...")
    qa_corpus = (train_df['clean_question'].fillna('') + ' ' + train_df['clean_a'].fillna('')).tolist()
    qa_features, _ = feature_engineer.transform_corpus(qa_corpus)
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(qa_features)
    print("k-means clustering complete.")
    
    # --- supervised components: 4 traditional models ---
    x_train, y_train = prepare_verification_features(train_df, feature_engineer)
    
    print("training 4 traditional supervised models (lr, svm, nb, rf)...")
    log_reg = LogisticRegression(max_iter=1000)
    svm_model = SVC(kernel='linear', probability=True, max_iter=1000)
    nb_model = MultinomialNB()
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    
    log_reg.fit(x_train, y_train)
    svm_model.fit(x_train, y_train)
    nb_model.fit(x_train, y_train)
    rf_model.fit(x_train, y_train)
    
    # save models
    os.makedirs('models/model_a/traditional', exist_ok=True)
    with open('models/model_a/traditional/logistic_regression.pkl', 'wb') as f:
        pickle.dump(log_reg, f)
    with open('models/model_a/traditional/svm.pkl', 'wb') as f:
        pickle.dump(svm_model, f)
    with open('models/model_a/traditional/naive_bayes.pkl', 'wb') as f:
        pickle.dump(nb_model, f)
    with open('models/model_a/traditional/random_forest.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    with open('models/model_a/traditional/kmeans.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
        
    # --- generation component: evaluate using nlp generation metrics ---
    print("\nevaluating question generation using bleu, rouge, and meteor...")
    q_gen = QuestionGenerator(svm_model, feature_engineer)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    smooth = SmoothingFunction().method1
    
    bleu_scores = []
    rouge_scores = []
    meteor_scores = []
    
    # evaluate on a small subset of dev to save time
    eval_subset = dev_df.head(100).dropna(subset=['article', 'question', 'answer'])
    
    for _, row in eval_subset.iterrows():
        article = row['article']
        ref_question = str(row['question'])
        
        # map correct answer char to column
        ans_char = str(row['answer']).strip()
        if ans_char in ['A', 'B', 'C', 'D']:
            correct_ans_text = row[ans_char]
        else:
            continue
            
        gen_question = q_gen.generate_question(article, correct_ans_text)
        
        # compute bleu
        ref_tokens = [ref_question.lower().split()]
        gen_tokens = gen_question.lower().split()
        bleu = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=smooth)
        bleu_scores.append(bleu)
        
        # compute rouge
        rouge = scorer.score(ref_question, gen_question)
        rouge_scores.append(rouge['rougeL'].fmeasure)
        
        # compute meteor
        try:
            meteor = single_meteor_score(ref_tokens[0], gen_tokens)
            meteor_scores.append(meteor)
        except Exception:
            pass # fallback if nltk wordnet data is missing
            
    print(f"average bleu score: {np.mean(bleu_scores):.4f}")
    print(f"average rouge-l score: {np.mean(rouge_scores):.4f}")
    if meteor_scores:
        print(f"average meteor score: {np.mean(meteor_scores):.4f}")
        
    print("model a training and evaluation complete.")

if __name__ == '__main__':
    train_and_evaluate_models()
