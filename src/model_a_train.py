import pandas as pd
import numpy as np
import os
import pickle
import re
import string
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import MiniBatchKMeans
import nltk
from tqdm import tqdm
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
        
    def generate_question(self, article, correct_answer=None):
        sentences = re.split(r'(?<=[.!?]) +', str(article))
        
        # words that should never be selected as the 'correct answer'
        blacklist = {
            'there', 'their', 'these', 'those', 'where', 'which', 'whose', 'while',
            'this', 'that', 'they', 'them', 'from', 'with', 'under', 'after',
            'before', 'into', 'each', 'every', 'some', 'many', 'much', 'very',
            'when', 'what', 'then', 'than', 'just', 'more', 'also', 'only'
        }
        
        # filter for sentences with enough words to be meaningful
        valid_sentences = [s for s in sentences if len(s.split()) > 6]
        if not valid_sentences:
            valid_sentences = sentences
        
        # if no correct answer, select one using improved heuristics
        if not correct_answer:
            # try to find a capitalized word that is NOT at the start of a sentence
            best_entities = []
            for s in valid_sentences:
                # find capitalized words that are preceded by a space (not at start of string)
                mid_sentence_entities = re.findall(r' [A-Z][a-z]+\b', s)
                for ent in mid_sentence_entities:
                    clean_ent = ent.strip().strip(string.punctuation)
                    if clean_ent.lower() not in blacklist and len(clean_ent) > 3:
                        best_entities.append(clean_ent)
            
            if best_entities:
                correct_answer = best_entities[0]
                # find the specific sentence this entity came from
                for s in valid_sentences:
                    if correct_answer in s:
                        target_sentence = s
                        break
            else:
                # fallback: just pick a long noun-like word from the first valid sentence
                target_sentence = valid_sentences[0]
                words = target_sentence.split()
                long_words = [w.strip(string.punctuation) for w in words if len(w) > 5 and w.lower() not in blacklist]
                correct_answer = long_words[0] if long_words else words[0]
        else:
            # find the sentence that best matches the given correct answer
            ans_words = set(str(correct_answer).lower().split())
            target_sentence = valid_sentences[0]
            max_overlap = -1
            for sentence in valid_sentences:
                sent_words = set(sentence.lower().split())
                overlap = len(ans_words.intersection(sent_words))
                if overlap > max_overlap:
                    max_overlap = overlap
                    target_sentence = sentence
        
        # final cleaning
        clean_answer = str(correct_answer).strip(string.punctuation)
        
        # determine wh-word based on what type of entity the answer is
        if re.match(r'^\d{4}$', clean_answer):
            wh_phrase = f"When did the events regarding \"{clean_answer}\" occur?"
        elif re.search(r'\b(city|town|country|school|university|street|road|park)\b', target_sentence.lower()):
            wh_phrase = f"Where in the passage is \"{clean_answer}\" located?"
        elif re.match(r'^[A-Z][a-z]+', clean_answer):
            wh_phrase = f"Who is \"{clean_answer}\" as described in the passage?"
        else:
            wh_phrase = f"What is mentioned in the passage about \"{clean_answer}\"?"
        
        return wh_phrase, clean_answer

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
    train_df = load_data('data/processed/train_clean.csv')
    dev_df = load_data('data/processed/dev_clean.csv')
    
    # --- unsupervised component: k-means clustering ---
    print("running unsupervised k-means clustering on qa pairs...")
    qa_corpus = (train_df['clean_question'].fillna('') + ' ' + train_df['clean_a'].fillna('')).tolist()
    qa_features, _ = feature_engineer.transform_corpus(qa_corpus)
    
    # minibatchkmeans uses far less memory than standard kmeans
    kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, n_init=3, batch_size=2048)
    kmeans.fit(qa_features)
    print("k-means clustering complete.")
    
    # --- supervised components: 4 traditional models ---
    x_train, y_train = prepare_verification_features(train_df, feature_engineer)
    
    print("training 4 traditional supervised models...")
    log_reg = LogisticRegression(max_iter=1000, n_jobs=-1)
    svm_model = LinearSVC(max_iter=1000, dual="auto")
    nb_model = MultinomialNB()
    rf_model = RandomForestClassifier(n_estimators=20, max_depth=10, n_jobs=-1, random_state=42)
    
    print("fitting logistic regression...")
    log_reg.fit(x_train, y_train)
    
    print("fitting linear svc...")
    svm_model.fit(x_train, y_train)
    
    print("fitting naive bayes...")
    nb_model.fit(x_train, y_train)
    
    print("fitting random forest...")
    rf_model.fit(x_train, y_train)
    
    print("saving models...")
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
    
    # --- model evaluation ---
    eval_subset = dev_df.dropna(subset=['article', 'question', 'answer'])
    from model_b_train import DistractorGenerator
    dist_gen = DistractorGenerator(feature_engineer)
    dist_success_count = 0
    
    print(f"starting generation evaluation on {len(eval_subset)} samples...")
    for _, row in tqdm(eval_subset.iterrows(), total=len(eval_subset), desc="Evaluating Models"):
        article = row['article']
        ref_question = str(row['question'])
        
        # map correct answer char to column
        ans_char = str(row['answer']).strip()
        if ans_char in ['A', 'B', 'C', 'D']:
            correct_ans_text = row[ans_char]
        else:
            continue
            
        # evaluate model a
        gen_question, gen_ans = q_gen.generate_question(article, correct_ans_text)
        
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
            pass
            
        # evaluate model b
        distractors = dist_gen.generate_distractors(article, gen_ans)
        # success if we got 3 unique distractors and none are the correct answer
        if len(set(distractors)) == 3 and gen_ans.lower() not in [d.lower() for d in distractors]:
            dist_success_count += 1
            
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
    avg_rouge = np.mean(rouge_scores) if rouge_scores else 0
    avg_meteor = np.mean(meteor_scores) if meteor_scores else 0
    dist_success_rate = (dist_success_count / len(eval_subset)) * 100 if len(eval_subset) > 0 else 0
    
    print(f"average bleu score: {avg_bleu:.4f}")
    print(f"average rouge-l score: {avg_rouge:.4f}")
    print(f"average meteor score: {avg_meteor:.4f}")
    print(f"distractor extraction success: {dist_success_rate:.2f}%")
    
    # save metrics to json for the ui dashboard
    import json
    metrics = {
        "bleu": round(float(avg_bleu), 4),
        "rouge": round(float(avg_rouge), 4),
        "meteor": round(float(avg_meteor), 4),
        "distractor_success": round(dist_success_rate, 2)
    }
    os.makedirs('models', exist_ok=True)
    with open('models/performance_metrics.json', 'w') as f:
        json.dump(metrics, f)
        
    print("training and evaluation complete. metrics saved.")

if __name__ == '__main__':
    train_and_evaluate_models()
