import pandas as pd
import numpy as np
import os
import pickle
import re
import string
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.semi_supervised import LabelPropagation
from sklearn.calibration import CalibratedClassifierCV
import nltk
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
from preprocessing import load_data, FeatureEngineer
from model_b_train import DistractorGenerator, HintGenerator

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
            
        candidates = []
        
        # Generation Step: Extract potential entities/answers and create question candidates
        if not correct_answer:
            # find multiple capitalized words as candidates
            potential_entities = []
            for s in valid_sentences:
                mid_sentence_entities = re.findall(r' [A-Z][a-z]+\b', s)
                for ent in mid_sentence_entities:
                    clean_ent = ent.strip().strip(string.punctuation)
                    if clean_ent.lower() not in blacklist and len(clean_ent) > 3:
                        potential_entities.append((clean_ent, s))
            
            # if few entities, add some long nouns
            if len(potential_entities) < 3:
                for s in valid_sentences:
                    words = s.split()
                    for w in words:
                        cw = w.strip(string.punctuation)
                        if len(cw) > 6 and cw.lower() not in blacklist:
                            potential_entities.append((cw, s))
            
            # Create candidates
            for ent, sent in potential_entities[:10]: # limit to top 10 for ranking
                candidates.append(self._create_question(ent, sent))
        else:
            # if correct_answer is fixed, find the best sentence match
            ans_words = set(str(correct_answer).lower().split())
            best_sent = valid_sentences[0]
            max_overlap = -1
            for sentence in valid_sentences:
                sent_words = set(sentence.lower().split())
                overlap = len(ans_words.intersection(sent_words))
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_sent = sentence
            candidates.append(self._create_question(correct_answer, best_sent))
            
        if not candidates:
            return "What is the main idea of this passage?", "Information"

        # Ranking Step: Use the ML Ranker (SVM/Ensemble) to score candidates
        if len(candidates) > 1 and self.ranker_model:
            ranked_candidates = []
            for q, a in candidates:
                # use verifier as a proxy for ranking: how confident is it that 'a' is the answer to 'q'?
                combined = f"{article} {q} {a}"
                feats, _ = self.feature_engineer.transform_corpus([combined])
                
                # if model has predict_proba (like Voting or calibrated SVM)
                if hasattr(self.ranker_model, "predict_proba"):
                    score = self.ranker_model.predict_proba(feats)[0][1]
                else:
                    # fallback for LinearSVC: use decision function
                    try:
                        score = self.ranker_model.decision_function(feats)[0]
                    except:
                        score = 0
                
                ranked_candidates.append((score, q, a))
            
            # sort by score descending
            ranked_candidates.sort(key=lambda x: x[0], reverse=True)
            _, best_q, best_a = ranked_candidates[0]
            return best_q, best_a
            
        return candidates[0]

    def _create_question(self, clean_answer, target_sentence):
        clean_answer = str(clean_answer).strip(string.punctuation)
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


class FITBGenerator:
    def __init__(self, feature_engineer):
        self.feature_engineer = feature_engineer
        
    def generate_fitb_question(self, article):
        import re
        import string
        import numpy as np
        
        # split into sentences
        sentences = re.split(r'(?<=[.!?]) +', str(article))
        valid_sentences = [s for s in sentences if len(s.split()) > 8 and len(s.split()) < 25]
        
        if not valid_sentences:
            valid_sentences = sentences
            
        # words to ignore
        blacklist = {
            'there', 'their', 'these', 'those', 'where', 'which', 'whose', 'while',
            'this', 'that', 'they', 'them', 'from', 'with', 'under', 'after',
            'before', 'into', 'each', 'every', 'some', 'many', 'much', 'very',
            'when', 'what', 'then', 'than', 'just', 'more', 'also', 'only'
        }
        
        # pick a sentence from the middle-ish to avoid intros/outros
        target_idx = len(valid_sentences) // 2
        target_sentence = valid_sentences[target_idx]
        
        # tokenize and find potential blanks using OHE significance
        words = target_sentence.split()
        potential_blanks = []
        
        # get vocabulary to check OHE significance
        vocab = self.feature_engineer.count_vectorizer.vocabulary_
        
        for i, word in enumerate(words):
            clean_word = word.strip(string.punctuation)
            if len(clean_word) > 4 and clean_word.lower() not in blacklist:
                # check if word is in our OHE vocabulary (The Intelligence Gate)
                if clean_word.lower() in vocab:
                    potential_blanks.append((clean_word, i))
        
        if not potential_blanks:
            # fallback to any long word
            for i, word in enumerate(words):
                clean_word = word.strip(string.punctuation)
                if len(clean_word) > 3:
                    potential_blanks.append((clean_word, i))
        
        # pick the word that is most 'central' to the sentence (heuristic for subject/object)
        # we pick the one closest to the middle of the sentence
        mid_p = len(words) // 2
        potential_blanks.sort(key=lambda x: abs(x[1] - mid_p))
        
        target_word, target_pos = potential_blanks[0]
        
        # create masked sentence
        masked_words = list(words)
        masked_words[target_pos] = "__________"
        masked_sentence = " ".join(masked_words)
        
        return masked_sentence, target_word

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
    
    # --- unsupervised component: gaussian mixture model (GMM) ---
    print("running unsupervised gmm on qa pairs...")
    # gmm is memory intensive on sparse data, use a dense subset
    gmm_subset_size = min(5000, qa_features.shape[0])
    gmm_indices = np.random.choice(qa_features.shape[0], gmm_subset_size, replace=False)
    x_gmm = qa_features[gmm_indices].toarray()
    
    gmm = GaussianMixture(n_components=5, random_state=42)
    gmm.fit(x_gmm)
    print("gmm clustering complete.")
    
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
    with open('models/model_a/traditional/gmm.pkl', 'wb') as f:
        pickle.dump(gmm, f)

    # --- semi-supervised component: label propagation ---
    print("running semi-supervised label propagation on subset...")
    # label propagation is memory intensive, using a smaller subset (e.g. 2000 samples)
    lp_subset_size = min(2000, x_train.shape[0])
    x_lp_subset = x_train[:lp_subset_size].toarray() # convert to dense for label propagation
    y_lp_subset = y_train[:lp_subset_size].copy()
    
    # simulate unlabelled data by setting 30% of labels to -1
    rng = np.random.RandomState(42)
    random_unlabeled_points = rng.rand(len(y_lp_subset)) < 0.3
    y_lp_subset[random_unlabeled_points] = -1
    
    lp_model = LabelPropagation(kernel='knn', n_neighbors=7)
    lp_model.fit(x_lp_subset, y_lp_subset)
    print("label propagation complete.")
    
    with open('models/model_a/traditional/label_propagation.pkl', 'wb') as f:
        pickle.dump(lp_model, f)

    # --- ensemble strategy: soft voting classifier ---
    print("creating ensemble voting classifier...")
    # linearSVC doesn't have predict_proba, so we wrap it in calibration
    calibrated_svm = CalibratedClassifierCV(LinearSVC(max_iter=1000, dual="auto"), cv=3)
    
    ensemble_model = VotingClassifier(
        estimators=[
            ('lr', log_reg),
            ('svm', calibrated_svm),
            ('nb', nb_model),
            ('rf', rf_model)
        ],
        voting='soft'
    )
    # fit on a subset for speed
    ensemble_subset_x, ensemble_subset_y = x_train[:15000], y_train[:15000]
    ensemble_model.fit(ensemble_subset_x, ensemble_subset_y)
    print("ensemble training complete.")
    
    # --- real metric calculation for verification ---
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    x_dev, y_dev = prepare_verification_features(dev_df, feature_engineer)
    
    print("evaluating ensemble on dev set...")
    y_pred = ensemble_model.predict(x_dev)
    ens_acc = accuracy_score(y_dev, y_pred)
    ens_f1 = f1_score(y_dev, y_pred, average='macro')
    cm = confusion_matrix(y_dev, y_pred)
    
    print(f"ensemble accuracy: {ens_acc:.4f}")
    print(f"ensemble macro f1: {ens_f1:.4f}")
    
    lp_f1 = 0
    if 'lp_model' in locals():
        # evaluate label propagation on its own subset (dense)
        y_lp_pred = lp_model.predict(x_lp_subset)
        lp_f1 = f1_score(y_lp_subset[y_lp_subset != -1], y_lp_pred[y_lp_subset != -1], average='macro')
        print(f"label propagation f1: {lp_f1:.4f}")

    with open('models/model_a/traditional/ensemble_model.pkl', 'wb') as f:
        pickle.dump(ensemble_model, f)
        
    # --- generation component: evaluate using nlp generation metrics ---
    print("\nevaluating question generation using bleu, rouge, and meteor...")
    # use the ensemble as the ranker for question generation
    q_gen = QuestionGenerator(ensemble_model, feature_engineer)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    smooth = SmoothingFunction().method1
    
    bleu_scores = []
    rouge_scores = []
    meteor_scores = []
    em_scores = []
    
    # --- model evaluation ---
    eval_subset = dev_df.dropna(subset=['article', 'question', 'answer', 'A', 'B', 'C', 'D'])
    
    # Load the trained distractor generator with ranker from disk if available
    dist_gen_path = 'models/model_b/traditional/distractor_generator.pkl'
    if os.path.exists(dist_gen_path):
        # ensure DistractorGenerator is in __main__ for the pickle loader
        import sys
        import __main__
        __main__.DistractorGenerator = DistractorGenerator
        
        with open(dist_gen_path, 'rb') as f:
            dist_gen = pickle.load(f)
    else:
        dist_gen = DistractorGenerator(feature_engineer)
        
    dist_success_count = 0
    dist_precisions = []
    dist_recalls = []
    
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
            
        # 1. evaluate model a (Question Generation)
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

        # compute exact match (EM) - strict character-level match to gold answer
        em = 1 if str(gen_ans).lower().strip() == str(correct_ans_text).lower().strip() else 0
        em_scores.append(em)
            
        # 2. evaluate model b (Distractor Generation)
        # Gold distractors from dataset
        gold_distractors = {str(row[c]).lower().strip() for c in ['A', 'B', 'C', 'D'] if c != ans_char}
        
        # Predicted distractors from model
        pred_distractors = dist_gen.generate_distractors(article, gen_ans)
        pred_distractors_set = {str(d).lower().strip() for d in pred_distractors}
        
        # Success Rate (Metric for robustness)
        if len(set(pred_distractors)) == 3 and gen_ans.lower() not in [d.lower() for d in pred_distractors]:
            dist_success_count += 1
            
        # Precision/Recall (Metric for alignment with human distractors)
        intersection = gold_distractors.intersection(pred_distractors_set)
        precision = len(intersection) / len(pred_distractors_set) if pred_distractors_set else 0
        recall = len(intersection) / len(gold_distractors) if gold_distractors else 0
        
        dist_precisions.append(precision)
        dist_recalls.append(recall)
            
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
    avg_rouge = np.mean(rouge_scores) if rouge_scores else 0
    avg_meteor = np.mean(meteor_scores) if meteor_scores else 0
    avg_em = np.mean(em_scores) if em_scores else 0
    dist_success_rate = (dist_success_count / len(eval_subset)) * 100 if len(eval_subset) > 0 else 0
    
    avg_dist_precision = np.mean(dist_precisions) if dist_precisions else 0
    avg_dist_recall = np.mean(dist_recalls) if dist_recalls else 0
    avg_dist_f1 = 2 * (avg_dist_precision * avg_dist_recall) / (avg_dist_precision + avg_dist_recall) if (avg_dist_precision + avg_dist_recall) > 0 else 0
    
    print(f"average bleu score: {avg_bleu:.4f}")
    print(f"average rouge-l score: {avg_rouge:.4f}")
    print(f"average meteor score: {avg_meteor:.4f}")
    print(f"average exact match: {avg_em:.4f}")
    print(f"distractor extraction success: {dist_success_rate:.2f}%")
    print(f"distractor precision: {avg_dist_precision:.4f}")
    print(f"distractor recall: {avg_dist_recall:.4f}")
    print(f"distractor f1: {avg_dist_f1:.4f}")
    
    # save metrics to json for the ui dashboard
    import json
    metrics = {
        "bleu": round(float(avg_bleu), 4),
        "rouge": round(float(avg_rouge), 4),
        "meteor": round(float(avg_meteor), 4),
        "exact_match": round(float(avg_em), 4),
        "distractor_success": round(dist_success_rate, 2),
        "distractor_precision": round(float(avg_dist_precision), 4),
        "distractor_recall": round(float(avg_dist_recall), 4),
        "distractor_f1": round(float(avg_dist_f1), 4),
        "ensemble_accuracy": round(float(ens_acc), 4),
        "ensemble_f1": round(float(ens_f1), 4),
        "semi_supervised_f1": round(float(lp_f1), 4),
        "confusion_matrix": cm.tolist()
    }
    os.makedirs('models', exist_ok=True)
    with open('models/performance_metrics.json', 'w') as f:
        json.dump(metrics, f)
        
    print("training and evaluation complete. metrics saved.")

if __name__ == '__main__':
    train_and_evaluate_models()
