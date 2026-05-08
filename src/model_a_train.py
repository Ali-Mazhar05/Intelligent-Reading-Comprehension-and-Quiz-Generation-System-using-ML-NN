import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from preprocessing import load_data, FeatureEngineer

def prepare_verification_features(df, feature_engineer):
    # we need to create a dataset where we predict if an option is correct
    # we will concatenate article, question, and the option text
    # label will be 1 if option is correct, 0 otherwise
    
    # drop rows with missing values
    df = df.dropna(subset=['clean_article', 'clean_question', 'clean_a', 'clean_b', 'clean_c', 'clean_d', 'answer'])
    
    # lists to store features and labels
    corpus = []
    labels = []
    
    # mapping for answer characters to column names
    answer_map = {'A': 'clean_a', 'B': 'clean_b', 'C': 'clean_c', 'D': 'clean_d'}
    
    print(f"preparing verification features for {len(df)} rows...")
    
    # iterate through dataframe to build training pairs
    for _, row in df.iterrows():
        correct_answer = str(row['answer']).strip().upper()
        
        # skip if answer is invalid
        if correct_answer not in answer_map:
            continue
            
        article_text = row['clean_article']
        question_text = row['clean_question']
        
        # create one positive and three negative samples per question
        for option_char, col_name in answer_map.items():
            option_text = row[col_name]
            
            # combine text for context
            combined_text = f"{article_text} {question_text} {option_text}"
            corpus.append(combined_text)
            
            # assign label 1 if this option is the correct answer
            if option_char == correct_answer:
                labels.append(1)
            else:
                labels.append(0)
                
    print("transforming text to one hot encoded features...")
    # transform the entire corpus using the pre-fitted vectorizer
    ohe_features, _ = feature_engineer.transform_corpus(corpus)
    
    return ohe_features, np.array(labels)

def train_and_evaluate_models():
    # initialize feature engineer and load fitted vectorizers
    print("loading vectorizer models...")
    feature_engineer = FeatureEngineer(use_tf_idf=False)
    feature_engineer.load_models()
    
    # load clean datasets
    print("loading clean datasets...")
    train_df = load_data('data/processed/train_clean.csv')
    dev_df = load_data('data/processed/dev_clean.csv')
    
    # we'll use a subset of training data for faster local training
    # using 5000 rows for rapid prototyping on local machine
    train_subset = train_df.head(5000)
    dev_subset = dev_df.head(1000)
    
    # extract features and labels
    x_train, y_train = prepare_verification_features(train_subset, feature_engineer)
    x_dev, y_dev = prepare_verification_features(dev_subset, feature_engineer)
    
    # initialize traditional machine learning models
    log_reg = LogisticRegression(max_iter=1000)
    svm_model = SVC(kernel='linear', probability=True, max_iter=2000)
    
    # train logistic regression model
    print("training logistic regression model...")
    log_reg.fit(x_train, y_train)
    
    # evaluate logistic regression model
    lr_preds = log_reg.predict(x_dev)
    lr_acc = accuracy_score(y_dev, lr_preds)
    lr_f1 = f1_score(y_dev, lr_preds, average='macro')
    print(f"logistic regression - accuracy: {lr_acc:.4f}, macro f1: {lr_f1:.4f}")
    
    # train svm model
    print("training svm model...")
    svm_model.fit(x_train, y_train)
    
    # evaluate svm model
    svm_preds = svm_model.predict(x_dev)
    svm_acc = accuracy_score(y_dev, svm_preds)
    svm_f1 = f1_score(y_dev, svm_preds, average='macro')
    print(f"svm - accuracy: {svm_acc:.4f}, macro f1: {svm_f1:.4f}")
    
    # save trained models
    print("saving trained models...")
    os.makedirs('models/model_a/traditional', exist_ok=True)
    with open('models/model_a/traditional/logistic_regression.pkl', 'wb') as file:
        pickle.dump(log_reg, file)
    with open('models/model_a/traditional/svm.pkl', 'wb') as file:
        pickle.dump(svm_model, file)
        
    print("model a training complete.")

if __name__ == '__main__':
    train_and_evaluate_models()
