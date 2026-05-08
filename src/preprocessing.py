import pandas as pd
import string
import re
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    # return empty string if text is not valid
    if not isinstance(text, str):
        return ""
    
    # lowercase the input text
    text = text.lower()
    
    # remove punctuation from the text
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # remove extra spaces and strip
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_data(file_path):
    # read csv file into a pandas dataframe
    df = pd.read_csv(file_path)
    
    # drop the unnamed index column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    return df

def preprocess_dataframe(df):
    # apply cleaning function to the article column
    df['clean_article'] = df['article'].apply(clean_text)
    
    # apply cleaning function to the question column
    df['clean_question'] = df['question'].apply(clean_text)
    
    # apply cleaning function to option a
    df['clean_a'] = df['A'].apply(clean_text)
    
    # apply cleaning function to option b
    df['clean_b'] = df['B'].apply(clean_text)
    
    # apply cleaning function to option c
    df['clean_c'] = df['C'].apply(clean_text)
    
    # apply cleaning function to option d
    df['clean_d'] = df['D'].apply(clean_text)
    
    return df

class FeatureEngineer:
    def __init__(self, use_tf_idf=True):
        # flag to enable tf idf vectorization
        self.use_tf_idf = use_tf_idf
        
        # initialize count vectorizer for one hot encoding
        self.count_vectorizer = CountVectorizer(binary=True)
        
        # initialize tf idf vectorizer if enabled
        if self.use_tf_idf:
            self.tf_idf_vectorizer = TfidfVectorizer()
    
    def fit_transform_corpus(self, corpus):
        # fit and transform corpus using count vectorizer
        ohe_features = self.count_vectorizer.fit_transform(corpus)
        
        # default tf idf features to none
        tf_idf_features = None
        
        # fit and transform corpus using tf idf vectorizer if enabled
        if self.use_tf_idf:
            tf_idf_features = self.tf_idf_vectorizer.fit_transform(corpus)
            
        return ohe_features, tf_idf_features
    
    def transform_corpus(self, corpus):
        # transform corpus using fitted count vectorizer
        ohe_features = self.count_vectorizer.transform(corpus)
        
        # default tf idf features to none
        tf_idf_features = None
        
        # transform corpus using fitted tf idf vectorizer if enabled
        if self.use_tf_idf:
            tf_idf_features = self.tf_idf_vectorizer.transform(corpus)
            
        return ohe_features, tf_idf_features
    
    def compute_similarity(self, matrix_a, matrix_b):
        # calculate cosine similarity between two sparse matrices
        return cosine_similarity(matrix_a, matrix_b)

    def save_models(self, save_directory="models/traditional"):
        # create save directory if it does not exist
        os.makedirs(save_directory, exist_ok=True)
        
        # save the count vectorizer model
        with open(os.path.join(save_directory, 'count_vectorizer.pkl'), 'wb') as file:
            pickle.dump(self.count_vectorizer, file)
            
        # save the tf idf vectorizer model if enabled
        if self.use_tf_idf:
            with open(os.path.join(save_directory, 'tf_idf_vectorizer.pkl'), 'wb') as file:
                pickle.dump(self.tf_idf_vectorizer, file)

    def load_models(self, load_directory="models/traditional"):
        # load the count vectorizer model
        with open(os.path.join(load_directory, 'count_vectorizer.pkl'), 'rb') as file:
            self.count_vectorizer = pickle.load(file)
            
        # load the tf idf vectorizer model if enabled
        if self.use_tf_idf:
            with open(os.path.join(load_directory, 'tf_idf_vectorizer.pkl'), 'rb') as file:
                self.tf_idf_vectorizer = pickle.load(file)

def process_and_save_datasets():
    # define file paths for raw data
    train_path = 'data/raw/train.csv'
    dev_path = 'data/raw/dev.csv'
    test_path = 'data/raw/test.csv'
    
    # load datasets into dataframes
    print("loading datasets...")
    train_df = load_data(train_path)
    dev_df = load_data(dev_path)
    test_df = load_data(test_path)
    
    # preprocess dataframes to clean text
    print("preprocessing datasets...")
    train_df = preprocess_dataframe(train_df)
    dev_df = preprocess_dataframe(dev_df)
    test_df = preprocess_dataframe(test_df)
    
    # save preprocessed dataframes to processed directory
    print("saving preprocessed datasets...")
    os.makedirs('data/processed', exist_ok=True)
    train_df.to_csv('data/processed/train_clean.csv', index=False)
    dev_df.to_csv('data/processed/dev_clean.csv', index=False)
    test_df.to_csv('data/processed/test_clean.csv', index=False)
    
    # initialize feature engineer instance
    print("initializing feature engineer...")
    feature_engineer = FeatureEngineer(use_tf_idf=True)
    
    # combine clean articles and questions for vocabulary building
    # taking a subset if memory is an issue, but we'll try the full set
    print("fitting vectorizers on training data...")
    train_corpus = train_df['clean_article'].fillna('') + ' ' + train_df['clean_question'].fillna('')
    feature_engineer.fit_transform_corpus(train_corpus)
    
    # save fitted vectorizer models
    print("saving vectorizer models...")
    feature_engineer.save_models()
    
    print("preprocessing completed successfully.")

if __name__ == '__main__':
    # run the full processing pipeline when script is executed
    process_and_save_datasets()
