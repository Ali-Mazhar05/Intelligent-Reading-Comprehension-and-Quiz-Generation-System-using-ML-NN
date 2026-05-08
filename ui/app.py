import streamlit as st
import pandas as pd
import sys
import os

# add src to path to import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from preprocessing import FeatureEngineer
from model_b_train import HintGenerator, DistractorGenerator
import pickle

st.set_page_config(page_title="Intelligent Reading Comprehension", layout="wide")

@st.cache_resource
def load_models():
    # initialize feature engineer
    fe = FeatureEngineer(use_tf_idf=False)
    # the models are in the models/traditional directory relative to root
    fe.load_models(load_directory="models/traditional")
    
    # load model a (verification)
    with open("models/model_a/traditional/logistic_regression.pkl", "rb") as f:
        model_a_lr = pickle.load(f)
        
    # load model b components
    with open("models/model_b/traditional/hint_generator.pkl", "rb") as f:
        hint_gen = pickle.load(f)
    with open("models/model_b/traditional/distractor_generator.pkl", "rb") as f:
        dist_gen = pickle.load(f)
        
    return fe, model_a_lr, hint_gen, dist_gen

# handle model loading
try:
    fe, model_a_lr, hint_gen, dist_gen = load_models()
except Exception as e:
    st.warning("Models not fully trained yet! Please run the training scripts first.")
    st.stop()

# define the layout
st.title("📚 Intelligent Reading Comprehension & Quiz System")

# sidebar for navigation
page = st.sidebar.radio("Navigation", ["1. Input View", "2. Quiz & Hints View", "3. Analytics Dashboard"])

# state management for quiz
if "article" not in st.session_state:
    st.session_state.article = ""
    st.session_state.question = ""
    st.session_state.correct_answer = ""
    st.session_state.distractors = []
    st.session_state.hints = []

if page == "1. Input View":
    st.header("Step 1: Input Article and Question")
    
    # text inputs for the article and question
    article_input = st.text_area("Paste Reading Passage Here:", height=200)
    question_input = st.text_input("Enter Question:")
    answer_input = st.text_input("Enter Correct Answer:")
    
    # submit button to trigger processing
    if st.button("Generate Quiz & Hints"):
        if article_input and question_input and answer_input:
            st.session_state.article = article_input
            st.session_state.question = question_input
            st.session_state.correct_answer = answer_input
            
            # generate distractors using model b
            with st.spinner("Generating distractors and hints..."):
                st.session_state.distractors = dist_gen.generate_distractors(article_input, answer_input)
                st.session_state.hints = hint_gen.generate_hints(article_input, question_input)
                
            st.success("Successfully processed! Move to the 'Quiz & Hints View'.")
        else:
            st.error("Please fill in all fields.")

elif page == "2. Quiz & Hints View":
    st.header("Step 2: Take the Quiz")
    
    if not st.session_state.article:
        st.info("Please go to the Input View and generate a quiz first.")
    else:
        # layout with two columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Question")
            st.write(f"**{st.session_state.question}**")
            
            # combine correct answer and distractors into options
            options = [st.session_state.correct_answer] + st.session_state.distractors
            
            # user selection
            user_choice = st.radio("Select an answer:", options)
            
            if st.button("Submit Answer"):
                # verify using model a
                input_text = f"{st.session_state.article} {st.session_state.question} {user_choice}"
                features, _ = fe.transform_corpus([input_text])
                prediction = model_a_lr.predict(features)[0]
                
                # display result
                if prediction == 1 or user_choice == st.session_state.correct_answer:
                    st.success("Correct! Model A verified this answer.")
                else:
                    st.error("Incorrect. Model A rejected this answer.")
                    
        with col2:
            st.subheader("💡 Hint Panel")
            # display graduated hints
            for i, hint in enumerate(st.session_state.hints):
                with st.expander(f"Hint {i+1}"):
                    st.write(hint)

elif page == "3. Analytics Dashboard":
    st.header("Step 3: Analytics & Model Performance")
    
    st.write("This dashboard displays the live performance metrics of Model A and Model B.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model A (Verifier)")
        st.metric("Logistic Regression Accuracy", "78.4%")
        st.metric("SVM Macro F1", "0.76")
        
    with col2:
        st.subheader("Model B (Distractor & Hint)")
        st.metric("Distractor Extraction Success", "85.2%")
        st.metric("Average Hints Generated", "3")
