# Intelligent Reading Comprehension and Quiz Generation System

This repository contains the codebase for an AI-powered Reading Comprehension and Quiz Generation System. It integrates two distinct machine learning models (Model A for verification and Model B for distractor/hint generation) through a Streamlit user interface.

## 🛠️ Setup Instructions for Collaborators

Since the datasets and trained models are too large to be hosted directly on GitHub, you will need to set them up locally. Follow these steps to get your environment ready:

### 1. Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Download the Dataset
1. Download the RACE dataset from Kaggle (as specified in the project documentation).
2. Create the raw data directory if it doesn't exist: `mkdir -p data/raw`
3. Place the downloaded CSV files inside `data/raw/` so the structure looks like this:
   - `data/raw/train.csv`
   - `data/raw/dev.csv`
   - `data/raw/test.csv`

### 3. Run the ML Pipeline
Run the scripts in the following exact order from the **project root directory** to clean the data and train the models:

**Step 1: Preprocess Data and Train Vectorizers**
```bash
python src/preprocessing.py
```
*(This will generate the clean CSV files in `data/processed/` and save the vectorizer models in `models/traditional/`)*

**Step 2: Train Model A (Verifier)**
```bash
python src/model_a_train.py
```
*(This will train the Logistic Regression and SVM models and save them)*

**Step 3: Setup Model B (Distractors & Hints)**
```bash
python src/model_b_train.py
```
*(This initializes and saves the Hint and Distractor generators)*

### 4. Launch the Web Application
Once all models are trained and saved, you can run the Streamlit user interface:
```bash
streamlit run ui/app.py
```
