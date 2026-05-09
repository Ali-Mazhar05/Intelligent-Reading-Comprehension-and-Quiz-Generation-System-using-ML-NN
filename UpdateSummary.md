# Project Update Summary & Requirements Checklist

This document details the complete implementation progression for the Intelligent Reading Comprehension and Quiz Generation System. It maps each developed component to the specified project rubric and the updated instructor guidelines.

---

## 1. Data Foundation & Feature Engineering (`src/preprocessing.py`)
- [x] **Data Splitting**: Examination of the source data revealed identical structures across the Kaggle subsets (`train`, `test`, `dev`). The pipeline was modified to load only `train.csv`, employing `sklearn.model_selection.train_test_split` to enforce a strict **80% Train / 10% Dev / 10% Test** empirical split.
- [x] **Text Standardization**: The `clean_text` module was implemented to eliminate punctuation and apply lowercasing uniformly across all articles, questions, and respective options.
- [x] **Feature Vectorization**: A `FeatureEngineer` class was constructed, incorporating `CountVectorizer` (for One-Hot Encoding schemas) and `TfidfVectorizer`. 
- [x] **Computational Resource Optimization**: A cap of `max_features=10000` was enforced within the vectorizers. This constraint prevents memory exhaustion stemming from excessively large sparse matrices, enabling the pipeline to process 100,000 samples on standard consumer hardware.

---

## 2. Model A: Verification & Generation (`src/model_a_train.py`)
- [x] **Four Traditional Supervised Models**: Four distinct algorithms were implemented: `LogisticRegression`, `LinearSVC`, `MultinomialNB`, and `RandomForestClassifier`. Note that `LinearSVC` was utilized in lieu of the standard `SVC` to scale complexity linearly ($O(N)$) instead of quadratically ($O(N^2)$), drastically reducing Random Access Memory (RAM) consumption.
- [x] **Unsupervised Approach**: `MiniBatchKMeans` (a RAM-optimized alternative to standard `KMeans`) was deployed with 5 clusters to partition the Question-Answer pairs based on their One-Hot encoded semantic features.
- [x] **Question Generation Pipeline**: A `QuestionGenerator` class was established. The logic extracts the sentence exhibiting the highest keyword overlap with the correct answer and transforms it utilizing a heuristic Wh-word template.
- [x] **Updated Instructor Evaluation Criteria**: Conventional Accuracy and Precision metrics were entirely removed from the generation evaluation phase. The `nltk` and `rouge-score` libraries were integrated to evaluate the generated questions against the dataset reference text. The system successfully computes **BLEU, ROUGE-L, and METEOR** scores.

---

## 3. Model B: Distractor & Hint Engine (`src/model_b_train.py`)
- [x] **Distractor Extraction**: A `DistractorGenerator` class was developed. The logic parses candidate tokens from the reading passage, calculates **Cosine Similarity** relative to the correct answer using One-Hot Vectors, and outputs the top 3 semantically related but factually incorrect candidates.
- [x] **Hint Generation**: A `HintGenerator` class was finalized. The module tokenizes the passage into sentences, computes semantic overlap against the question via vectorization, and ranks the sentences to present graduated context clues.

---

## 4. UI Integration & Analytics (`ui/app.py`)
- [x] **Streamlit Framework**: The `streamlit` framework was leveraged to construct a 4-screen interactive web application.
- [x] **Screen 1 (Input View)**: Facilitates the input of reading passages, questions, and correct answers. It automatically interfaces with Model B to generate preliminary distractors and hints.
- [x] **Screen 2 (Quiz View)**: Renders the generated quiz alongside the correct answer and 3 generated distractors. Selection logic triggers Model A's Verifier to confirm contextual accuracy.
- [x] **Screen 3 (Hint Panel)**: A collapsible layout designed to display graduated sentence clues derived from Model B.
- [x] **Screen 4 (Analytics Dashboard)**: Renders the live system evaluation metrics (**BLEU, ROUGE-L, and METEOR**), ensuring strict compliance with the updated instructor guidelines.

---

## 5. Coding & Workflow Standards
- [x] **Lexical & Syntactical Style**: All variables and functions rigidly adhere to `snake_case` conventions. Discrete code blocks are preceded by a single-line descriptive comment, and appropriate line spacing is maintained for optimal readability.
- [x] **Version Control Protocols**: The data artifacts (`data/`) and serialized models (`models/`) directories were explicitly appended to `.gitignore`. This prevents server rejection errors caused by excessive payload sizes during git operations. A robust `README.md` was also authored to ensure standardized local setup procedures for all contributors.
