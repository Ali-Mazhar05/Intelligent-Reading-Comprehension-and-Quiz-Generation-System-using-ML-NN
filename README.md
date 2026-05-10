# Intelligent Reading Comprehension and Quiz Generation System

An AI-powered system designed to automate the generation of reading comprehension quizzes, including Wh-questions, Fill-in-the-Blank (FITB) exercises, and context-aware hints/distractors.

## 🚀 Features
- **Model A**: Automated Question Generation (Wh-Q and FITB) using heuristic-based NLP and answer verification using traditional ML (Logistic Regression, SVM, Naive Bayes).
- **Model B**: Intelligent Distractor and Hint Generation based on cosine similarity and vector embeddings.
- **Interactive UI**: A Streamlit-based dashboard for real-time quiz generation, hint reveal, and performance analytics.
- **Performance Tracking**: Live metrics including BLEU, ROUGE-L, and METEOR scores.

## 🛠️ Setup Instructions

### 1. Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### 2. Clone the Repository
```bash
git clone https://github.com/Ali-Mazhar05/Intelligent-Reading-Comprehension-and-Quiz-Generation-System-using-ML-NN.git
cd Intelligent-Reading-Comprehension-and-Quiz-Generation-System-using-ML-NN
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Data Preparation
Ensure your raw data is placed in `data/raw/`. The system expects `train.csv`, `dev.csv`, and `test.csv`.

## 🏋️ Training Instructions

To train the models and generate the necessary vectorizers/pickled models, run the following scripts in order:

### 1. Preprocessing
Clean and prepare the datasets:
```bash
python src/preprocessing.py
```

### 2. Train Model A
Train the question generation and verification models:
```bash
python src/model_a_train.py
```

### 3. Train Model B
Set up the distractor and hint generators:
```bash
python src/model_b_train.py
```

*Note: Training results and metrics will be saved in the `models/` directory.*

## 🏃 Run Instructions

Once the models are trained, you can launch the interactive web application:

```bash
streamlit run ui/app.py
```

### Navigating the App:
1. **Input View**: Paste a passage and click "Generate Quiz & Hints".
2. **Quiz & Hints View**: Answer the generated question and use the "Hint Panel" if you get stuck.
3. **Analytics Dashboard**: View the technical performance metrics of the underlying models.

## 📊 Evaluation Metrics
The system is evaluated on:
- **BLEU/ROUGE/METEOR**: For linguistic accuracy of generated questions.
- **Distractor Success Rate**: For the validity and uniqueness of generated multiple-choice options.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
