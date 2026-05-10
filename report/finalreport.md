# Final Report: Intelligent Reading Comprehension and Quiz Generation System

## 1. Abstract
This report presents the design and implementation of an Intelligent Reading Comprehension and Quiz Generation System. The system utilizes a hybrid approach, combining traditional machine learning classifiers with heuristic-based natural language processing to automate the creation of educational assessments. The architecture is divided into two primary components: **Model A**, which handles question generation (Wh-questions and Fill-in-the-Blank) and answer verification using models like Logistic Regression and SVM; and **Model B**, which generates contextually relevant distractors and hints using vector-based similarity measures. Evaluation results indicate a high success rate in distractor generation (99.99%) and baseline performance in linguistic generation metrics (BLEU: 0.019, ROUGE: 0.1454, METEOR: 0.11), reflecting the system's focus on structural accuracy and educational utility over purely generative fluency.

## 2. Introduction & Motivation
Reading comprehension is a fundamental pillar of education, yet the manual creation of high-quality quizzes is time-consuming for educators. The motivation behind this project is to leverage Machine Learning and Natural Language Processing (NLP) to automate this process. By providing a system that can take any arbitrary text and generate meaningful multiple-choice questions, fill-in-the-blank exercises, and guided hints, we can enhance the learning experience and provide immediate feedback to students. The system aims to bridge the gap between raw text processing and interactive educational tools.

## 3. Related Work
The field of Automated Question Generation (AQG) has evolved significantly over the last decade. Key works influencing this project include:
1. **Rajpurkar et al. (2016)**: Introduced the SQuAD dataset, which revolutionized machine reading comprehension by providing a large-scale benchmark for Q&A pairs.
2. **Du et al. (2017)**: Explored neural question generation, demonstrating the effectiveness of seq2seq models in generating human-like questions.
3. **Vaswani et al. (2017)**: The "Attention Is All You Need" paper, which introduced the Transformer architecture, forming the backbone of modern NLP models that this system seeks to emulate or interface with.
4. **Devlin et al. (2018)**: BERT (Bidirectional Encoder Representations from Transformers) set new standards for understanding context, which informs our heuristic approach to entity extraction.
5. **Pan et al. (2019)**: Provided a comprehensive review of AQG techniques, highlighting the transition from rule-based systems to deep learning approaches.

## 4. Dataset Analysis
The system was trained and evaluated using the SQuAD-style dataset structure, consisting of over 100,000 samples. 
- **Structure**: The dataset includes an `article` (context), a `question`, and the `answer` along with distractors.
- **Preprocessing**: Data cleaning involved lowercase conversion, punctuation removal, and tokenization. A `FeatureEngineer` component was used to transform raw text into One-Hot Encoded (OHE) vectors, facilitating traditional machine learning analysis.
- **Splits**: The data was split into training (`train_clean.csv`) and development (`dev_clean.csv`) sets to ensure robust evaluation.

## 5. Model A: Design, Training, Results
### Design
Model A serves as the **Generator and Verifier**. It consists of:
- **QuestionGenerator**: Uses heuristics to identify significant entities in the text and generates Wh-questions (Who, What, Where, When) based on entity type.
- **FITBGenerator**: Identifies "significant" words using OHE vocabulary importance and masks them to create Fill-in-the-Blank questions.
- **Verification Engine**: Employs a suite of traditional ML models (Logistic Regression, Linear SVC, Naive Bayes, and Random Forest) trained to verify if an answer matches a given question-article pair.

### Training
Models were trained on OHE features. Logistic Regression and SVM were found to be the most stable for binary classification (Correct vs. Incorrect answer). Unsupervised K-Means clustering was also implemented to group QA pairs into logical clusters for further analysis.

### Results
- **BLEU Score**: 0.019
- **ROUGE-L Score**: 0.1454
- **METEOR Score**: 0.11
The scores reflect a rule-based generation approach which, while accurate in content, lacks the linguistic variability of large-scale transformer models.

## 6. Model B: Design, Training, Results
### Design
Model B acts as the **Educational Support Engine**, handling:
- **DistractorGenerator**: Uses cosine similarity between the correct answer and other words in the article. It selects words that are semantically related but contextually distinct to create challenging multiple-choice options.
- **HintGenerator**: Identifies sentences in the article most similar to the question using vector representations, providing "scaffolding" for the student.

### Training
Model B relies on vector representations generated by the `FeatureEngineer`. It is saved as a set of pickled generators that interact dynamically with the UI.

### Results
- **Distractor Success Rate**: 99.99%
The system is exceptionally successful at extracting unique, non-overlapping distractors from the provided source text, ensuring that every quiz generated is valid and challenging.

## 7. User Interface Description
The UI is built using **Streamlit**, providing a clean, responsive web interface:
1. **Input View**: Users paste a passage and select a quiz mode (MCQ or FITB). The system processes the text in real-time.
2. **Quiz & Hints View**: Presents the generated question. It features a sequential hint system where students can reveal up to 3 hints before seeing the final answer.
3. **Analytics Dashboard**: Displays live performance metrics (BLEU, ROUGE, Distractor Success) to provide transparency into model performance.

## 8. Evaluation & Discussion
The evaluation shows a clear trade-off. By using traditional ML and heuristics (Model A), we achieve high **reliability and speed** but lower **linguistic diversity** (as seen in BLEU/ROUGE). However, for educational purposes, the accuracy of the "Correct Answer" and the quality of the "Distractors" (Model B) are often more critical than the phrasing of the question. The 99.99% distractor success rate indicates that the system is highly effective at creating valid multiple-choice structures.

## 9. Limitations & Future Work
### Limitations
- **Linguistic Fluency**: The questions are generated using templates, which can feel repetitive.
- **Dependency on Vocabulary**: The OHE approach is limited to the vocabulary seen during training.

### Future Work
- **Transformer Integration**: Incorporating T5 or GPT-based fine-tuning for more natural question phrasing.
- **Semantic Distractors**: Moving beyond keyword similarity to deeper semantic relationships for distractors.
- **User Personalization**: Tracking user performance to adjust question difficulty.

## 10. Conclusion
The Intelligent Reading Comprehension and Quiz Generation System successfully demonstrates how hybrid ML approaches can automate educational tasks. By combining the verification power of traditional classifiers with heuristic generation logic, the system provides a robust tool for educators and students alike. The high success rate in distractor generation and the integrated hint system make it a practical solution for real-world learning environments.

## 11. References
1. Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text. *EMNLP*.
2. Du, X., Shao, J., & Cardie, C. (2017). Learning to Ask: Neural Question Generation for Reading Comprehension. *ACL*.
3. Vaswani, A., et al. (2017). Attention Is All You Need. *NIPS*.
4. Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL*.
5. Pan, L., et al. (2019). A Review of Automatic Question Generation from Text: Techniques and Applications. *arXiv*.
