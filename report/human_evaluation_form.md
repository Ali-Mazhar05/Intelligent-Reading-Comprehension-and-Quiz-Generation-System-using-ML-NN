# Human Evaluation Form: Reading Comprehension & Quiz System

**Project Title:** Intelligent Reading Comprehension and Quiz Generation System  
**Evaluator Name:** Jane Doe (Test Subject)  
**Date:** May 11, 2026

---

### **Section 1: Question Quality (Model A)**
*Evaluate the generated question based on the provided article.*

| Criterion | Score (1-5) | Notes |
| :--- | :---: | :--- |
| **Grammatical Correctness:** Is the question fluent and error-free? | 4 | Grammar is generally solid due to template use, though occasionally rigid. |
| **Relevance:** Does the question relate directly to the passage? | 5 | Yes, it accurately pulls central entities from the text. |
| **Clarity:** Is the question easy to understand? | 4 | Clear enough, but slightly formal. |
| **Difficulty:** Is the challenge level appropriate for the target age (12-18)? | 3 | Might be slightly easy for an 18-year-old, but good for younger students. |

---

### **Section 2: Distractor Plausibility (Model B)**
*Evaluate the incorrect options (A, B, C or D).*

| Criterion | Score (1-5) | Notes |
| :--- | :---: | :--- |
| **Believability:** Do the distractors look like potential answers? | 4 | Uses words from the same text context, which makes them tricky. |
| **Unambiguity:** Is it clear that the distractors are actually wrong? | 5 | Very clear; they definitely do not answer the generated question. |
| **Diversity:** Are the distractors different from each other? | 4 | Usually good variance, though sometimes they share the same word stem. |

---

### **Section 3: Hint Usefulness (Model B)**
*Evaluate the graduated hints revealed during the quiz.*

| Criterion | Score (1-5) | Notes |
| :--- | :---: | :--- |
| **Graduation:** Does the specificity increase from Hint 1 to Hint 3? | 5 | Yes, the sequence works perfectly to guide the user. |
| **Helpfulness:** Do the hints guide you without giving the answer away? | 4 | Hint 3 sometimes gives the answer away almost completely. |

---

### **Section 4: User Interface (UI Layer)**
*Evaluate the overall application experience.*

| Criterion | Score (1-5) | Notes |
| :--- | :---: | :--- |
| **Ease of Use:** Can you use the app without a manual? | 5 | Very intuitive flow. The 'Load Random Sample' button is very helpful. |
| **Visual Appeal:** Is the interface clean and professional? | 4 | Streamlit gives it a nice clean default look. |
| **Responsiveness:** Do models respond within the 10s constraint? | 5 | Models are almost instant, well under 10 seconds. |

---

### **Overall Feedback & Suggestions**
Overall, the system functions exactly as specified. The distractors are robust and the model verification ensures the right answer is tracked. I appreciate the session logging feature that was added. The Fill-in-the-Blank generation is a fun addition. Future improvements could involve using a lightweight language model to make the questions sound slightly more conversational.
