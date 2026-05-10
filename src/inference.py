import sys
import os
import pickle

class InferenceAPI:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.fe = None
        self.model_a_ensemble = None
        self.hint_gen = None
        self.dist_gen = None
        self.q_gen = None
        self.fitb_gen = None
        self._load_models()

    def _load_models(self):
        from preprocessing import FeatureEngineer
        from model_b_train import HintGenerator, DistractorGenerator
        from model_a_train import QuestionGenerator, FITBGenerator

        self.fe = FeatureEngineer(use_tf_idf=False)
        self.fe.load_models(load_directory=os.path.join(self.models_dir, 'traditional'))

        # load model a (verification)
        with open(os.path.join(self.models_dir, 'model_a/traditional/ensemble_model.pkl'), 'rb') as f:
            self.model_a_ensemble = pickle.load(f)

        # load model b components
        with open(os.path.join(self.models_dir, 'model_b/traditional/hint_generator.pkl'), 'rb') as f:
            self.hint_gen = pickle.load(f)
        with open(os.path.join(self.models_dir, 'model_b/traditional/distractor_generator.pkl'), 'rb') as f:
            self.dist_gen = pickle.load(f)

        self.q_gen = QuestionGenerator(self.model_a_ensemble, self.fe)
        self.fitb_gen = FITBGenerator(self.fe)

    def generate_quiz(self, article, mode="mcq"):
        if mode == "mcq":
            gen_q, gen_ans = self.q_gen.generate_question(article)
            distractors = self.dist_gen.generate_w2v_distractors(article, gen_ans)
        else:
            gen_q, gen_ans = self.fitb_gen.generate_fitb_question(article)
            distractors = self.dist_gen.generate_w2v_distractors(article, gen_ans)
        
        hints = self.hint_gen.generate_hints(article, gen_q)

        return {
            "question": gen_q,
            "correct_answer": gen_ans,
            "distractors": distractors,
            "hints": hints
        }

    def verify_answer(self, article, question, user_choice):
        input_text = f"{article} {question} {user_choice}"
        features, _ = self.fe.transform_corpus([input_text])
        prediction = self.model_a_ensemble.predict(features)[0]
        return prediction == 1

if __name__ == "__main__":
    print("Loading inference API...")
    api = InferenceAPI()
    sample_text = "The quick brown fox jumps over the lazy dog. The dog was very lazy and did not care."
    print("Generating quiz...")
    quiz = api.generate_quiz(sample_text)
    print("Quiz Generated:", quiz)
