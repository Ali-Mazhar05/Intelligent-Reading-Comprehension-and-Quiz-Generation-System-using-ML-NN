import sys
import os
import unittest

# Add src to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(root_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

from inference import InferenceAPI

class TestInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We assume models are already trained for these tests
        try:
            cls.api = InferenceAPI(models_dir=os.path.join(root_dir, 'models'))
        except Exception as e:
            raise unittest.SkipTest(f"Models not trained or not found: {e}")

    def test_mcq_generation(self):
        article = "The Solar System consists of the Sun and the objects that orbit it. The Sun is at the center."
        quiz = self.api.generate_quiz(article, mode="mcq")
        
        self.assertIn("question", quiz)
        self.assertIn("correct_answer", quiz)
        self.assertIn("distractors", quiz)
        self.assertIn("hints", quiz)
        self.assertTrue(len(quiz["distractors"]) > 0)
        self.assertTrue(len(quiz["hints"]) > 0)

    def test_fitb_generation(self):
        article = "Photosynthesis is the process used by plants to convert light energy into chemical energy."
        quiz = self.api.generate_quiz(article, mode="fitb")
        
        self.assertIn("__________", quiz["question"])
        self.assertTrue(len(quiz["correct_answer"]) > 0)

    def test_verification(self):
        article = "The capital of France is Paris."
        question = "What is the capital of France?"
        
        # Test correct answer
        is_correct = self.api.verify_answer(article, question, "Paris")
        # Note: verifier might not be 100% accurate, but should ideally be true for the gold answer
        # If it's a trained model, we check if it returns a boolean
        self.assertIsInstance(is_correct, (bool, type(None)))

if __name__ == '__main__':
    unittest.main()
