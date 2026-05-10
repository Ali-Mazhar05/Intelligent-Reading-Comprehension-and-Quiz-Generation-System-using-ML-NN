import sys
import os

# Add src to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(root_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

from preprocessing import FeatureEngineer
from model_b_train import DistractorGenerator

fe = FeatureEngineer(use_tf_idf=False)
try:
    fe.load_models(load_directory=os.path.join(root_dir, 'models/traditional'))
except:
    pass # might not be fitted yet in this environment

dg = DistractorGenerator(fe)
article = "The quick brown fox jumps over the lazy dog."
answer = "dog"

print("Testing W2V distractor generation...")
try:
    distractors = dg.generate_w2v_distractors(article, answer)
    print("Distractors:", distractors)
except Exception as e:
    print("Error during W2V generation:", e)
