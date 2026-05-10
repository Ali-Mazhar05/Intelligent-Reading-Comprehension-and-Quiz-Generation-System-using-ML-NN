import json
import os
import sys

def display_metrics():
    # adjust path if called from src/ directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    metrics_path = os.path.join(root_dir, 'models', 'performance_metrics.json')
    
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found at {metrics_path}. Please run training scripts first.")
        return
        
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    print("\n" + "="*40)
    print("   SYSTEM EVALUATION METRICS")
    print("="*40)
    print(f"BLEU Score:                {metrics.get('bleu')}")
    print(f"ROUGE-L Score:             {metrics.get('rouge')}")
    print(f"METEOR Score:              {metrics.get('meteor')}")
    print(f"Ensemble Accuracy:         {metrics.get('ensemble_accuracy')}")
    print(f"Ensemble Macro F1:         {metrics.get('ensemble_f1')}")
    print(f"Semi-Supervised F1:        {metrics.get('semi_supervised_f1')}")
    print(f"Distractor Success Rate:   {metrics.get('distractor_success')}%")
    print("="*40 + "\n")

if __name__ == '__main__':
    display_metrics()
