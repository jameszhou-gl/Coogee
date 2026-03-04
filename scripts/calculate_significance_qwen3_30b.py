import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def calculate_significance(labels, *score_lists):
    """
    Pools scores from multiple clinicians and runs Mann-Whitney U test.
    """
    real_group = []
    syn_group = []

    # Iterate through each record
    for i, label in enumerate(labels):
        if i>=len(score_lists[0]):
            continue
        # Normalize label to lowercase
        lbl = str(label).lower().strip()
        
        # Collect scores from all clinicians for this index
        for scores in score_lists:
            if scores[i] is None:
                continue
            if lbl in ['real', '1', 'true']:
                real_group.append(scores[i])
            else:
                syn_group.append(scores[i])

    # Convert to numpy arrays for stats
    real_group = np.array(real_group)
    syn_group = np.array(syn_group)

    # Descriptive Stats
    print(f"--- Descriptive Statistics ---")
    print(f"Real Records (n={len(real_group)}):      Mean = {np.mean(real_group):.2f} (SD={np.std(real_group):.2f})")
    print(f"Synthetic Records (n={len(syn_group)}): Mean = {np.mean(syn_group):.2f} (SD={np.std(syn_group):.2f})")
    
    # Mann-Whitney U Test
    # We use this because Likert scores (1-10) are ordinal, not strictly normal.
    u_stat, p_val = stats.mannwhitneyu(real_group, syn_group, alternative='two-sided')
    
    print(f"\n--- Statistical Test (Mann-Whitney U) ---")
    print(f"P-value: {p_val:.5e}")  # Scientific notation for very small numbers

    # Check Significance
    if p_val < 0.001:
        print(f"\n[RESULT] Significant? YES (p < 0.001)")
        print("You can report: 'The difference in realism scores was statistically significant (p < 0.001).'")
    else:
        print(f"\n[RESULT] Significant? NO (p = {p_val:.4f})")
        print("You should report the exact p-value.")

# Load real and synthetic scores
real_data = json.load(open('final_patient_timelines/real/Qwen3_30B.json'))
syn_data = json.load(open('final_patient_timelines/synthetic_post_processed_icd10cm_1/Qwen3_30B.json'))

# Extract realism scores
real_scores = [item['realism_score'] for item in real_data if item.get('realism_score') is not None][:200]
syn_scores = [item['realism_score'] for item in syn_data if item.get('realism_score') is not None][:200]
labels = ['real'] * len(real_scores) + ['synthetic'] * len(syn_scores)
scores = real_scores + syn_scores

calculate_significance(labels, scores)