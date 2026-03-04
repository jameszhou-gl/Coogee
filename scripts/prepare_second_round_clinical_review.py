import os
import json
import shutil
import random
import argparse
import pandas as pd
from pathlib import Path

# Set random seed for reproducibility
random.seed(1337)

parser = argparse.ArgumentParser()
parser.add_argument('--LOG_DIR', type=str, required=True, help='Path that stores the final patient timelines')
args = parser.parse_args()

NUM = 20
SCORE_THRESHOLD = 7
real_dir = os.path.join(args.LOG_DIR, "final_patient_timelines_for_clinical_review", "real")
synthetic_dir = os.path.join(args.LOG_DIR, "final_patient_timelines_for_clinical_review", "synthetic_post_processed_icd10cm_1")
output_dir = os.path.join(args.LOG_DIR, "final_patient_timelines_for_clinical_review", "clinical_review_round2")
os.makedirs(output_dir, exist_ok=True)

qwen_scores = json.load(open(os.path.join(synthetic_dir, "Qwen3_30B_with_reasoning.json")))
file_mapping = []

chosen_num = 0
chosen_ids = []
for score in qwen_scores:
    if score['realism_score'] >= SCORE_THRESHOLD:
        file_mapping.append({
            'original_file': score['file'],
            'original_path': os.path.join(synthetic_dir, score['file']),
            'type': 'synthetic',
            'subject_id': score['file'].replace('.csv', '')
        })
        chosen_num += 1
        chosen_ids.append(score['file'].replace('.csv', ''))
        if chosen_num >= NUM:
            break

if len(file_mapping) < NUM:
    print(f"Only {len(file_mapping)} files found with score >= {SCORE_THRESHOLD}, expected {NUM}")
    exit()

# Get real files
real_files = sorted([f for f in os.listdir(real_dir) if f.endswith('.csv')])
for file in real_files:
    subject_id = file.replace('.csv', '')
    if subject_id not in chosen_ids:
        continue
    file_mapping.append({
        'original_file': file,
        'original_path': os.path.join(real_dir, file),
        'type': 'real',
        'subject_id': subject_id
    })

# print(f"Found {len(real_files)} real files")
print(f"Total files: {len(file_mapping)}")
# print(file_mapping)

# # Shuffle the files
random.shuffle(file_mapping)

# Copy and rename files
mapping_records = []
for idx, file_info in enumerate(file_mapping, start=1):
    new_filename = f"{idx:03d}.csv"
    new_path = os.path.join(output_dir, new_filename)
    
    # Copy file
    shutil.copy2(file_info['original_path'], new_path)
    
    # Record mapping
    mapping_records.append({
        'new_filename': new_filename,
        'type': file_info['type'],
        'subject_id': file_info['subject_id']
    })
    
    print(f"Copied {file_info['original_file']} -> {new_filename} ({file_info['type']})")

# Save mapping to CSV
mapping_df = pd.DataFrame(mapping_records)
mapping_csv_path = os.path.join(args.LOG_DIR, "final_patient_timelines_for_clinical_review", "clinical_review_round2", "clinical_review_round2_mapping.csv")
mapping_df.to_csv(mapping_csv_path, index=False)

print(f"\nMapping saved to {mapping_csv_path}")
print(f"\nSummary:")
print(f"Total files copied: {len(mapping_records)}")
print(f"Real files: {len([r for r in mapping_records if r['type'] == 'real'])}")
print(f"Synthetic files: {len([r for r in mapping_records if r['type'] == 'synthetic'])}")
print(f"\nAll files have been copied to '{output_dir}' folder")
print("\nFirst few mappings:")
print(mapping_df.head(10))

# python -m scripts.prepare_second_round_clinical_review --LOG_DIR output/coogee-final-sanity-check/2025-12-19_10_58_00-rm_know_emb_labtest_w_n_embd_factor