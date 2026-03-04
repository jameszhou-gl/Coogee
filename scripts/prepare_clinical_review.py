import os
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

real_dir = os.path.join(args.LOG_DIR, "final_patient_timelines", "real")
synthetic_dir = os.path.join(args.LOG_DIR, "final_patient_timelines", "synthetic_post_processed_icd10cm")
output_dir = os.path.join(args.LOG_DIR, "final_patient_timelines", "clinical_review_round1")
os.makedirs(output_dir, exist_ok=True)

# Collect files from both directories
file_mapping = []

# Get real files
real_files = sorted([f for f in os.listdir(real_dir) if f.endswith('.csv')])
for file in real_files:
    subject_id = file.replace('.csv', '')
    file_mapping.append({
        'original_file': file,
        'original_path': os.path.join(real_dir, file),
        'type': 'real',
        'subject_id': subject_id
    })

# Get synthetic files
synthetic_files = sorted([f for f in os.listdir(synthetic_dir) if f.endswith('.csv')])
for file in synthetic_files:
    subject_id = file.replace('.csv', '')
    file_mapping.append({
        'original_file': file,
        'original_path': os.path.join(synthetic_dir, file),
        'type': 'synthetic',
        'subject_id': subject_id
    })

print(f"Found {len(real_files)} real files")
print(f"Found {len(synthetic_files)} synthetic files")
print(f"Total files: {len(file_mapping)}")

# Shuffle the files
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
mapping_csv_path = os.path.join(args.LOG_DIR, "final_patient_timelines", "clinical_review_round1", "clinical_review_round1_mapping.csv")
mapping_df.to_csv(mapping_csv_path, index=False)

print(f"\nMapping saved to {mapping_csv_path}")
print(f"\nSummary:")
print(f"Total files copied: {len(mapping_records)}")
print(f"Real files: {len([r for r in mapping_records if r['type'] == 'real'])}")
print(f"Synthetic files: {len([r for r in mapping_records if r['type'] == 'synthetic'])}")
print(f"\nAll files have been copied to '{output_dir}' folder")
print("\nFirst few mappings:")
print(mapping_df.head(10))

# python -m scripts.prepare_clinical_review --LOG_DIR output/coogee-final-sanity-check/2025-12-12_14_38_00-rm_know_emb_labtest