import shutil
import os
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--LOG_DIR', type=str, required=True, help='Path that stores the final patient timelines')
args = parser.parse_args()

# Source directory
src_dir = Path(os.path.join(args.LOG_DIR, "final_patient_timelines", "synthetic_post_processed_icd10cm"))

# Number of splits
SPLIT_NUM = 16

# Target directories - dynamically create based on SPLIT_NUM
dst_dirs = [
    Path(os.path.join(args.LOG_DIR, "final_patient_timelines", f"synthetic_post_processed_icd10cm_{i+1}"))
    for i in range(SPLIT_NUM)
]

# Create target directories
for d in dst_dirs:
    d.mkdir(parents=True, exist_ok=True)

# Collect and sort CSV files (deterministic split)
csv_files = sorted(f for f in src_dir.iterdir() if f.is_file() and f.suffix == ".csv")

n = len(csv_files)
print(f"Found {n} CSV files. Splitting into {SPLIT_NUM} parts.")

# Even split - dynamically create based on SPLIT_NUM
splits = []
for i in range(SPLIT_NUM):
    start_idx = i * n // SPLIT_NUM
    end_idx = (i + 1) * n // SPLIT_NUM if i < SPLIT_NUM - 1 else n
    splits.append(csv_files[start_idx:end_idx])
    print(f"Split {i+1}: {len(splits[i])} files")

# Copy files
for files, dst in zip(splits, dst_dirs):
    for f in files:
        shutil.copy2(f, dst / f.name)

print("Copying completed.")
