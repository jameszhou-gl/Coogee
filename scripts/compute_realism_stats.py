import json
import os
from pathlib import Path
from typing import List, Dict
import polars as pl
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--LOG_DIR", type=str, required=True)
    parser.add_argument("--NUM", type=int, required=False, default=7, help="Realism score threshold")
    return parser.parse_args()



def main():
    args = parse_args()
    final_patient_timelines_dir = os.path.join(args.LOG_DIR, "final_patient_timelines")

    subdirs = [
        "synthetic_post_processed_icd10cm_1",
        "synthetic_post_processed_icd10cm_2",
        "synthetic_post_processed_icd10cm_3",
        "synthetic_post_processed_icd10cm_4",
        "synthetic_post_processed_icd10cm_5",
        "synthetic_post_processed_icd10cm_6",
        "synthetic_post_processed_icd10cm_7",
        "synthetic_post_processed_icd10cm_8",
        "synthetic_post_processed_icd10cm_9",
        "synthetic_post_processed_icd10cm_10",
        "synthetic_post_processed_icd10cm_11",
        "synthetic_post_processed_icd10cm_12",
        "synthetic_post_processed_icd10cm_13",
        "synthetic_post_processed_icd10cm_14",
        "synthetic_post_processed_icd10cm_15",
        "synthetic_post_processed_icd10cm_16"
    ]

    all_records: List[Dict] = []

    for subdir in subdirs:
        json_path = os.path.join(final_patient_timelines_dir, subdir, "Qwen3_30B_with_reasoning.json")
        records = json.load(open(json_path))
        all_records.extend(records)

    total = len(all_records)
    
    high_realism = [r for r in all_records if r.get("realism_score", 0) >= args.NUM]
    num_high = len(high_realism)
    ratio_high = num_high / total if total > 0 else 0.0

    print("===== Realism Score Statistics =====")
    print(f"Total files           : {total}")
    print(f"Realism score >= {args.NUM}     : {num_high}")
    print(f"Ratio (>={args.NUM})            : {ratio_high:.4f}")

    # Optional: sanity check output
    # print("Example high-realism files:", high_realism[:5])
    syn_data = pl.read_parquet(os.path.join(args.LOG_DIR, "synthetic_data_topp_0.98_temperature_1.0/synthetic_data.parquet"))
    
    # Extract subject_ids with realism_score >= 7
    # The JSON records have "file" field like "11168175.csv", so extract the numeric subject_id
    high_realism_subject_ids = [int(r.get("file").replace(".csv", "")) for r in high_realism]
    print(f"\nFiltering synthetic data to {len(high_realism_subject_ids)} subjects with realism_score >= {args.NUM}")
    
    # Filter synthetic data
    filtered_syn_data = syn_data.filter(pl.col("subject_id").is_in(high_realism_subject_ids))
    print(f"Filtered data shape: {filtered_syn_data.shape}")
    print(f"Original data shape: {syn_data.shape}")
    
    # Create output directory and save filtered data
    output_dir = os.path.join(args.LOG_DIR, f"synthetic_data_topp_0.98_temperature_1.0_realism_geq_{args.NUM}_w_reasoning")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "synthetic_data.parquet")
    
    filtered_syn_data.write_parquet(output_path)
    print(f"\nFiltered synthetic data saved to: {output_path}")

if __name__ == "__main__":
    main()
