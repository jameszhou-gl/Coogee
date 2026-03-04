#!/usr/bin/env python3
"""
Post-process synthetic patient timelines to ensure ICD10CM codes are:
1. Grouped together (no non-ICD codes between them)
2. Placed at the end of each visit (right before END_VISIT)
3. Have timestamps matching END_VISIT
"""

import os
import csv
import argparse
from pathlib import Path
from tqdm import tqdm


def process_visit(visit_rows):
    """
    Process a single visit to group ICD10CM codes at the end.
    
    Args:
        visit_rows: List of row dicts for a single visit (excluding START_VISIT and END_VISIT)
    
    Returns:
        List of processed row dicts with ICD codes grouped at the end
    """
    if not visit_rows:
        return visit_rows
    
    # Separate ICD and non-ICD rows
    icd_rows = [row for row in visit_rows if row['code'].startswith('ICD10CM_')]
    non_icd_rows = [row for row in visit_rows if not row['code'].startswith('ICD10CM_')]
    
    # Return non-ICD rows first, then ICD rows (grouped at the end)
    return non_icd_rows + icd_rows


def process_file(input_path, output_path):
    """
    Process a single CSV file to group ICD10CM codes at the end of each visit.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
    """
    rows = []
    
    # Read all rows
    with open(input_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    # Process rows by visit
    processed_rows = []
    current_visit = []
    end_visit_time = None
    
    for row in rows:
        code = row['code']
        
        if code == 'START_VISIT':
            # Add START_VISIT as-is
            processed_rows.append(row)
            current_visit = []
            end_visit_time = None
            
        elif code == 'END_VISIT':
            # Get END_VISIT timestamp
            end_visit_time = row['time']
            
            # Process the visit: group ICD codes at the end
            processed_visit = process_visit(current_visit)
            
            # Update ICD code timestamps to match END_VISIT
            for visit_row in processed_visit:
                if visit_row['code'].startswith('ICD10CM_'):
                    visit_row['time'] = end_visit_time
                processed_rows.append(visit_row)
            
            # Add END_VISIT
            processed_rows.append(row)
            current_visit = []
            
        elif code in ['START_RECORD', 'END_RECORD']:
            # Add record markers as-is
            processed_rows.append(row)
            
        elif code.startswith(('AGE_', 'SEX_', 'RACE_', 'MARITAL_STATUS_', 'YEAR_')):
            # Add demographic codes as-is (they're outside visits)
            processed_rows.append(row)
            
        else:
            # Collect visit content
            current_visit.append(row)
    
    # Write processed rows
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--LOG_DIR', type=str, required=True, help='Path that stores the final patient timelines')
    args = parser.parse_args()
    real_path = os.path.join(args.LOG_DIR, "final_patient_timelines", "real")
    synthetic_path = os.path.join(args.LOG_DIR, "final_patient_timelines", "synthetic")
    synthetic_post_processed_path = os.path.join(args.LOG_DIR, "final_patient_timelines", "synthetic_post_processed_icd10cm")
    os.makedirs(synthetic_post_processed_path, exist_ok=True)
    synthetic_post_processed_dir = Path(synthetic_post_processed_path)
    
    # Get all CSV files
    csv_files = sorted(Path(synthetic_path).glob("*.csv"))
    print(f"Found {len(csv_files)} files in synthetic directory to process")
    
    # Process each file
    for i, csv_file in enumerate(tqdm(csv_files, desc="Processing files"), 1):
        output_file = synthetic_post_processed_dir / csv_file.name
        process_file(csv_file, output_file)
    print(f"\nDone! Processed files saved to: {synthetic_post_processed_dir}")


if __name__ == "__main__":
    main()

# python -m scripts.post_process_icd10cm_syn --LOG_DIR output/coogee-final-sanity-check/2025-12-12_14_38_00-rm_know_emb_labtest