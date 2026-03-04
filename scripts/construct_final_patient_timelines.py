import argparse
import os
import json
import numpy as np
import polars as pl
import random
from typing import List
from datetime import datetime, timedelta
from tqdm import tqdm
from model.local_tokenizer import LocalTokenizer
from scripts.utils import SEPARATORS, SEPARATOR_NAMES

SEED = 1337
np.random.seed(SEED)

def sample_lab_test_value(lab_test_code: str, quantiles_code: str, lab_test_quantiles: dict, lab_test_format: dict) -> str:
    """Sample a lab test value from the quantiles and return as formatted string.
    
    The quantiles dict has P0-P100 boundaries defining 10 bins:
    - _Q1: [P0, P10]
    - _Q2: (P10, P20]
    - ...
    - _Q10: (P90, P100]
    """    
    if lab_test_code not in lab_test_quantiles:
        raise ValueError(f"Lab test code {lab_test_code} not found in lab test quantiles")
    
    quantiles = lab_test_quantiles[lab_test_code]
    q_num = int(quantiles_code.replace("_Q", ""))  # _Q1 -> 1, _Q10 -> 10
    
    # Map bucket to percentile boundaries
    percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    lower_p, upper_p = percentiles[q_num - 1], percentiles[q_num]
    lower = quantiles[f"P{lower_p}"]
    upper = quantiles[f"P{upper_p}"]
    
    # Handle degenerate case (e.g., when 40% of values are exactly 0.0)
    if lower == upper:
        sample_val = lower
    else:
        sample_val = np.random.uniform(lower, upper)
    
    # Return as formatted string to preserve integer vs float formatting in CSV
    if lab_test_format[lab_test_code]['most_common_format'] == 'integer':
        return str(int(round(sample_val)))
    elif lab_test_format[lab_test_code]['most_common_format'] == 'one_decimal':
        return str(round(sample_val, 1))
    else:
        return str(round(sample_val, 2))
    
def is_time_gap_code(code: str) -> bool:
    """Check if the code is a time gap code"""
    return code in SEPARATOR_NAMES

def sample_minutes_value(time_gap_code: str) -> int:
    """Sample a time value from the time gap code"""
    upper_minutes = SEPARATORS[time_gap_code]
    sep_ind = SEPARATOR_NAMES.index(time_gap_code)
    lower_minutes = 0 if sep_ind == 0 else SEPARATORS[SEPARATOR_NAMES[sep_ind - 1]]
    return int(np.random.uniform(lower_minutes, upper_minutes))

def random_timestamp_for_year(year: int) -> datetime:
    """Generate a random datetime for a given year with seconds=0."""
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31, 23, 59, 0)
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    dt = start + timedelta(seconds=random_seconds)
    # Zero out seconds
    return dt.replace(second=0)

def safe_year_shift(dt: datetime, year_offset: int) -> datetime:
    """Safely shift a datetime's year, handling leap year edge cases (e.g., Feb 29)."""
    try:
        return dt.replace(year=dt.year - year_offset)
    except ValueError:
        # Handle Feb 29 in leap year -> non-leap year
        # If the day is invalid for the target year, use the last valid day of that month
        if dt.month == 2 and dt.day == 29:
            return dt.replace(year=dt.year - year_offset, day=28)
        else:
            raise

def construct_syn_patient_timeline(df_sequence: pl.DataFrame, tokenizer: LocalTokenizer, save_dir: str, code_to_label: dict, lab_test_quantiles: dict, lab_test_format: dict):
    """Construct a final patient timeline from a synthetic sequence"""
    # we use four columns: time, code, numerical value (for lab tests), code label (for ICD and any other labeled codes), and save in a csv file
    df_sequence = df_sequence.with_columns(pl.col("synthetic_sequence").alias("token_seq")).select(["subject_id", "token_seq"])
    subject_ids = []
    for subject_id, token_seq in tqdm(df_sequence.iter_rows(), total=len(df_sequence), desc="Processing subjects"):
        # print(f"Subject {subject_id}")
        current_time = random_timestamp_for_year(int(tokenizer.decode_token_id(token_seq[5]).split("_")[1]))
        rows = []
        last_lab_test = None
        for token_id in token_seq:
            medical_concept = tokenizer.decode_token_id(token_id)
            if is_time_gap_code(medical_concept):
                time_gap_code = medical_concept
                current_time += timedelta(minutes=sample_minutes_value(time_gap_code))
            else:
                if medical_concept.startswith("LAB_"):
                    last_lab_test = medical_concept
                    continue
                elif medical_concept.startswith("_Q"):
                    lab_test_value = sample_lab_test_value(last_lab_test, medical_concept, lab_test_quantiles, lab_test_format)
                    medical_concept = last_lab_test
                    code_label = None
                else:
                    code_label = code_to_label[medical_concept]
                    lab_test_value = None
                rows.append({
                    "time": current_time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "code": medical_concept,
                    "numerical_value": lab_test_value,
                    "code_label": code_label
                })
                # Stop processing after END_RECORD (avoid multiple patient records in one sequence)
                if medical_concept == "END_RECORD":
                    break
        df = pl.DataFrame(rows, infer_schema_length=None)
        df = df.select(["time", "code", "numerical_value", "code_label"])
        df.write_csv(os.path.join(save_dir, f"{subject_id}.csv"))
        subject_ids.append(subject_id)
    return subject_ids

def format_lab_value(code: str, numeric_value: float, lab_test_format: dict) -> str:
    """Format lab test value based on lab_test_format (integer, one_decimal, two_or_more_decimal)."""
    if code not in lab_test_format:
        return str(round(numeric_value, 2))
    
    fmt = lab_test_format[code].get('most_common_format', 'two_or_more_decimal')
    if fmt == 'integer':
        return str(int(round(numeric_value)))
    elif fmt == 'one_decimal':
        return str(round(numeric_value, 1))
    else:
        return str(round(numeric_value, 2))

def construct_real_patient_timeline(real_data: pl.DataFrame, subject_ids: List[int], code_to_label: dict, lab_test_format: dict, save_dir: str):
    """Construct patient timeline from real data with date shifting and visit markers."""
    # Demographic codes that should use first visit time
    DEMOGRAPHIC_CODES = ["AGE_", "SEX_", "RACE_", "MARITAL_STATUS_", "YEAR_"]
    
    for subject_id in tqdm(subject_ids, total=len(subject_ids), desc="Processing subjects"):
        # print(f"Real Subject {subject_id}")
        df_subj = real_data.filter(pl.col("subject_id") == subject_id)
        
        # Get actual year from YEAR_XXXX code (row index 4)
        year_code = df_subj.row(4)[2]  # column 2 is 'code'
        actual_year = int(year_code.split("_")[1])  # YEAR_2014 -> 2014
        
        # Get de-identified year from first visit timestamp (row index 5 or later with visit_id)
        first_visit_time = df_subj.row(5)[1]  # column 1 is 'time'
        # Handle both datetime object and string
        if isinstance(first_visit_time, datetime):
            first_visit_dt = first_visit_time
        else:
            first_visit_dt = datetime.fromisoformat(str(first_visit_time).replace(".000000", ""))
        deidentified_year = first_visit_dt.year
        
        # Calculate year offset
        year_offset = deidentified_year - actual_year
        
        # Shift the first visit time to get demographics time
        first_visit_shifted = safe_year_shift(first_visit_dt, year_offset)
        demographics_time = first_visit_shifted.strftime("%Y-%m-%dT%H:%M:%S")
        
        rows = []
        current_visit_id = None
        last_time_formatted = None
        
        for row in df_subj.iter_rows():
            subject_id_row, time_val, code, numeric_value, seq_num, table, text_value, visit_id, sort_order = row
            
            # Skip _Q codes (quantile markers)
            if code.startswith("_Q"):
                continue
            
            # Parse and shift the timestamp
            if time_val:
                if isinstance(time_val, datetime):
                    dt = time_val
                else:
                    dt = datetime.fromisoformat(str(time_val).replace(".000000", ""))
                shifted_dt = safe_year_shift(dt, year_offset)
                time_formatted = shifted_dt.strftime("%Y-%m-%dT%H:%M:%S")
            else:
                time_formatted = demographics_time
            
            # Check if this is a demographic code
            is_demographic = any(code.startswith(prefix) for prefix in DEMOGRAPHIC_CODES)
            
            if is_demographic:
                # Add START_RECORD before first demographic
                if not rows:
                    rows.append({
                        "time": demographics_time,
                        "code": "START_RECORD",
                        "numerical_value": None,
                        "code_label": None
                    })
                # Use first visit time for demographics
                time_formatted = demographics_time
            else:
                # Handle visit transitions
                if visit_id is not None and visit_id != current_visit_id:
                    # End previous visit if exists
                    if current_visit_id is not None and last_time_formatted:
                        rows.append({
                            "time": last_time_formatted,
                            "code": "END_VISIT",
                            "numerical_value": None,
                            "code_label": None
                        })
                    # Start new visit
                    rows.append({
                        "time": time_formatted,
                        "code": "START_VISIT",
                        "numerical_value": None,
                        "code_label": None
                    })
                    current_visit_id = visit_id
            
            # Get code label
            code_label = code_to_label.get(code, None)
            
            # Format numeric value
            if code.startswith("AGE_"):
                # Remove numeric value for AGE codes
                numeric_value_str = None
            elif code.startswith("LAB_") and numeric_value is not None:
                # Format lab values using lab_test_format
                numeric_value_str = format_lab_value(code, numeric_value, lab_test_format)
            elif numeric_value is not None:
                numeric_value_str = str(numeric_value)
            else:
                numeric_value_str = None
            
            rows.append({
                "time": time_formatted,
                "code": code,
                "numerical_value": numeric_value_str,
                "code_label": code_label
            })
            
            last_time_formatted = time_formatted
        
        # Add END_VISIT for the last visit and END_RECORD
        if last_time_formatted:
            rows.append({
                "time": last_time_formatted,
                "code": "END_VISIT",
                "numerical_value": None,
                "code_label": None
            })
            rows.append({
                "time": last_time_formatted,
                "code": "END_RECORD",
                "numerical_value": None,
                "code_label": None
            })
        
        df = pl.DataFrame(rows, infer_schema_length=None)
        df = df.select(["time", "code", "numerical_value", "code_label"])
        df.write_csv(os.path.join(save_dir, f"{subject_id}.csv"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--LOG_DIR", required=True, type=str)
    ap.add_argument("--n_samples", required=False, type=int, default=20)
    args = ap.parse_args()
    # paths
    final_patient_timelines_dir = os.path.join(args.LOG_DIR, "final_patient_timelines")
    os.makedirs(final_patient_timelines_dir, exist_ok=True)
    
    syn_data_dir = os.path.join(args.LOG_DIR, "synthetic_data_topp_0.98_temperature_1.0")
    synthetic_data = pl.read_parquet(os.path.join(syn_data_dir, "synthetic_data.parquet"))
    synthetic_data = synthetic_data.filter(pl.col("synthetic_sequence").list.len() < 2048)
    if args.n_samples == -1:
        n_samples = synthetic_data.shape[0]
    else:
        n_samples = args.n_samples
    synthetic_data = synthetic_data.sample(n=n_samples, shuffle=True, seed=SEED)
    os.makedirs(os.path.join(final_patient_timelines_dir, "synthetic"), exist_ok=True)
    
    tokenizer = LocalTokenizer(os.path.join(args.LOG_DIR, "tokenizer", "vocab.csv"))
    code_label_df = pl.read_csv(os.path.join(args.LOG_DIR, "tokenizer", "vocab_w_concept_label.csv"))
    code_to_label = dict(zip(code_label_df["concept_code"], code_label_df["concept_label"]))
    lab_test_quantiles = json.load(open(os.path.join(args.LOG_DIR, "cohort_stat", "lab_test_quantiles.json")))
    lab_test_format = json.load(open(os.path.join(args.LOG_DIR, "cohort_stat", "lab_test_format.json")))
    subject_ids = construct_syn_patient_timeline(synthetic_data, tokenizer, os.path.join(final_patient_timelines_dir, "synthetic"), code_to_label, lab_test_quantiles, lab_test_format)
    
    real_data_dir = os.path.join(args.LOG_DIR, "cohort_stat")
    real_data = pl.read_parquet(os.path.join(real_data_dir, "mimiciv_2.2_meds_processed.parquet"))
    real_data = real_data.filter(pl.col("subject_id").is_in(subject_ids))
    # real_data = real_data.filter(pl.col("code").list.len() < 2048).sample(n=args.n_samples, shuffle=True, seed=SEED)
    os.makedirs(os.path.join(final_patient_timelines_dir, "real"), exist_ok=True)
    construct_real_patient_timeline(real_data, subject_ids, code_to_label, lab_test_format, os.path.join(final_patient_timelines_dir, "real"))

if __name__ == "__main__":
    main()
# python -m scripts.construct_final_patient_timelines --LOG_DIR output/coogee-final-sanity-check/2025-12-12_14_38_00-rm_know_emb_labtest