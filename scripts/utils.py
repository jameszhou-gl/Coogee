import json
import os
import yaml
import numpy as np
import polars as pl
from typing import Dict, List, Any
from datetime import datetime

# Define separators in fractional years
# intervals are right-inclusive: (a, b]
# TODO: consider change to time gaps in minutes as in output/exp_coogee_lab_test_constrained_timegap_head/2025-11-29_01_10_19/raw_sequences/timegap_analysis_20buckets/timegap_analysis.json
SEPARATORS = {
    "_<=5m": 5,
    "_5m-15m": 15,
    "_15m-1h": 60,
    "_1h-2h": 2 * 60,
    "_2h-6h": 6 * 60,
    "_6h-12h": 12 * 60,
    "_12h-1d": 24 * 60,
    "_1d-3d": 3 * 24 * 60,
    "_3d-1w": 7 * 24 * 60,
    "_1w-2w": 2 * 7 * 24 * 60,
    "_2w-1mt": 30 * 24 * 60,
    "_1mt-3mt": 3 * 30 * 24 * 60,
    "_3mt-6mt": 6 * 30 * 24 * 60,
    "_>6mt": 12 * 30 * 24 * 60,
}

# Sort keys for ordered comparison
SEPARATOR_NAMES = list(SEPARATORS.keys())
SEPARATOR_SIZES = list(SEPARATORS.values())

def compute_years_delta(t1: datetime, t2: datetime) -> float:
    """Compute fractional year difference between two datetimes."""
    delta_days = (t2 - t1).days
    delta_seconds = (t2 - t1).seconds
    total_days = (t2 - t1).total_seconds() / (24 * 60 * 60)
    return round(total_days / 365.25, 2)

def compute_minutes_delta(t1: datetime, t2: datetime) -> float:
    """Compute minutes difference between two datetimes."""
    delta_seconds = (t2 - t1).total_seconds()
    return round(delta_seconds / 60, 2)

def gap_to_separator(t1: datetime, t2: datetime) -> str:
    """Given two timestamps, return the symbolic token representing the time gap."""
    if t2 < t1:
        raise ValueError(f"t2 must be after t1: t1: {t1}, t2: {t2}")
    delta_minutes = compute_minutes_delta(t1, t2)

    # Special handling for gaps less than or equal to 5 minutes
    if delta_minutes <= SEPARATOR_SIZES[0]:  # First threshold is 5 minutes
        return SEPARATOR_NAMES[0]  # Return "_<=5m"
    
    # Rest of the logic for larger gaps
    for i in range(len(SEPARATOR_NAMES) - 1):
        if delta_minutes <= SEPARATOR_SIZES[i]:
            return SEPARATOR_NAMES[i]
    
    # If we've exceeded all intervals except the last one
    return SEPARATOR_NAMES[-1]  # Return _>6mt for gaps > 6 months



def get_dataset_statistics(df_events: pl.DataFrame, LOG_DIR: str) -> None:
    """Compute and display various statistics about the dataset."""
    # Create a dictionary to store all statistics
    stats = {}

    print("\nDataset Statistics:")
    print("=" * 50)

    # Number of unique patients
    n_patients = df_events.select('subject_id').unique().shape[0]
    stats['n_patients'] = n_patients
    print(f"\n1. Total number of unique patients: {n_patients}")

    # Vocabulary statistics
    total_vocab = df_events.select('code').unique().shape[0]
    stats['vocabulary'] = {'total_size': total_vocab}
    print(f"\n2. Vocabulary Statistics:")
    print(f"   Total vocabulary size: {total_vocab}")

    # Count codes by type
    code_types = {
        'ATC': 'ATC_',
        'ICD10CM': 'ICD10CM_',
        'ICD10PCS': 'ICD10PCS_',
        'Demographics': 'SEX_|RACE_|MARITAL_STATUS_|AGE_|YEAR_',
        'Death': 'DEATH',
        'Lab Tests': 'LAB_'
    }

    # Initialize dictionaries for different statistics
    stats['unique_codes_by_type'] = {}
    stats['events_by_type'] = {}

    print("\n3. Number of unique codes by type:")
    for code_type, prefix in code_types.items():
        count = df_events.filter(pl.col('code').str.contains(
            prefix)).select('code').unique().shape[0]
        stats['unique_codes_by_type'][code_type] = count
        print(f"   {code_type}: {count}")

    # Sequence length statistics
    seq_lengths = (
        df_events
        .group_by('subject_id')
        .agg(pl.count().alias('seq_length'))
        .select('seq_length')
    )

    min_len = seq_lengths.min().item()
    max_len = seq_lengths.max().item()
    avg_len = seq_lengths.mean().item()
    median_len = seq_lengths.median().item()

    stats['sequence_length'] = {
        'min': min_len,
        'max': max_len,
        'mean': round(avg_len, 2),
        'median': median_len
    }

    print("\n4. Sequence Length Statistics:")
    print(f"   Minimum length: {min_len}")
    print(f"   Maximum length: {max_len}")
    print(f"   Average length: {avg_len:.2f}")
    print(f"   Median length: {median_len}")

    # Distribution of sequence lengths
    percentiles = [25, 50, 75, 90, 95, 99]
    stats['sequence_length_percentiles'] = {}

    print("\n5. Sequence Length Percentiles:")
    for p in percentiles:
        length = seq_lengths.select(
            pl.col('seq_length').quantile(p/100)).item()
        stats['sequence_length_percentiles'][f'p{p}'] = round(length, 2)
        print(f"   {p}th percentile: {length:.0f}")

    # Count events by type
    print("\n6. Number of events by code type:")
    total_events = df_events.shape[0]
    stats['total_events'] = total_events

    for code_type, prefix in code_types.items():
        count = df_events.filter(pl.col('code').str.contains(prefix)).shape[0]
        percentage = (count / total_events) * 100
        stats['events_by_type'][code_type] = {
            'count': count,
            'percentage': f"{percentage:.2f}%"
        }
        print(f"   {code_type}: {count} ({percentage:.2f}%)")

    # Demographic ratios analysis
    print("\n7. Demographic Ratios:")
    stats['demographic_ratios'] = {}
    
    # Sex ratios
    sex_data = (
        df_events
        .filter(pl.col('code').str.starts_with('SEX_'))
        .group_by('code')
        .agg(pl.col('subject_id').n_unique().alias('count'))
        .sort('count', descending=True)
    )
    
    if sex_data.shape[0] > 0:
        total_sex = sex_data['count'].sum()
        sex_ratios = {}
        print("   Sex distribution:")
        print(f"   The number of patients with gender: {total_sex}")
        for row in sex_data.iter_rows(named=True):
            code = row['code'].replace('SEX_', '')
            count = row['count']
            ratio = (count / total_sex) * 100
            sex_ratios[code] = {
                'count': count,
                'percentage': f"{ratio:.2f}%"
            }
            print(f"     {code}: {count} ({ratio:.2f}%)")
        stats['demographic_ratios']['sex'] = sex_ratios
    
    # Race ratios
    race_data = (
        df_events
        .filter(pl.col('code').str.starts_with('RACE_'))
        .group_by('code')
        .agg(pl.col('subject_id').n_unique().alias('count'))
        .sort('count', descending=True)
    )
    
    if race_data.shape[0] > 0:
        total_race = race_data['count'].sum()
        race_ratios = {}
        print("   Race distribution:")
        print(f"   The number of patients with race: {total_race}")
        for row in race_data.iter_rows(named=True):
            code = row['code'].replace('RACE_', '')
            count = row['count']
            ratio = (count / total_race) * 100
            race_ratios[code] = {
                'count': count,
                'percentage': f"{ratio:.2f}%"
            }
            print(f"     {code}: {count} ({ratio:.2f}%)")
        stats['demographic_ratios']['race'] = race_ratios
    
    # Marital Status ratios
    marital_data = (
        df_events
        .filter(pl.col('code').str.starts_with('MARITAL_STATUS_'))
        .group_by('code')
        .agg(pl.col('subject_id').n_unique().alias('count'))
        .sort('count', descending=True)
    )
    
    if marital_data.shape[0] > 0:
        total_marital = marital_data['count'].sum()
        marital_ratios = {}
        print("   Marital Status distribution:")
        print(f"   The number of patients with marital status: {total_marital}")
        for row in marital_data.iter_rows(named=True):
            code = row['code'].replace('MARITAL_STATUS_', '')
            count = row['count']
            ratio = (count / total_marital) * 100
            marital_ratios[code] = {
                'count': count,
                'percentage': f"{ratio:.2f}%"
            }
            print(f"     {code}: {count} ({ratio:.2f}%)")
        stats['demographic_ratios']['marital_status'] = marital_ratios
    
    # Age group ratios
    age_data = (
        df_events
        .filter(pl.col('code').str.starts_with('AGE_'))
        .group_by('code')
        .agg(pl.col('subject_id').n_unique().alias('count'))
        .sort('count', descending=True)
    )
    
    if age_data.shape[0] > 0:
        total_age = age_data['count'].sum()
        age_ratios = {}
        print("   Age group distribution:")
        print(f"   The number of patients with age: {total_age}")
        for row in age_data.iter_rows(named=True):
            code = row['code'].replace('AGE_', '')
            count = row['count']
            ratio = (count / total_age) * 100
            age_ratios[code] = {
                'count': count,
                'percentage': f"{ratio:.2f}%"
            }
            print(f"     {code}: {count} ({ratio:.2f}%)")
        stats['demographic_ratios']['age'] = age_ratios
        
    # Year group ratios
    year_data = (
        df_events
        .filter(pl.col('code').str.starts_with('YEAR_'))
        .group_by('code')
        .agg(pl.col('subject_id').n_unique().alias('count'))
        .sort('count', descending=True)
    )

    if year_data.shape[0] > 0:
        total_year = year_data['count'].sum()
        year_ratios = {}
        print("   Year group distribution:")
        print(f"   The number of patients with year: {total_year}")
        for row in year_data.iter_rows(named=True):
            code = row['code'].replace('YEAR_', '')
            count = row['count']
            ratio = (count / total_year) * 100
            year_ratios[code] = {
                'count': count,
                'percentage': f"{ratio:.2f}%"
            }
            print(f"     {code}: {count} ({ratio:.2f}%)")
        stats['demographic_ratios']['year'] = year_ratios

    # Save statistics to JSON file
    json_path = f"{LOG_DIR}/mimiciv_2.2_meds_processed_statistics.json"
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"\nStatistics saved to {json_path}")

def remove_unused_columns(df_events: pl.DataFrame) -> pl.DataFrame:
    """Remove unused columns: caregiver_id, comments, priority."""
    columns_to_drop = ['caregiver_id', 'comments', 'priority']
    
    # Check which columns exist in the DataFrame before dropping
    existing_columns = df_events.columns
    columns_to_drop_existing = [col for col in columns_to_drop if col in existing_columns]
    return df_events.drop(columns_to_drop_existing)
    
def save_all_plus_one_subject(df_events: pl.DataFrame, LOG_DIR: str) -> None:
    
    sel_subject = 10024043
    df_events.write_parquet(
        f"{LOG_DIR}/mimiciv_2.2_meds_processed.parquet")
    # # Save example patient as CSV for easy viewings
    # df_events.filter(pl.col('subject_id') == sel_subject).write_csv(
    #     f"{LOG_DIR}/mimiciv_2.2_meds_processed_{sel_subject}.csv")
    
def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config