import polars as pl
import argparse
import logging
from typing import List, Dict
from scripts.utils import gap_to_separator, compute_years_delta, compute_minutes_delta, load_config
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import os
import warnings
import yaml
import numpy as np
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Tokenize the filtered processed MIMIC-IV data.')
    parser.add_argument('--LOG_DIR', type=str, default='output/del',
                        help='Path to save the vocabulary file')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed for train/val/test split')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Test set ratio')
    parser.add_argument('--config', type=str, default='configs/basic.yaml',
                        help='Path to configuration file')
    parser.add_argument('--quick_test', action="store_true",
                    help="Quick test mode on 10% of the dataset with random seed")
    parser.add_argument('--config_override', type=str, nargs='+', default=[],
                    help='Override config values using dot notation. Example: '
                         '--config_override auxiliary_embeddings.position.max_length=512 '
                         'auxiliary_embeddings.time.mode=month '
                         'concetps.LAB.enabled=false')
    return parser.parse_args()

def update_nested_dict(d: dict, key_path: str, value: str) -> None:
    """Update a nested dictionary using dot notation key path.
    
    Args:
        d: Dictionary to update
        key_path: Dot notation path (e.g., 'auxiliary_embeddings.position.max_length')
        value: Value to set (will be converted to appropriate type)
    """
    keys = key_path.split('.')
    current = d
    for key in keys[:-1]:
        current = current[key]
    
    # Convert value to appropriate type
    final_key = keys[-1]
    current_value = current[final_key]
    if isinstance(current_value, bool):
        # Handle boolean values
        new_value = value.lower() == 'true'
    elif isinstance(current_value, int):
        new_value = int(value)
    elif isinstance(current_value, float):
        new_value = float(value)
    else:
        new_value = value
    
    current[final_key] = new_value

def update_config_with_args(config: dict, args: argparse.Namespace) -> dict:
    """Update config dictionary with command line arguments."""
    # Create a copy to avoid modifying the original
    config = config.copy()
    
    # Process each override
    for override in args.config_override:
        try:
            key_path, value = override.split('=')
            update_nested_dict(config, key_path, value)
            log.info(f"Override: {key_path} = {value}")
        except Exception as e:
            log.info(f"Warning: Failed to apply override '{override}': {e}")
    
    return config

def load_admissions_data() -> pl.DataFrame:
    """Load and process the MIMIC-IV admissions data."""
    log.info("Loading admissions data...")
    df_admissions = pl.read_csv(
        'dataset/mimiciv/2.2/hosp/admissions.csv.gz',
        try_parse_dates=True
    )
    
    # Select only the columns we need and ensure correct types
    df_admissions = df_admissions.select([
        'subject_id',
        pl.col('hadm_id').cast(pl.Int64).alias('visit_id'),  # Ensure visit_id is Int64
        'admittime',
        'dischtime'
    ])
    
    return df_admissions

def clean_admission_times(df_admissions: pl.DataFrame) -> pl.DataFrame:
    """
    Clean and validate admission times data:
    1. Remove visits where discharge time is earlier than admit time
    2. Remove overlapping visits (when next visit's admit time is earlier than previous visit's discharge time)
    
    Args:
        df_admissions: DataFrame containing admission data with columns:
            subject_id, visit_id, admittime, dischtime
    Returns:
        Cleaned DataFrame with valid admission times
    """
    # First remove visits where dischtime < admittime
    df_valid = df_admissions.filter(
        pl.col("dischtime") >= pl.col("admittime")
    )
    
    # Sort by subject_id and admittime to process overlapping visits
    df_sorted = df_valid.sort(['subject_id', 'admittime', 'dischtime'])
    
    # For each patient, track visits and remove overlapping ones
    cleaned_data = []
    current_subject = None
    last_disch_time = None
    
    for row in df_sorted.iter_rows(named=True):
        if current_subject != row['subject_id']:
            # New patient, reset tracking
            current_subject = row['subject_id']
            last_disch_time = None
            cleaned_data.append(row)
            last_disch_time = row['dischtime']
            continue
            
        # Check for overlap with previous visit
        if last_disch_time and row['admittime'] < last_disch_time:
            # Skip this visit as it overlaps with previous one
            continue
            
        # No overlap, keep this visit
        cleaned_data.append(row)
        last_disch_time = row['dischtime']
    
    # Convert back to DataFrame
    df_cleaned = pl.DataFrame(cleaned_data)
    
    # Print statistics about removed visits
    total_visits = len(df_admissions)
    invalid_time_visits = total_visits - len(df_valid)
    overlap_visits = len(df_valid) - len(df_cleaned)
    
    log.info("Admission data cleaning statistics:")
    log.info(f"Total visits: {total_visits}")
    log.info(f"Visits removed due to invalid times (discharge < admit): {invalid_time_visits}")
    log.info(f"Visits removed due to overlaps: {overlap_visits}")
    log.info(f"Final valid visits: {len(df_cleaned)}")
    
    return df_cleaned


def get_enabled_concepts(config: dict) -> List[str]:
    """Get list of enabled concept tokens from config."""
    enabled_concepts = []
    cate_keys = ['demographics', 'clinical_codes', 'temporal', 'specials']
    for cate_key in cate_keys:
        if cate_key in config['concetps']:
            for token, token_cfg in config['concetps'][cate_key].items():
                if isinstance(token_cfg, bool) and token_cfg:
                    enabled_concepts.append(token)
                elif token_cfg.get('enabled', True):
                    enabled_concepts.append(token)
    log.info(f"Enabled concepts: {enabled_concepts}")
    log.info('One additional PADDING token is added to the end of the sequence, used for separating patients')
    return enabled_concepts

def get_token_config(config: dict, token: str) -> dict:
    """Get configuration for a specific token."""
    return config['concetps'].get(token, {})

def append_one_row(rows, subject_id, position, concept):
    
    row = {
        "subject_id": subject_id,
        "concept": concept
    }
    
    rows.append(row)
    position += 1
    return rows, position

def get_token_type(code: str, token_type_map: dict) -> int:
    """Get token type based on code prefix using the provided mapping."""
    for prefix, type_id in token_type_map.items():
        if code.startswith(prefix):
            return type_id
    return 0  # Default token type if no match found

def load_patient_year_shifts() -> Dict[int, int]:
    """
    Load patient data and create a mapping of patient IDs to year shifts.
    
    Returns:
        Dict[int, int]: Mapping of subject_id to year_shift (real_year - shift_year)
    """
    df_patients = pl.read_csv('dataset/mimiciv/2.2/hosp/patients.csv.gz')
    patient_year_map = {}
    
    for row in df_patients.iter_rows(named=True):
        subject_id = row['subject_id']
        shift_year = row['anchor_year']
        real_year = int(row['anchor_year_group'].split(' - ')[0])  # Get start year of the group
        year_shift = real_year - shift_year  # Calculate years to shift
        patient_year_map[subject_id] = year_shift
    
    return patient_year_map

def convert_time_to_real(fake_time: datetime, year_shift: int, mode: str = 'year') -> float:
    """
    Convert de-identified time to number of time units (year/month/week) from 2000-01-01.
    
    Args:
        fake_time (datetime): De-identified timestamp
        year_shift (int): Number of years to shift (can be negative)
        mode (str): Time unit to return ('year', 'month', or 'week')
    
    Returns:
        float: Number of time units from 2000-01-01 to the real timestamp
    """
    # First convert to real time by applying year shift
    real_time = fake_time + pd.DateOffset(years=year_shift)
    
    # Define anchor date
    anchor_date = pd.Timestamp('2000-01-01')
    
    # Calculate time difference based on mode
    if mode == 'year':
        time_diff = (real_time - anchor_date).days / 365.25
    elif mode == 'month':
        time_diff = ((real_time.year - anchor_date.year) * 12 + 
                    (real_time.month - anchor_date.month) +
                    (real_time.day - anchor_date.day) / 30.44)  # Average month length
    elif mode == 'week':
        time_diff = (real_time - anchor_date).days / 7
    else:
        raise ValueError("Mode must be one of: 'year', 'month', 'week'")
        
    return round(time_diff, 2)  # Round to 2 decimal places for readability

def generate_patient_sequences(df_events: pl.DataFrame, df_admissions: pl.DataFrame, config: dict, output_dir: str, quick_test: bool, seed: int) -> None:
    """Generate and save patient sequences one at a time."""
    # Load patient year shifts for time conversion
    patient_year_map = load_patient_year_shifts()
    
    df_events = df_events.with_columns([
        pl.col('visit_id').cast(pl.Int64)
    ])
    total_subj_ids = df_events["subject_id"].unique().to_list()
    log.info(f"Number of patients in the dataset: {len(total_subj_ids)}")
    if quick_test:
        # Set random seed before using choice
        np.random.seed(seed)
        unique_patients = np.random.choice(total_subj_ids, size=int(len(total_subj_ids) * 0.1), replace=False)
    else:
        unique_patients = total_subj_ids
    log.info(f"Number of patients we used: {len(unique_patients)}")
    num_filtered_patients = 0
    kept_subject_ids = []
    enabled_concepts = get_enabled_concepts(config)
    max_seq_length = config['sequence']['max_seq_length']
    
    # Create output directories
    raw_dir = os.path.join(output_dir, "raw_sequences")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tokenized_sequences"), exist_ok=True)
    
    # Keep track of all unique concepts for vocabulary creation
    log.info(f"Starting to process patients... (roughly 3-4 hours)")
    all_concepts = set()
    for subject_id in tqdm(unique_patients, desc="Processing patients"):
        position = 0
        df_patient = df_events.filter(pl.col("subject_id") == subject_id)
        
        # Initialize rows for this patient
        patient_rows = []
        # Keep track of concepts used for this patient
        patient_concepts = set()
        
        # Process START_RECORD if enabled
        if 'START_RECORD' in enabled_concepts:
            patient_rows, position = append_one_row(patient_rows, subject_id, position, 'START_RECORD')
            patient_concepts.add('START_RECORD')
        
        # Process demographics in specific order if enabled
        demo_types = ['AGE', 'SEX', 'RACE', 'MARITAL_STATUS', 'YEAR']
        for demo_type in demo_types:
            if demo_type in enabled_concepts:
                demo_df = df_patient.filter(pl.col('code').str.contains(f'{demo_type}_'))
                if len(demo_df) > 0:
                    # Get the first (and should be only) demographic code for this type
                    demo_code = demo_df['code'].to_list()[0]
                    patient_rows, position = append_one_row(patient_rows, subject_id, position, demo_code)
                    patient_concepts.add(demo_code)

        # Get unique visit IDs maintaining first appearance order
        # ! remove time sort, it will destroy the LAB and Q_ tokens order
        df_visits = df_patient.filter(
            (pl.col("sort_order") == 5) & (pl.col("visit_id").is_not_null())).sort(['visit_id'])
        # Get ordered visit IDs from admissions data
        visit_ids = df_admissions.filter(
            (pl.col("subject_id") == subject_id)
        ).sort(['admittime', 'dischtime']).get_column('visit_id').to_list()
        
        last_visit_disch_time = None
        # Before looping visits: get deathtime (or flag) once per subject
        death_df = df_events.filter(
            (pl.col("subject_id") == subject_id) & (pl.col("sort_order") == 6)
        )
        has_death = len(death_df) > 0
        death_time = death_df["time"][0] if has_death else None
        death_inserted = False  # track if we already placed DEATH into some visit
        for visit_id in visit_ids:
            # Get all events for this visit
            df_single_visit = df_visits.filter(pl.col("visit_id").eq(visit_id))
            
            # Get admission times for LOS calculation
            admission_times = df_admissions.filter(
                (pl.col("subject_id") == subject_id) & 
                (pl.col("visit_id").eq(visit_id))
            )
            assert len(admission_times) > 0, f"No admission record found for subject {subject_id}, visit {visit_id}"
            
            admit_time = admission_times["admittime"][0]
            disch_time = admission_times["dischtime"][0]
            
            # ! Adjust event times to be within admission and discharge times
            df_single_visit = df_single_visit.with_columns([
                pl.when(pl.col('time') < admit_time)
                .then(pl.lit(admit_time))
                .when(pl.col('time') > disch_time)
                .then(pl.lit(disch_time))
                .otherwise(pl.col('time'))
                .alias('time')
            ]).sort('time')  # Sort by adjusted times
            
            visit_times = df_single_visit['time'].to_list()
            visit_codes = df_single_visit['code'].to_list()
            
            # Process between_visits if enabled
            if last_visit_disch_time is not None and 'between_visits' in enabled_concepts:
                gap_token = gap_to_separator(last_visit_disch_time, admit_time)
                patient_rows, position = append_one_row(
                    patient_rows, subject_id, position, gap_token)
                patient_concepts.add(gap_token)

            # Process START_VISIT if enabled
            if 'START_VISIT' in enabled_concepts:
                patient_rows, position = append_one_row(
                    patient_rows, subject_id, position, 'START_VISIT')
                patient_concepts.add('START_VISIT')

            if len(visit_codes) > 0:
                # Add first event of the visit
                first_code = visit_codes[0]
                # Only add if the code type is enabled
                code_type = next((prefix for prefix in enabled_concepts if first_code.startswith(prefix)), None)
                if code_type:
                    patient_rows, position = append_one_row(patient_rows, subject_id, position, first_code)
                    patient_concepts.add(first_code)
                
                # Add subsequent events with time gaps
                for i in range(1, len(visit_times)):
                    # Process within_visit if enabled
                    if 'within_visit' in enabled_concepts:
                        time_diff = compute_minutes_delta(visit_times[i-1], visit_times[i])
                        # Only add time gap token if difference exceeds threshold
                        if time_diff >= config['concetps']['temporal']['within_visit']['gap_threshold_min']:
                            gap_token = gap_to_separator(visit_times[i-1], visit_times[i])
                            patient_rows, position = append_one_row(patient_rows, subject_id, position, gap_token)
                            patient_concepts.add(gap_token)
                    
                    # Process medical codes if their type is enabled
                    code = visit_codes[i]
                    code_type = next((prefix for prefix in enabled_concepts if code.startswith(prefix)), None)
                    if code_type:
                        patient_rows, position = append_one_row(
                            patient_rows, subject_id, position, code)
                        patient_concepts.add(code)
            
            # If death happened during THIS admission, emit DEATH **before** END_VISIT
            death_in_this_visit = (
                has_death and
                (death_time is not None) and
                (admit_time <= death_time) and
                (disch_time is None or death_time <= disch_time)
            )
            if death_in_this_visit and 'DEATH' in enabled_concepts:
                # optionally, insert a small within-visit gap token from last event to death_time
                # if you want that fidelity; otherwise, just drop DEATH now:
                patient_rows, position = append_one_row(patient_rows, subject_id, position, 'DEATH')
                patient_concepts.add('DEATH')
                death_inserted = True
                # log.info(f"Death inserted for subject {subject_id}, visit {visit_id}")
                
            # Process END_VISIT if enabled
            if 'END_VISIT' in enabled_concepts:
                patient_rows, position = append_one_row(
                    patient_rows, subject_id, position, "END_VISIT")
                patient_concepts.add('END_VISIT')
            
            last_visit_disch_time = disch_time

        # # Process DEATH if enabled
        # if 'DEATH' in enabled_concepts:
        #     death_df = df_events.filter(
        #         (pl.col("subject_id") == subject_id) &
        #         (pl.col("sort_order") == 6)
        #     )
        #     if len(death_df) > 0:
        #         death_codes = death_df["code"].to_list()
        #         assert len(death_codes) == 1, f"Expected 1 death code for subject {subject_id}, visit {visit_id}"
        #         patient_rows, position = append_one_row(patient_rows, subject_id, position, death_codes[0])
        #         patient_concepts.add(death_codes[0])
        
        # After finishing all visits:
        # Only emit a trailing DEATH if it didn’t occur during any visit (e.g., out of hospital or unknown)
        if 'DEATH' in enabled_concepts and has_death and not death_inserted:
            patient_rows, position = append_one_row(patient_rows, subject_id, position, 'DEATH')
            patient_concepts.add('DEATH')
            # log.info(f"Death inserted in the end for subject {subject_id}")
            
            
        # Process END_RECORD if enabled
        if 'END_RECORD' in enabled_concepts:
            patient_rows, position = append_one_row(patient_rows, subject_id, position, "END_RECORD")
            patient_concepts.add('END_RECORD')

        # After processing all events for the patient
        # sequence_length = len(patient_rows)
        if config['sequence']['packing'] == 'dense':
            patient_rows, position = append_one_row(patient_rows, subject_id, position, "PADDING")
            patient_concepts.add("PADDING")
            df_patient_sequences = pl.DataFrame(patient_rows)
            df_patient_sequences.write_parquet(os.path.join(raw_dir, f"{subject_id}.parquet"))
            
            # Add all concepts from this patient to the global set
            all_concepts.update(patient_concepts)
            kept_subject_ids.append(subject_id)
        else:
            raise ValueError(f"Unknown sequence packing mode: {config['sequence']['packing']}")
            # if sequence_length <= max_seq_length:
            #     # Add padding tokens if needed
            #     while position < max_seq_length:
            #         patient_rows, position = append_one_row(
            #             patient_rows, subject_id, position, "PADDING")
            #     patient_concepts.add("PADDING")
                
            #     # Save this patient's sequences immediately
            #     df_patient_sequences = pl.DataFrame(patient_rows)
            #     df_patient_sequences.write_parquet(os.path.join(raw_dir, f"{subject_id}.parquet"))
                
            #     # Add all concepts from this patient to the global set
            #     all_concepts.update(patient_concepts)
            #     kept_subject_ids.append(subject_id)
            # else:
            #     num_filtered_patients += 1
            
    log.info(f"Number of filtered patients: {num_filtered_patients} due to exceeding length {max_seq_length}")
    # Save vocabulary
    vocab = create_vocabulary_from_concepts(all_concepts, output_dir)
    
    # Save kept subject IDs to CSV
    kept_ids_df = pl.DataFrame({"subject_id": kept_subject_ids})
    kept_ids_df.write_csv(os.path.join(output_dir, 'cohort_stat', "kept_subject_ids.csv"))
    return vocab, kept_subject_ids, config

def create_vocabulary_from_concepts(concepts: set, output_dir: str) -> dict:
    """Create vocabulary from collected concepts."""
    unique_tokens = sorted(list(concepts))
    vocab_df = pl.DataFrame({
        'token_id': range(len(unique_tokens)),
        'concept_code': unique_tokens
    })
    tokenizer_dir = os.path.join(output_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    output_path = os.path.join(tokenizer_dir, 'vocab.csv')
    log.info(f"Saving vocabulary to {output_path}")
    log.info(f"Total unique tokens: {len(unique_tokens)}")
    vocab_df.write_csv(output_path)
    
    return dict(zip(vocab_df['concept_code'], vocab_df['token_id']))

def convert_sequences_to_indices(sequences: List[dict], vocab: dict) -> List[dict]:
    """
    Convert token sequences to index sequences using vocabulary.
    
    Args:
        sequences: List of dictionaries containing patient event sequences
        vocab: Dictionary mapping tokens to indices
    Returns:
        List of dictionaries with concepts converted to indices
    """
    tokenized_sequences = []
    for event in sequences:
        # Create a copy of the event and update the concept to its index
        tokenized_event = event.copy()
        tokenized_event['concept'] = vocab[event['concept']]
        tokenized_sequences.append(tokenized_event)
    
    return tokenized_sequences

def tokenize_and_collect_stats(raw_dir: str, tokenized_dir: str, vocab: dict) -> None:
    """
    Tokenize sequences and collect statistics.
    
    Args:
        raw_dir: Directory containing raw sequence files
        tokenized_dir: Directory to save tokenized sequences
        vocab: Vocabulary mapping from tokens to indices
    """
    os.makedirs(tokenized_dir, exist_ok=True)
    
    # Initialize counters for statistics
    total_patients = 0
    total_visits = 0
    total_non_padding_records = 0
    
    # Process each patient's sequence
    for filename in tqdm(os.listdir(raw_dir), desc="Tokenizing sequences"):
        if filename.endswith('.parquet'):
            # Load patient sequence
            df_sequence = pl.read_parquet(os.path.join(raw_dir, filename))
            
            # Update statistics
            total_patients += 1
            total_visits += df_sequence.filter(pl.col('concept')=='START_VISIT').shape[0]
            total_non_padding_records += df_sequence.filter(pl.col('concept')!='PADDING').shape[0]
            
            # Convert concepts to list for processing
            concepts = df_sequence['concept'].to_list()
            # Map concepts to indices
            concept_indices = [vocab[c] for c in concepts]
            
            # Create new DataFrame with mapped indices
            df_tokenized = df_sequence.with_columns([
                pl.Series(name='concept', values=concept_indices)
            ])
            
            # Save tokenized sequence
            df_tokenized.write_parquet(os.path.join(tokenized_dir, filename))
    
    # Print final statistics
    log.info("After tokenization:")
    log.info(f"Patients: {total_patients}")
    log.info(f"Visits: {total_visits}")
    log.info(f"Total records (excluding padding): {total_non_padding_records}")

def split_patient_ids(subject_ids: List[int], val_ratio: float, test_ratio: float, seed: int) -> Dict[str, List[int]]:
    """
    Split patient IDs into train, validation and test sets.
    
    Args:
        subject_ids: List of patient IDs
        val_ratio: Ratio of validation set size to total
        test_ratio: Ratio of test set size to total
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing train, val, and test patient IDs
    """
    
    # Shuffle patient IDs
    subject_ids = np.array(subject_ids)
    # Set random seed before shuffling
    np.random.seed(seed)
    np.random.shuffle(subject_ids)
    
    # Calculate split points
    n_total = len(subject_ids)
    n_test = int(n_total * test_ratio)
    n_train_val = n_total - n_test
    n_val = int(n_train_val * val_ratio)
    n_train = n_train_val - n_val
    
    # Split into train/val/test
    train_ids = subject_ids[:n_train]
    val_ids = subject_ids[n_train:n_train + n_val]
    test_ids = subject_ids[n_train + n_val:]
    
    # Save splits to CSV
    splits = {
        'train': train_ids.tolist(),
        'val': val_ids.tolist(),
        'test': test_ids.tolist()
    }
    
    return splits

def save_splits(splits: Dict[str, List[int]], output_dir: str) -> None:
    """Save train/val/test splits to CSV files."""
    for split_name, ids in splits.items():
        df = pl.DataFrame({"subject_id": ids})
        df.write_csv(os.path.join(output_dir, f"{split_name}_subject_ids.csv"))

def calculate_split_statistics(df_events: pl.DataFrame, split_name: str, split_ids: List[int], output_dir: str) -> None:
    """
    Calculate and log.info detailed statistics for a data split.
    """
    # Get statistics from raw sequences (including time gaps)
    if split_name == "total":
        # For total stats, sum up all split files
        n_records = 0
        for split in ['train', 'val', 'test']:
            file_path = os.path.join(output_dir, "raw_sequences", f"{split}.parquet")
            if os.path.exists(file_path):
                df = pl.read_parquet(file_path)
                n_records += len(df)
    else:
        # For individual splits, read the specific split file
        file_path = os.path.join(output_dir, "raw_sequences", f"{split_name}.parquet")
        if os.path.exists(file_path):
            df = pl.read_parquet(file_path)
            n_records = len(df)
        else:
            n_records = 0
    
    # Get statistics from original events data
    split_data = df_events.filter(pl.col('subject_id').is_in(split_ids))
    n_patients = len(split_ids)
    n_visits = split_data.filter(pl.col('visit_id').is_not_null()).get_column('visit_id').n_unique()
    log.info(f"Number of visits: {n_visits}")
    
    
    # Age statistics
    age_data = split_data.filter(pl.col('sort_order') == 0)
    if len(age_data) > 0:
        age_values = age_data.get_column('numeric_value').cast(pl.Float64)
        age_mean = age_values.mean()
        age_std = age_values.std()
        log.info("Age Statistics:")
        log.info(f"Mean age: {age_mean:.2f}")
        log.info(f"Std age: {age_std:.2f}")
    else:
        log.info("Age Statistics: No age data available")
    
    # Gender distribution
    gender_data = split_data.filter(pl.col('sort_order') == 1)
    if len(gender_data) > 0:
        gender_dist = (gender_data
                    .group_by('code')
                    .agg(pl.count())
                    .sort('count', descending=True))
        log.info("Gender Distribution:")
        for row in gender_dist.iter_rows():
            code, count = row
            percentage = (count/n_patients) * 100
            log.info(f"{code}: {count} ({percentage:.1f}%)")
    else:
        log.info("Gender Distribution: No gender data available")
    
    # Race distribution
    race_data = split_data.filter(pl.col('sort_order') == 2)
    if len(race_data) > 0:
        race_dist = (race_data
                  .group_by('code')
                  .agg(pl.count())
                  .sort('count', descending=True))
        log.info("Race Distribution:")
        for row in race_dist.iter_rows():
            code, count = row
            percentage = (count/n_patients) * 100
            log.info(f"{code}: {count} ({percentage:.1f}%)")
    else:
        log.info("Race Distribution: No race data available")
    
    # Marital status distribution
    marital_data = split_data.filter(pl.col('sort_order') == 3)
    if len(marital_data) > 0:
        marital_dist = (marital_data
                     .group_by('code')
                     .agg(pl.count())
                     .sort('count', descending=True))
        log.info("Marital Status Distribution:")
        for row in marital_dist.iter_rows():
            code, count = row
            percentage = (count/n_patients) * 100
            log.info(f"{code}: {count} ({percentage:.1f}%)")
    else:
        log.info("Marital Status Distribution: No marital status data available")

def organize_train_val_test_splits(kept_subject_ids: List[int], 
                                vocab: dict,
                                df_events: pl.DataFrame,
                                args: argparse.Namespace,
                                config: dict) -> None:
    """
    Organize patient data into train/val/test splits and process each split.
    """
    
    # Split patients into train/val/test sets
    splits = split_patient_ids(kept_subject_ids, args.val_ratio, args.test_ratio, args.seed)
    
    # Save splits to CSV files
    save_splits(splits, os.path.join(args.LOG_DIR, "raw_sequences"))
    
    # Print total statistics
    log.info("TOTAL STATISTICS:")
    calculate_split_statistics(df_events, "total", kept_subject_ids, args.LOG_DIR)
    
    # Process each split
    for split_name, split_ids in splits.items():
        log.info(f"Processing {split_name} split...")
        
        # Collect all patient data for this split
        split_dfs = []
        for subject_id in tqdm(split_ids, desc=f"Merging {split_name} sequences"):
            patient_file = os.path.join(args.LOG_DIR, "raw_sequences", f"{subject_id}.parquet")
            if os.path.exists(patient_file):
                # Read the parquet file
                df = pl.read_parquet(patient_file)
                
                # # Cast subject_id and position (always present)
                # cast_exprs = [
                #     pl.col('subject_id').cast(pl.Int64),
                #     pl.col('position').cast(pl.Int64),
                # ]

                # # Apply the casts
                # df = df.with_columns(cast_exprs)
                split_dfs.append(df)
                # Remove individual file after reading
                os.remove(patient_file)
        
        # Concatenate all patient data and save as a single file
        if split_dfs:
            log.info(f"Concatenating {len(split_dfs)} patient sequences for {split_name} split...")
            split_df = pl.concat(split_dfs)
            split_output = os.path.join(args.LOG_DIR, "raw_sequences", f"{split_name}.parquet")
            log.info(f"Saving {split_name} sequences to {split_output}")
            split_df.write_parquet(split_output)
            if args.quick_test:
                split_df.write_csv(os.path.join(args.LOG_DIR, "raw_sequences", f"{split_name}.csv"))
            
            # Create tokenized version
            log.info(f"Creating tokenized sequences for {split_name} split...")
            # Convert concepts to indices using the vocabulary
            concept_list = split_df['concept'].to_list()
            concept_indices = [vocab[c] for c in concept_list]
            
            # Create new DataFrame with tokenized concepts
            split_df = split_df.with_columns([
                pl.Series(name='concept', values=concept_indices).cast(pl.Int64)  # Ensure Int64 type
            ])
            
            # Prepare aggregation expressions based on config
            agg_exprs = [
                pl.col("concept").alias("concept_token_ids"),
                pl.count().alias("seq_length"),
            ]
            
            # Group by subject_id and aggregate
            tokenized_df = split_df.group_by("subject_id").agg(agg_exprs)
            
            tokenized_output = os.path.join(args.LOG_DIR, "tokenized_sequences", f"{split_name}.parquet")
            log.info(f"Saving tokenized {split_name} sequences to {tokenized_output}")
            tokenized_df.write_parquet(tokenized_output)
        
        # Print split statistics
        log.info(f"{split_name.upper()} SPLIT:")
        log.info(f"Number of patients: {len(split_ids)} ({len(split_ids)/len(kept_subject_ids)*100:.1f}%)")
        log.info(f"Number of tokens: {tokenized_df['seq_length'].sum()}")
        calculate_split_statistics(df_events, split_name, split_ids, args.LOG_DIR)

def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{args.LOG_DIR}/logs/token_processor_complete.log"),  # Add a file
        ]
    )
    # Load configuration
    config = load_config(args.config)
    
    # Update config with command line arguments
    config = update_config_with_args(config, args)
    # Load data
    log.info(f"Loading data from {args.LOG_DIR}/cohort_stat/mimiciv_2.2_meds_processed.parquet")
    df_events = pl.read_parquet(f"{args.LOG_DIR}/cohort_stat/mimiciv_2.2_meds_processed.parquet")
    log.info(f"Patients: {df_events.select('subject_id').unique().shape[0]}")
    log.info(f"Visits (non-null; null visit_id for demographics): {df_events.filter(pl.col('visit_id').is_not_null()).select('visit_id').unique().shape[0]}")
    log.info(f"Total records: {df_events.shape[0]}")
    
    # Load admissions data
    df_admissions = load_admissions_data()
    
    # Clean admission times data
    df_admissions = clean_admission_times(df_admissions)
    
    # Generate and save sequences, get vocabulary and updated config
    vocab, kept_subject_ids, config = generate_patient_sequences(df_events, df_admissions, config, args.LOG_DIR, args.quick_test, args.seed)
    
    # Organize and process train/val/test splits
    organize_train_val_test_splits(
        kept_subject_ids=kept_subject_ids,
        vocab=vocab,
        df_events=df_events,
        args=args,
        config=config
    )
    
    # Save the complete modified config
    modified_config_path = os.path.join(args.LOG_DIR, "config.yaml")
    with open(modified_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    log.info(f"Saved modified config to: {modified_config_path}")

if __name__ == "__main__":
    main() 