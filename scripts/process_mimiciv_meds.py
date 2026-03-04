import polars as pl
import glob
import os
import argparse
import warnings
import json
import logging
from scripts.utils import get_dataset_statistics, save_all_plus_one_subject, remove_unused_columns
# import pdb; pdb.set_trace()
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Process MIMIC-IV v2.2 meds data.')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--LOG_DIR', type=str, default='output/del')
    parser.add_argument('--icd10cm_top_n', type=int, default=5000,
                       help='Number of top ICD10CM codes to keep')
    parser.add_argument('--icd10pcs_top_n', type=int, default=3000,
                       help='Number of top ICD10PCS codes to keep')
    parser.add_argument('--atc_top_n', type=int, default=400,
                       help='Number of top ATC codes to keep')
    parser.add_argument('--lab_top_n', type=int, default=200,
                       help='Number of top lab test events to keep')
    return parser.parse_args()

def load_data():
    """Load and combine all parquet files from the dataset directory."""
    data_dir = f"dataset/mimiciv/2.2_meds/data"
    parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    dfs = [pl.read_parquet(f) for f in parquet_files]
    df_combined = pl.concat(dfs)
    
    log.info(f'Loaded data from {data_dir}')
    return df_combined

def add_sort_order(df_events: pl.DataFrame) -> pl.DataFrame:
    """Add sort_order column for organizing records."""
    return df_events.with_columns([
        pl.when(pl.col('code').str.contains('MEDS_BIRTH'))
        .then(0)  # BIRTH stays first for reference
        .when(pl.col('code').str.contains('MIMIC_IV_Gender'))
        .then(1)  # SEX
        .when(pl.col('code').str.contains('MIMIC_IV_Race'))
        .then(2)  # RACE
        .when(pl.col('code').str.contains('MIMIC_IV_Marital_Status'))
        .then(3)  # MARITAL_STATUS
        .when(pl.col('code') == 'AGE')
        .then(0)  # AGE (will be added later)
        .when(pl.col('code').str.contains('Anchor_Year'))
        .then(4)  # AGE (will be added later)
        .when(pl.col('code').str.contains('DEATH'))
        .then(6)  # DEATH
        .otherwise(5)  # medical codes
        .alias('sort_order')
    ]).sort(['subject_id', 'sort_order', 'time'])

def calculate_and_add_age(df_events: pl.DataFrame) -> pl.DataFrame:
    """Calculate age and add AGE records using sort_order == 5 as reference time."""
    
    # Extract birth time for each patient
    num_subjects = df_events.select('subject_id').unique().shape[0]
    birth_times = (
        df_events
        .filter(pl.col('code').str.contains('MEDS_BIRTH'))
        .select(['subject_id', 'time'])
        .rename({'time': 'birth_time'})
    )
    assert birth_times.shape[0] == num_subjects, "Number of subjects with birth records does not match"
    
    # Use the earliest time with sort_order == 5 as the reference time
    reference_times = (
        df_events
        .filter(pl.col('sort_order') == 5)
        .group_by('subject_id')
        .agg([
            pl.col('time').min().alias('reference_time'),
            pl.col('visit_id').first().alias('reference_visit_id')
        ])
    )
    assert reference_times.shape[
        0] == num_subjects, f"Number of subjects with reference time does not match: {num_subjects} != {reference_times.shape[0]}"
    
    # Join both tables and compute age
    age_base = (
        birth_times
        .join(reference_times, on='subject_id', how='inner')
        .with_columns([
            ((pl.col('reference_time') - pl.col('birth_time')).dt.total_days() / 365.25)
            .round(0)
            .cast(pl.Int64)
            .alias('age_years')
        ])
    )

    # Bin the age into discrete AGE categories
    def bin_age(age_col):
        return (
            pl.when(age_col <= 5).then(pl.lit('AGE_0_5_years'))
            .when(age_col <= 10).then(pl.lit('AGE_5_10_years'))
            .when(age_col <= 15).then(pl.lit('AGE_10_15_years'))
            .when(age_col <= 20).then(pl.lit('AGE_15_20_years'))
            .when(age_col <= 25).then(pl.lit('AGE_20_25_years'))
            .when(age_col <= 30).then(pl.lit('AGE_25_30_years'))
            .when(age_col <= 35).then(pl.lit('AGE_30_35_years'))
            .when(age_col <= 40).then(pl.lit('AGE_35_40_years'))
            .when(age_col <= 45).then(pl.lit('AGE_40_45_years'))
            .when(age_col <= 50).then(pl.lit('AGE_45_50_years'))
            .when(age_col <= 55).then(pl.lit('AGE_50_55_years'))
            .when(age_col <= 60).then(pl.lit('AGE_55_60_years'))
            .when(age_col <= 65).then(pl.lit('AGE_60_65_years'))
            .when(age_col <= 70).then(pl.lit('AGE_65_70_years'))
            .when(age_col <= 75).then(pl.lit('AGE_70_75_years'))
            .when(age_col <= 80).then(pl.lit('AGE_75_80_years'))
            .when(age_col <= 85).then(pl.lit('AGE_80_85_years'))
            .when(age_col <= 90).then(pl.lit('AGE_85_90_years'))
            .when(age_col <= 95).then(pl.lit('AGE_90_95_years'))
            .when(age_col <= 100).then(pl.lit('AGE_95_100_years'))
            .otherwise(pl.lit('AGE_100_plus_years'))
        )

    # Prepare AGE records with all columns from df_events
    age_records = (
        age_base
        .with_columns([
            bin_age(pl.col('age_years')).alias('code'),
            pl.col('reference_time').alias('time'),
            pl.lit(0).alias('sort_order'),
            pl.col('age_years').cast(df_events.schema['numeric_value']).alias('numeric_value'),  # Store actual age
            pl.lit('years').cast(df_events.schema['unit']).alias('unit'),
            pl.lit(None).cast(df_events.schema['text_value']).alias('text_value'),
            pl.lit(None).cast(df_events.schema['seq_num']).alias('seq_num'),
            pl.lit(None).alias('table'),
            pl.lit(None).cast(df_events.schema['visit_id']).alias('visit_id')
        ])
        .with_columns([
            pl.col('subject_id')
        ])
        .select(df_events.columns)  # ensure all columns are in the same order
    )

    # Concatenate new AGE records with original events and sort
    df_augmented = pl.concat([df_events, age_records]).sort(['subject_id', 'sort_order', 'time'])

    return df_augmented

def filter_null_visit(df_events: pl.DataFrame) -> pl.DataFrame:
    """Process and deduplicate demographic information."""
    # Handle demographics (keep first occurrence)
    df_demographics = (
        df_events
        .filter(pl.col('code').str.contains('BIRTH|Gender|Race|Marital_Status|DEATH|Anchor_Year'))
        .group_by(['subject_id', 'code'])
        .agg(pl.col('*').first())
        .select(
            'subject_id',
            'time',
            'code',
            pl.all().exclude(['subject_id', 'time', 'code'])
        )
    )
    
    # Remove records with null visit_id
    df_other = (
        df_events
        .filter(~pl.col('code').str.contains('BIRTH|Gender|Race|Marital_Status|DEATH|Anchor_Year'))
        .filter(pl.col('visit_id').is_not_null())
    )
    
    # Get list of patients who have at least one record with valid visit_id
    patients_with_valid_visits = df_other.select('subject_id').unique()
    
    # Filter demographics to keep only patients with valid visits
    df_demographics_filtered = df_demographics.filter(
        pl.col('subject_id').is_in(patients_with_valid_visits.get_column('subject_id'))
    )
    
    # Print statistics about removed patients
    total_patients = df_demographics.select('subject_id').unique().shape[0]
    remaining_patients = patients_with_valid_visits.shape[0]
    removed_patients = total_patients - remaining_patients
    
    log.info(f"Patient Filtering Statistics:")
    log.info(f"Total patients before filtering: {total_patients}")
    log.info(f"Patients with valid visits: {remaining_patients}")
    log.info(f"Patients removed (no valid visits): {removed_patients}")
    
    return pl.concat([df_demographics_filtered, df_other]).sort(['subject_id', 'sort_order', 'time'])

def standardize_birth_death_gender(df_events: pl.DataFrame) -> pl.DataFrame:
    """Standardize various code formats."""
    # Standardize birth and death codes
    df_events = df_events.with_columns([
        pl.when(pl.col('code') == 'MEDS_BIRTH')
        .then(pl.lit('BIRTH'))
        .when(pl.col('code') == 'MEDS_DEATH')
        .then(pl.lit('DEATH'))
        .otherwise(pl.col('code'))
        .alias('code')
    ])
    
    # Standardize gender codes
    df_events = df_events.with_columns([
        pl.when(pl.col('code') == 'MIMIC_IV_Gender/M')
        .then(pl.lit('SEX_M'))
        .when(pl.col('code') == 'MIMIC_IV_Gender/F')
        .then(pl.lit('SEX_F'))
        .otherwise(pl.col('code'))
        .alias('code')
    ])
    
    return df_events

def standardize_race(df_events: pl.DataFrame) -> pl.DataFrame:
    """Standardize race codes."""
    race_unknown = ["UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER"]
    race_minor = [
        "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER",
        "AMERICAN INDIAN/ALASKA NATIVE",
        "MULTIPLE RACE/ETHNICITY",
        "Race/OTHER"
    ]
    
    # First set visit_id to None for all race records
    df_events = df_events.with_columns([
        pl.when(pl.col('code').str.starts_with('MIMIC_IV_Race'))
        .then(pl.lit(None).cast(df_events.schema['visit_id']))
        .otherwise(pl.col('visit_id'))
        .alias('visit_id')
    ])
    
    # Then standardize the race codes
    return df_events.with_columns([
        pl.when(pl.col('code').str.starts_with('MIMIC_IV_Race/') & pl.col('code').str.contains('|'.join(race_unknown)))
        .then(pl.lit('RACE_UNKNOWN'))
        .when(pl.col('code').str.starts_with('MIMIC_IV_Race/') & pl.col('code').str.contains('|'.join(race_minor)))
        .then(pl.lit('RACE_OTHER'))
        .when(pl.col('code').str.contains('SOUTH AMERICAN'))
        .then(pl.lit('RACE_HISPANIC'))
        .when(pl.col('code').str.contains('PORTUGUESE|WHITE'))
        .then(pl.lit('RACE_WHITE'))
        .when(pl.col('code').str.contains('HISPANIC'))
        .then(pl.lit('RACE_HISPANIC'))
        .when(pl.col('code').str.contains('BLACK'))
        .then(pl.lit('RACE_BLACK'))
        .when(pl.col('code').str.contains('ASIAN'))
        .then(pl.lit('RACE_ASIAN'))
        .otherwise(pl.col('code'))
        .alias('code')
    ])

def standardize_marital_status(df_events: pl.DataFrame) -> pl.DataFrame:
    """Standardize marital status codes."""
        
    # First set visit_id to None for all race records
    df_events = df_events.with_columns([
        pl.when(pl.col('code').str.starts_with('MIMIC_IV_Marital_Status'))
        .then(pl.lit(None).cast(df_events.schema['visit_id']))
        .otherwise(pl.col('visit_id'))
        .alias('visit_id')
    ])
    
    return df_events.with_columns([
        pl.when(pl.col('code') == 'MIMIC_IV_Marital_Status/WIDOWED')
        .then(pl.lit('MARITAL_STATUS_WIDOWED'))
        .when(pl.col('code') == 'MIMIC_IV_Marital_Status/DIVORCED')
        .then(pl.lit('MARITAL_STATUS_DIVORCED'))
        .when(pl.col('code') == 'MIMIC_IV_Marital_Status/MARRIED')
        .then(pl.lit('MARITAL_STATUS_MARRIED'))
        .when(pl.col('code') == 'MIMIC_IV_Marital_Status/SINGLE')
        .then(pl.lit('MARITAL_STATUS_SINGLE'))
        .when(pl.col('code') == 'MIMIC_IV_Marital_Status/UNKNOWN')
        .then(pl.lit('MARITAL_STATUS_UNKNOWN'))
        .otherwise(pl.col('code'))
        .alias('code'),
        
        pl.when(pl.col('code').str.starts_with('MARITAL_STATUS_'))
        .then(pl.lit(None).cast(df_events.schema['visit_id']))
        .otherwise(pl.col('visit_id'))
        .alias('visit_id')
    ])

def process_diagnoses(df_events: pl.DataFrame) -> pl.DataFrame:
    """Convert ICD9CM codes to ICD10CM codes."""
    # Load ICD mapping
    icd_mapping = pl.read_csv(
        "scripts/maps/icd_cm_9_to_10_mapping.csv.gz",
        use_pyarrow=True
    )
    # Load supplementary mapping and add flags column
    icd_mapping_supp = pl.read_csv(
        "scripts/maps/icd_cm_9_to_10_mapping_supplementary.csv",
        use_pyarrow=True,
    ).with_columns([
        # Add flags column with null values
        pl.lit(None).cast(pl.Int64).alias('flags')
    ])

    # Concatenate the two mappings and ensure one-to-one mapping by taking first ICD10 code
    icd_mapping = (
        pl.concat([icd_mapping, icd_mapping_supp])
        .group_by('icd_9')
        .agg(pl.col('icd_10').first().alias('icd_10'))
    )
    
    # Print mapping statistics
    total_icd9_codes = df_events.filter(pl.col('code').str.starts_with('ICD9CM_')).shape[0]
    log.info(f"ICD9 to ICD10 Mapping Statistics:")
    log.info(f"Total ICD9CM records before mapping: {total_icd9_codes}")
    
    # Create temporary column for mapping
    df_events = df_events.with_columns([
        pl.when(pl.col('code').str.starts_with('ICD9CM_'))
        .then(pl.col('code').str.replace('ICD9CM_', ''))
        .otherwise(pl.col('code'))
        .alias('icd9_code_temp')
    ])
    
    # Join with mapping table
    df_events = df_events.join(
        icd_mapping.select(['icd_9', 'icd_10']),
        left_on='icd9_code_temp',
        right_on='icd_9',
        how='left'
    )
    
    # Convert codes
    df_events = df_events.with_columns([
        pl.when(pl.col('code').str.starts_with('ICD9CM_'))
        .then(
            pl.when(pl.col('icd_10').is_not_null())
            .then(pl.concat_str([pl.lit('ICD10CM_'), pl.col('icd_10')]))
            .otherwise(pl.col('code'))
        )
        .otherwise(pl.col('code'))
        .alias('code')
    ])
    
    # Print mapping results
    total_mapped = df_events.filter(pl.col('code').str.starts_with('ICD10CM_')).shape[0]
    log.info(f"Total ICD10CM records after mapping: {total_mapped}")
    log.info(f"Difference in records: {total_mapped - total_icd9_codes}")
    
    # Clean up temporary columns
    return df_events.drop(['icd9_code_temp', 'icd_10'])


def process_procedures(df_events: pl.DataFrame) -> pl.DataFrame:
    """Convert ICD9PCS codes to ICD10PCS codes."""
    # Load ICD mapping with explicit string type for icd_9
    icd_mapping = pl.read_csv(
        "scripts/maps/icd_pcs_9_to_10_mapping.csv.gz",
        use_pyarrow=True,
        dtypes={'icd_9': pl.Utf8}  # Force icd_9 to be read as string
    ).with_columns([
        # Ensure 4-digit format with leading zeros
        pl.col('icd_9').str.zfill(4).alias('icd_9'),
        pl.col('icd_10').cast(pl.Utf8).str.strip_chars().alias('icd_10'),
        pl.lit(None).cast(pl.Utf8).alias('flags'),
        pl.lit(None).cast(pl.Utf8).alias('year')
    ])
    
    # For codes with multiple mappings, take the first one
    icd_mapping = icd_mapping.group_by('icd_9').agg(
        pl.col('icd_10').first().alias('icd_10'),
        pl.col('flags').first().alias('flags'),
        pl.col('year').first().alias('year')
    )
    
    # Load supplementary mapping and add flags column
    icd_mapping_supp = pl.read_csv(
        "scripts/maps/icd_pcs_9_to_10_mapping_supplementary.csv",
        use_pyarrow=True,
        dtypes={'icd_9': pl.Utf8}  # Force icd_9 to be read as string
    ).with_columns([
        pl.col('icd_9').str.zfill(4).alias('icd_9'),  # Ensure 4-digit format
        pl.col('icd_10').cast(pl.Utf8).str.strip_chars().alias('icd_10'),
        pl.lit(None).cast(pl.Utf8).alias('flags'),
        pl.lit(None).cast(pl.Utf8).alias('year')
    ])

    # Combine mappings
    icd_mapping = pl.concat([icd_mapping, icd_mapping_supp])

    # Step 1: Normalize `code` column by padding ICD9PCS codes to 4 digits
    df_events = df_events.with_columns([
        pl.when(pl.col('code').str.starts_with('ICD9PCS_'))
        .then(
            'ICD9PCS_' +
            pl.col('code').str.replace('ICD9PCS_', '').str.zfill(4)
        )
        .otherwise(pl.col('code'))
        .alias('code')
    ])

    # Step 2: Extract the 4-digit ICD9PCS code into a new temporary column
    df_events = df_events.with_columns([
        pl.when(pl.col('code').str.starts_with('ICD9PCS_'))
        .then(pl.col('code').str.extract(r"ICD9PCS_(\d{4})", 1))
        .otherwise(None)
        .alias('icd9_code_temp')
    ])
    
    # Join with mapping table
    df_events = df_events.join(
        icd_mapping.select(['icd_9', 'icd_10']),
        left_on='icd9_code_temp',
        right_on='icd_9',
        how='left'
    )

    # Convert codes
    df_events = df_events.with_columns([
        pl.when(pl.col('code').str.starts_with('ICD9PCS_'))
        .then(
            pl.when(pl.col('icd_10').is_not_null())
            .then(pl.concat_str([pl.lit('ICD10PCS_'), pl.col('icd_10')]))
            .otherwise(pl.col('code'))
        )
        .otherwise(pl.col('code'))
        .alias('code')
    ])

    # Clean up temporary columns
    return df_events.drop(['icd9_code_temp', 'icd_10'])

def process_medications(df_events: pl.DataFrame) -> pl.DataFrame:
    """Convert drug names to ATC codes using the mapping file."""
    # Load drug to ATC mapping and handle duplicates by taking first ATC code for each drug
    drug_mapping = (
        pl.read_csv(
            "scripts/maps/mimic_drug_to_atc.csv.gz",
            use_pyarrow=True
        )
        # Convert drug names to lowercase for case-insensitive matching
        .with_columns([
            pl.col('drug').str.to_lowercase().alias('drug_lower')
        ])
        .group_by('drug_lower')
        .agg(
            pl.col('atc_code').first().alias('atc_code'),
            pl.col('drug').first().alias('drug_original')  # Keep original drug name for reference
        )
    )

    # Create temporary column with extracted drug names in lowercase
    df_events = df_events.with_columns([
        pl.when(pl.col('code').str.starts_with('MEDICATION_'))
        .then(pl.col('code').str.replace('MEDICATION_', '').str.to_lowercase())
        .otherwise(None)
        .alias('drug_name_temp')
    ])

    # Join with mapping table using lowercase names
    df_events = df_events.join(
        drug_mapping.select(['drug_lower', 'atc_code']),
        left_on='drug_name_temp',
        right_on='drug_lower',
        how='left'
    )

    # Print statistics about unmapped medications
    total_rows = df_events.filter(pl.col('code').str.starts_with('MEDICATION_')).shape[0]
    unmapped_meds = df_events.filter(
        pl.col('code').str.starts_with('MEDICATION_') & 
        pl.col('atc_code').is_null()
    )
    unmapped_count = unmapped_meds.shape[0]
    
    log.info("Medication mapping statistics:")
    log.info(f"Total rows of medication codes in dataset: {total_rows}")
    log.info(f"Rows with unmapped medications: {unmapped_count}")
    if total_rows > 0:
        log.info(f"Percentage of rows to be removed: {(unmapped_count/total_rows*100):.2f}%")

    # Convert codes and filter out unmapped medications
    df_events = (
        df_events
        .with_columns([
            pl.when(pl.col('code').str.starts_with('MEDICATION_'))
            .then(
                pl.when(pl.col('atc_code').is_not_null())
                .then(pl.concat_str([pl.lit('ATC_'), pl.col('atc_code')]))
                .otherwise(pl.col('code'))
            )
            .otherwise(pl.col('code'))
            .alias('code')
        ])
        # Filter out medications that couldn't be mapped to ATC
        .filter(
            ~(pl.col('code').str.starts_with('MEDICATION_'))
        )
    )

    # Clean up temporary columns
    return df_events.drop(['drug_name_temp', 'atc_code'])

def process_lab_tests(df_events: pl.DataFrame, args: argparse.Namespace) -> pl.DataFrame:
    """Convert LAB_TEST_xxx codes to meaningful names using the lab items mapping."""
    # Load lab items mapping
    lab_mapping = (
        pl.read_csv(
            f"dataset/mimiciv/2.2/hosp/d_labitems.csv.gz",
            use_pyarrow=True
        )
        .with_columns([
            # Remove double quotes from labels and clean up any extra spaces
            pl.col('label').str.replace('"', '').str.strip_chars().alias('label')
        ])
    )
    
    # Print initial statistics about lab tests with null values
    total_labs = df_events.filter(pl.col('code').str.starts_with('LAB_TEST_')).shape[0]
    # ! add null or negative numeric_value lab test values
    null_or_negative_value_labs = df_events.filter(
        pl.col('code').str.starts_with('LAB_TEST_') & 
        (pl.col('numeric_value').is_null() | (pl.col("numeric_value") < 0))
    ).shape[0]
    
    log.info("Lab test null value statistics (before filtering):")
    log.info(f"Total rows of lab test codes: {total_labs}")
    log.info(f"Rows with null or negative numeric_value: {null_or_negative_value_labs}")
    if total_labs > 0:
        log.info(f"Percentage of rows to be removed due to null or negative values: {(null_or_negative_value_labs/total_labs*100):.2f}%")
    
    # Filter out lab tests with null or negative numeric_value
    df_events = df_events.filter(
        ~(pl.col('code').str.starts_with('LAB_TEST_')) | 
        (pl.col('code').str.starts_with('LAB_TEST_') & (pl.col('numeric_value').is_not_null() & (pl.col("numeric_value") >= 0)))
    )
    
    # Create temporary column with extracted lab test IDs
    df_events = df_events.with_columns([
        pl.when(pl.col('code').str.starts_with('LAB_TEST_'))
        .then(
            pl.col('code')
            .str.extract(r'LAB_TEST_(\d+)', 1)  # Extract only the numeric part
            .cast(pl.Int64)
        )
        .otherwise(None)
        .alias('lab_id_temp')
    ])
    
    # Join with mapping table
    df_events = df_events.join(
        lab_mapping.select(['itemid', 'label']),
        left_on='lab_id_temp',
        right_on='itemid',
        how='left'
    )
    
    # Print statistics about unmapped lab tests
    remaining_labs = df_events.filter(pl.col('code').str.starts_with('LAB_TEST_')).shape[0]
    unmapped_labs = df_events.filter(
        pl.col('code').str.starts_with('LAB_TEST_') & 
        pl.col('label').is_null()
    )
    unmapped_count = unmapped_labs.shape[0]
    
    log.info("Lab test mapping statistics (after null value filtering):")
    log.info(f"Remaining rows of lab test codes: {remaining_labs}")
    log.info(f"Rows with unmapped lab tests: {unmapped_count}")
    if remaining_labs > 0:
        log.info(f"Percentage of remaining rows to be removed due to missing mappings: {(unmapped_count/remaining_labs*100):.2f}%")
    
    # Convert codes and filter out unmapped lab tests, incorporating units into lab names
    df_events = (
        df_events
        .with_columns([
            pl.when(pl.col('code').str.starts_with('LAB_TEST_'))
            .then(
                pl.when(pl.col('label').is_not_null())
                .then(
                    pl.concat_str([
                        pl.lit('LAB_'),
                        pl.col('label').str.replace_all(' ', '_').str.replace_all(',', '_'),
                        pl.lit('_'),
                        pl.when(pl.col('unit').is_null())
                        .then(pl.lit('NO_UNIT'))
                        .otherwise(pl.col('unit').str.replace_all(' ', '_').str.replace_all(',', '_'))
                    ])
                )
                .otherwise(pl.col('code'))
            )
            .otherwise(pl.col('code'))
            .alias('code')
        ])
        # Filter out lab tests that couldn't be mapped
        .filter(
            ~(pl.col('code').str.starts_with('LAB_TEST_'))
        )
    )
    
    # Get frequency of each lab test
    lab_frequencies = (
        df_events
        .filter(pl.col('code').str.starts_with('LAB_'))
        .group_by('code')
        .agg(pl.count().alias('frequency'))
        .sort('frequency', descending=True)
    )
    freq_dict = dict(zip(lab_frequencies.get_column('code').to_list(), lab_frequencies.get_column('frequency').to_list()))
    with open(f'{args.LOG_DIR}/cohort_stat/lab_test_frequency.json', 'w') as f:
        json.dump(freq_dict, f, indent=2)
    
    # Analyze the most common numeric format for each lab test
    def get_decimal_places(val):
        import math
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return None
        val_str = f"{val:.10f}".rstrip('0').rstrip('.')
        if '.' not in val_str:
            return 0
        return len(val_str.split('.')[1])
    
    lab_format_dict = {}
    lab_codes = df_events.filter(pl.col('code').str.starts_with('LAB_')).get_column('code').unique().to_list()
    for lab_code in lab_codes:
        values = df_events.filter(pl.col('code') == lab_code).get_column('numeric_value').to_list()
        format_counts = {'integer': 0, 'one_decimal': 0, 'two_or_more_decimal': 0}
        for v in values:
            dp = get_decimal_places(v)
            if dp is None:
                continue
            elif dp == 0:
                format_counts['integer'] += 1
            elif dp == 1:
                format_counts['one_decimal'] += 1
            else:
                format_counts['two_or_more_decimal'] += 1
        most_common = max(format_counts, key=format_counts.get) if sum(format_counts.values()) > 0 else 'unknown'
        lab_format_dict[lab_code] = {'most_common_format': most_common, 'counts': format_counts}
    
    with open(f'{args.LOG_DIR}/cohort_stat/lab_test_format.json', 'w') as f:
        json.dump(lab_format_dict, f, indent=2)
    
    # Get total number of unique lab tests
    total_lab_codes = lab_frequencies.shape[0]
    
    # Analyze coverage for different top-N selections
    log.info("Lab Test Coverage Analysis:")
    log.info(f"Total unique lab tests: {total_lab_codes}")
    
    # Get total records and patients with lab tests
    lab_records = df_events.filter(pl.col('code').str.starts_with('LAB_'))
    total_records = lab_records.shape[0]
    total_patients = lab_records.select('subject_id').unique().shape[0]
    log.info(f"Total lab test records: {total_records}")
    log.info(f"Total patients with lab tests: {total_patients}")
    
    log.info("Coverage analysis for different top-N selections:")
    log.info(f"{'Top-N':>8} | {'Records Kept':>12} | {'Records %':>9} | {'Patients Kept':>13} | {'Patients %':>10}")
    log.info("-" * 65)
    
    # Analyze coverage from 100 to total_lab_codes with interval of 100
    for n in range(100, total_lab_codes + 100, 100):
        if n > total_lab_codes:
            break
            
        # Get the top N codes
        top_n_codes = lab_frequencies.head(n).get_column('code').to_list()
        
        # Calculate coverage
        records_kept = lab_records.filter(pl.col('code').is_in(top_n_codes)).shape[0]
        patients_kept = lab_records.filter(pl.col('code').is_in(top_n_codes)).select('subject_id').unique().shape[0]
        
        records_pct = (records_kept / total_records) * 100
        patients_pct = (patients_kept / total_patients) * 100
        
        log.info(f"{n:>8} | {records_kept:>12} | {records_pct:>8.2f}% | {patients_kept:>13} | {patients_pct:>9.2f}%")
    
    # Get the top lab_top_n most frequent lab tests as a list
    if args.lab_top_n == -1:
        top_labs = lab_frequencies.get_column('code').to_list()
    else:   
        log.info(f"Proceeding with top {args.lab_top_n} lab tests as before...")
        top_labs = lab_frequencies.head(args.lab_top_n).get_column('code').to_list()
    
    # Count records that will be removed
    total_lab_records = df_events.filter(pl.col('code').str.starts_with('LAB_')).shape[0]
    df_events_filtered = df_events.filter(
        ~pl.col('code').str.starts_with('LAB_') | 
        pl.col('code').is_in(top_labs)
    )
    remaining_lab_records = df_events_filtered.filter(pl.col('code').str.starts_with('LAB_')).shape[0]
    removed_records = total_lab_records - remaining_lab_records
    
    log.info(f"Total lab records before filtering: {total_lab_records}")
    log.info(f"Lab records after filtering: {remaining_lab_records}")
    log.info(f"Records removed: {removed_records}")
    if total_lab_records > 0:
        log.info(f"Percentage of lab records removed: {(removed_records/total_lab_records*100):.2f}%")
    
    # Calculate percentile boundaries for each lab test (P0=min, P10-P90, P100=max)
    lab_quantiles = {}
    for lab_code in top_labs:
        # Get numeric values for this lab test
        values = df_events_filtered.filter(pl.col('code') == lab_code).select('numeric_value')
        # Calculate percentile boundaries: P0 (min), P10-P90, P100 (max)
        # ! use 0.01 and 0.99 instead of 0 and 1 to avoid extreme values
        quantiles = values.select([
            pl.col('numeric_value').quantile(0.01).alias('P0'),
            *[pl.col('numeric_value').quantile(q/10).alias(f'P{q*10}') for q in range(1, 10)],
            pl.col('numeric_value').quantile(0.99).alias('P100')
        ]).row(0)  # Get the first (and only) row as a tuple
        
        # Format quantiles based on the most common format for this lab test
        fmt = lab_format_dict[lab_code]['most_common_format']
        formatted_quantiles = {}
        for i, p in enumerate([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
            val = quantiles[i]
            if val is None:
                formatted_quantiles[f'P{p}'] = None
            elif fmt == 'integer':
                formatted_quantiles[f'P{p}'] = int(round(val))
            elif fmt == 'one_decimal':
                formatted_quantiles[f'P{p}'] = round(val, 1)
            else:  # two_or_more_decimal
                formatted_quantiles[f'P{p}'] = round(val, 2)
        lab_quantiles[lab_code] = formatted_quantiles
    
    # Save quantile boundaries to a JSON file for later reference
    with open(f'{args.LOG_DIR}/cohort_stat/lab_test_quantiles.json', 'w') as f:
        json.dump(lab_quantiles, f, indent=4)
    
    # Create a list to store DataFrames for each lab test's quantile rows
    quantile_dfs = []
    
    # Process each lab test
    for lab_code in top_labs:
        boundaries = lab_quantiles[lab_code]
        lab_records = df_events_filtered.filter(pl.col('code') == lab_code)
        
        # Create conditions for quantile bucket assignment (_Q1 to _Q10)
        # _Q1: ≤P10, _Q2: (P10,P20], ..., _Q10: >P90
        # Note: P0=P1 and P100=P99 to clip extreme outliers (bottom/top 1%)
        # Outliers are captured in _Q1/_Q10 but won't be reproduced during sampling
        percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for i in range(1, 11):  # _Q1 to _Q10
            lower_p, upper_p = percentiles[i-1], percentiles[i]
            if i == 1:
                condition = pl.col('numeric_value') <= boundaries[f'P{upper_p}']
            elif i == 10:
                condition = pl.col('numeric_value') > boundaries[f'P{lower_p}']  # No upper bound for last bucket
            else:
                condition = (pl.col('numeric_value') > boundaries[f'P{lower_p}']) & (pl.col('numeric_value') <= boundaries[f'P{upper_p}'])
            
            # Create new rows for this quantile
            quantile_rows = lab_records.filter(condition).with_columns([
                pl.lit(f'_Q{i}').alias('code'),
                pl.col('code').alias('original_code'),  # Store original lab code
                pl.lit(1).alias('is_quantile')  # Flag for quantile rows
            ])
            quantile_dfs.append(quantile_rows)
    
    # Add flags to original data
    df_events_filtered = df_events_filtered.with_columns([
        pl.col('code').alias('original_code'),
        pl.lit(0).alias('is_quantile')
    ])
    
    # Combine all quantile rows with original data
    if quantile_dfs:
        df_with_quantiles = pl.concat([df_events_filtered] + quantile_dfs)
    else:
        df_with_quantiles = df_events_filtered
    
    # Clean up temporary columns and drop the unit column since it's now part of the code
    df_with_quantiles = df_with_quantiles.drop(['lab_id_temp', 'label', 'unit'])
    
    # Sort by subject_id, time, original_code (to group lab tests together), 
    # and is_quantile (to put quantiles after their lab tests)
    df_with_quantiles = (
        df_with_quantiles
        .sort(['subject_id', 'sort_order', 'time', 'original_code', 'is_quantile'])
        .drop(['original_code', 'is_quantile'])  # Remove temporary sorting columns
    )
    
    return df_with_quantiles

def verify_codes(df_events: pl.DataFrame) -> None:
    """Verify the standardization of codes."""
    log.info("Verifying standardized codes:")
    log.info("Sex codes:")
    log.info(df_events.filter(pl.col('code').str.contains('SEX_')).select(pl.col('code').unique()))
    
    log.info("Race codes:")
    log.info(df_events.filter(pl.col('code').str.contains('RACE_')).select(pl.col('code').unique()))
    
    log.info("Marital status codes:")
    log.info(df_events.filter(pl.col('code').str.contains('MARITAL_STATUS_')).select(pl.col('code').unique()))
    
    log.info("Count of remaining ICD9CM codes:")
    log.info(df_events.filter(pl.col('code').str.starts_with('ICD9CM_')).select(['code']).unique().shape[0])
    assert df_events.filter(pl.col('code').str.starts_with('ICD9CM_')).select(['code']).unique().shape[0] == 0
    
    log.info("Count of remaining ICD9PCS codes:")
    log.info(df_events.filter(pl.col('code').str.starts_with('ICD9PCS_')).select(['code']).unique().shape[0])
    assert df_events.filter(pl.col('code').str.starts_with('ICD9PCS_')).select(['code']).unique().shape[0] == 0

    log.info("Count of remaining MEDICATION_ codes:")
    log.info(df_events.filter(pl.col('code').str.starts_with('MEDICATION_')).select('code').unique().shape[0])
    assert df_events.filter(pl.col('code').str.starts_with('MEDICATION_')).select('code').unique().shape[0] == 0
    
    log.info("Count of remaining LAB_TEST_ codes:")
    log.info(df_events.filter(pl.col('code').str.starts_with('LAB_TEST_')).select('code').unique().shape[0])
    assert df_events.filter(pl.col('code').str.starts_with('LAB_TEST_')).select('code').unique().shape[0] == 0

def remove_birth_records(df_events: pl.DataFrame) -> pl.DataFrame:
    """Remove BIRTH records since age is now encoded in age tokens."""
    return df_events.filter(~pl.col('code').str.contains('BIRTH'))

def add_year_from_2000_token(df_events: pl.DataFrame) -> pl.DataFrame:
    """Extract year from RACE records, calculate offset using Anchor_Year_Offset, and create discretized year offset tokens."""
    # Find Anchor_Year_Offset records
    anchor_year_data = (
        df_events
        .filter(pl.col('code').str.contains('Anchor_Year_Offset'))
        .select(['subject_id', 'code', 'visit_id'])
        .with_columns([
            # Extract offset from Anchor_Year_Offset/XXX format
            pl.col('code').str.extract(r'Anchor_Year_Offset/(\d+)', 1).cast(pl.Int64).alias('anchor_year_offset')
        ])
        .select(['subject_id', 'anchor_year_offset', 'visit_id'])
    )
    
    # Find RACE records (sort_order == 2) to get the event year
    race_data = (
        df_events
        .filter(pl.col('sort_order') == 2)
        .select(['subject_id', 'time', 'visit_id'])
        .with_columns([
            # Extract year from the time field
            pl.col('time').dt.year().alias('race_year')
        ])
        .select(['subject_id', 'race_year', 'visit_id', 'time'])  # Keep the time column
    )
    
    if anchor_year_data.shape[0] > 0 and race_data.shape[0] > 0:
        # Get the data types from the original DataFrame
        original_dtypes = dict(zip(df_events.columns, df_events.dtypes))
        
        # Join anchor year offset data with race data by subject_id
        combined_data = anchor_year_data.join(race_data, on='subject_id', how='inner')
        
        # Calculate the final year offset: race_year - anchor_year_offset - 2000
        year_offset_records = (
            combined_data
            .with_columns([
                (pl.col('race_year') - pl.col('anchor_year_offset')).alias('actual_year')
            ])
            .with_columns([
                # Create year token: YEAR_{actual_year}
                pl.concat_str([
                    pl.lit('YEAR_'), 
                    pl.col('actual_year').cast(pl.Utf8)
                ]).alias('code')
            ])
            .select(['subject_id', 'visit_id', 'code'])
            .join(race_data.select(['subject_id', 'time']), on='subject_id', how='left')  # Get time from race data
            .with_columns([
                pl.lit(4).alias('sort_order'),  # YEAR_FROM_2000 gets sort order 4
                pl.lit(None).cast(original_dtypes['numeric_value']).alias('numeric_value'),
                pl.lit(None).cast(original_dtypes['seq_num']).alias('seq_num'),
                pl.lit(None).alias('table'),
                pl.lit(None).cast(original_dtypes['text_value']).alias('text_value')
            ])
        )
        
        # Get the exact column order from df_events
        df_columns = df_events.columns
        
        # Select columns in the same order as df_events, filling missing ones with correct types
        year_offset_records_aligned = year_offset_records.select([
            pl.col(col).cast(original_dtypes[col]) if col in year_offset_records.columns 
            else pl.lit(None).cast(original_dtypes[col]).alias(col)
            for col in df_columns
        ])
        
        # Remove original Anchor_Year_Offset records and add new year offset records
        df_events_without_anchor = df_events.filter(~pl.col('code').str.contains('Anchor_Year_Offset'))
        df_events_with_year_offset = pl.concat([df_events_without_anchor, year_offset_records_aligned])
        
        # Re-sort everything
        return df_events_with_year_offset.sort(['subject_id', 'sort_order', 'time'])
    
    else:
        log.info("Warning: No Anchor_Year_Offset records or RACE records found")
        return df_events

def get_top_n_codes(df: pl.DataFrame, code_prefix: str, n: int, json_path: str = None) -> list[str]:
    """Get the top N most frequent codes for a given prefix.
    If n == -1, returns all codes sorted by frequency."""
    code_freq_df = (
        df.filter(pl.col('code').str.starts_with(code_prefix))
        .group_by('code')
        .agg(pl.count().alias('frequency'))
        .sort('frequency', descending=True)
    )
    # Optionally save full frequency dictionary
    if json_path:
        # Convert to dictionary: {code: frequency}
        freq_dict = dict(zip(
            code_freq_df['code'].to_list(),
            code_freq_df['frequency'].to_list()
        ))
        with open(json_path, 'w') as f:
            json.dump(freq_dict, f, indent=2)
    
    # Return all codes if n == -1, otherwise return top n codes
    if n == -1:
        log.info(f"Returning all {code_prefix} codes...")
        return code_freq_df.get_column('code').to_list()
    log.info(f"Returning top {n} {code_prefix} codes...")
    return code_freq_df.head(n).get_column('code').to_list()

def analyze_icd_coverage(df: pl.DataFrame, code_prefix: str, top_n: list[int]) -> None:
    """
    Analyze coverage of ICD codes for different top-N selections.
    
    Args:
        df: DataFrame containing the processed data
        code_prefix: Prefix of the codes to analyze (e.g., 'ICD10CM_' or 'ICD10PCS_')
        top_n: List of N values to analyze (e.g., [100, 200, 500])
    """
    # Get all records with this ICD type
    icd_records = df.filter(pl.col('code').str.starts_with(code_prefix))
    total_records = icd_records.shape[0]
    total_patients = icd_records.select('subject_id').unique().shape[0]
    
    # Get frequency of each code
    code_freq = (
        icd_records
        .group_by('code')
        .agg(
            pl.count().alias('frequency'),
            pl.col('subject_id').n_unique().alias('n_patients')
        )
        .sort('frequency', descending=True)
    )
    
    total_codes = code_freq.shape[0]
    log.info(f"{code_prefix} Analysis:")
    log.info(f"Total unique codes: {total_codes}")
    log.info(f"Total records: {total_records}")
    log.info(f"Total patients with these codes: {total_patients}")
    
    # Analyze coverage for different top-N selections
    log.info("Coverage analysis for different top-N selections:")
    log.info(f"{'Top-N':>8} | {'Records Kept':>12} | {'Records %':>9} | {'Patients Kept':>13} | {'Patients %':>10}")
    log.info("-" * 65)
    
    for n in top_n:
        if n > total_codes:
            log.info(f"Note: Requested top-{n} exceeds total codes ({total_codes})")
            continue
            
        # Get the top N codes
        top_n_codes = code_freq.head(n).get_column('code').to_list()
        
        # Calculate coverage
        records_kept = icd_records.filter(pl.col('code').is_in(top_n_codes)).shape[0]
        patients_kept = icd_records.filter(pl.col('code').is_in(top_n_codes)).select('subject_id').unique().shape[0]
        
        records_pct = (records_kept / total_records) * 100
        patients_pct = (patients_kept / total_patients) * 100
        
        log.info(f"{n:>8} | {records_kept:>12} | {records_pct:>8.2f}% | {patients_kept:>13} | {patients_pct:>9.2f}%")

def filter_codes(df_events: pl.DataFrame, args: argparse.Namespace):
    # Print initial statistics
    log.info("Initial Statistics:")
    log.info(f"Patients: {df_events.select('subject_id').unique().shape[0]}")
    log.info(f"Visits (non-null): {df_events.filter(pl.col('visit_id').is_not_null()).select('visit_id').unique().shape[0]}")
    log.info(f"Total records: {df_events.shape[0]}")
    
    # Define different intervals for different code types
    icd_values = list(range(1000, 10001, 1000))  # [1000, 2000, ..., 10000]
    
    # For ATC codes, first get total count to determine the range
    atc_total = df_events.filter(pl.col('code').str.starts_with('ATC_')).select('code').unique().shape[0]
    atc_values = list(range(100, atc_total + 100, 100))  # [100, 200, ..., up to total]
    
    # Analyze each code type with appropriate intervals
    analyze_icd_coverage(df_events, 'ICD10CM_', icd_values)
    analyze_icd_coverage(df_events, 'ICD10PCS_', icd_values)
    analyze_icd_coverage(df_events, 'ATC_', atc_values)
    
    # Get top N codes for each type
    top_icd10cm = get_top_n_codes(df_events, 'ICD10CM_', args.icd10cm_top_n, f"{args.LOG_DIR}/cohort_stat/icd10cm_code_frequency.json")
    top_icd10pcs = get_top_n_codes(df_events, 'ICD10PCS_', args.icd10pcs_top_n, f"{args.LOG_DIR}/cohort_stat/icd10pcs_code_frequency.json")
    top_atc = get_top_n_codes(df_events, 'ATC_', args.atc_top_n, f"{args.LOG_DIR}/cohort_stat/atc_code_frequency.json")
    
    # Print statistics for each code type before filtering
    def print_code_stats(df, code_prefix, description):
        records = df.filter(pl.col('code').str.starts_with(code_prefix))
        patients = records.select('subject_id').unique().shape[0]
        visits = records.filter(pl.col('visit_id').is_not_null()).select('visit_id').unique().shape[0]
        codes = records.select('code').unique().shape[0]
        total_records = records.shape[0]
        log.info(f"{description} Statistics Before Filtering:")
        log.info(f"Unique codes: {codes}")
        log.info(f"Total records: {total_records}")
        log.info(f"Patients with these codes: {patients}")
        log.info(f"Visits with these codes: {visits}")
        return total_records
    
    total_icd10cm = print_code_stats(df_events, 'ICD10CM_', 'ICD10CM')
    total_icd10pcs = print_code_stats(df_events, 'ICD10PCS_', 'ICD10PCS')
    total_atc = print_code_stats(df_events, 'ATC_', 'ATC')
    
    # Combine all top codes
    all_top_codes = top_icd10cm + top_icd10pcs + top_atc
    
    # Filter the dataset to keep only records with top N codes for ICD and ATC
    # but keep all demographics and lab test records
    filtered_df = df_events.filter(
        (pl.col('code').is_in(all_top_codes)) |  # Keep selected ICD and ATC codes
        (~pl.col('code').str.starts_with('ICD10CM_')) &  # Keep non-ICD10CM codes
        (~pl.col('code').str.starts_with('ICD10PCS_')) &  # Keep non-ICD10PCS codes
        (~pl.col('code').str.starts_with('ATC_'))  # Keep non-ATC codes
    )
    
    # Print statistics after filtering
    def print_filtered_code_stats(df, code_prefix, description, total_before):
        records = df.filter(pl.col('code').str.starts_with(code_prefix))
        patients = records.select('subject_id').unique().shape[0]
        visits = records.filter(pl.col('visit_id').is_not_null()).select('visit_id').unique().shape[0]
        codes = records.select('code').unique().shape[0]
        total_records = records.shape[0]
        log.info(f"{description} Statistics After Filtering:")
        log.info(f"Unique codes: {codes}")
        log.info(f"Total records: {total_records}")
        log.info(f"Records removed: {total_before - total_records}")
        log.info(f"Patients with these codes: {patients}")
        log.info(f"Visits with these codes: {visits}")
    
    print_filtered_code_stats(filtered_df, 'ICD10CM_', 'ICD10CM', total_icd10cm)
    print_filtered_code_stats(filtered_df, 'ICD10PCS_', 'ICD10PCS', total_icd10pcs)
    print_filtered_code_stats(filtered_df, 'ATC_', 'ATC', total_atc)
    
    # Print final overall statistics
    log.info("Final Overall Statistics:")
    log.info(f"Patients: {filtered_df.select('subject_id').unique().shape[0]}")
    log.info(f"Visits (non-null): {filtered_df.filter(pl.col('visit_id').is_not_null()).select('visit_id').unique().shape[0]}")
    log.info(f"Total records: {filtered_df.shape[0]}")
    log.info(f"Records removed: {df_events.shape[0] - filtered_df.shape[0]}")
    
    # Save the filtered dataset
    log.info(f"Saving filtered dataset to {args.LOG_DIR}/cohort_stat/mimiciv_2.2_meds_processed.parquet...")
    filtered_df.write_parquet(f"{args.LOG_DIR}/cohort_stat/mimiciv_2.2_meds_processed.parquet")
    log.info("Filtering Done!")
    get_dataset_statistics(filtered_df, f'{args.LOG_DIR}/cohort_stat')
    save_all_plus_one_subject(filtered_df, f'{args.LOG_DIR}/cohort_stat')

def keep_first_visit_race(df_events: pl.DataFrame) -> pl.DataFrame:
    """Keep only the race record from the first visit for each patient."""
    
    # Get the first race record for each patient based on earliest time
    first_race_records = (
        df_events
        .filter(pl.col('code').str.contains('MIMIC_IV_Race'))
        .group_by('subject_id')
        .agg([
            pl.col('time').min().alias('first_time')
        ])
    )
    
    # Join back to get complete records and filter out later race records
    df_with_first_race = (
        df_events
        .join(first_race_records, on='subject_id', how='left')
        .filter(
            ~pl.col('code').str.contains('MIMIC_IV_Race') | 
            ((pl.col('code').str.contains('MIMIC_IV_Race')) & (pl.col('time') == pl.col('first_time')))
        )
        .drop('first_time')
    )
    
    # Print diagnostic information
    total_race_before = df_events.filter(pl.col('code').str.contains('MIMIC_IV_Race')).shape[0]
    total_race_after = df_with_first_race.filter(pl.col('code').str.contains('MIMIC_IV_Race')).shape[0]
    log.info(f"Race Record Deduplication:")
    log.info(f"Race records before: {total_race_before}")
    log.info(f"Race records after: {total_race_after}")
    log.info(f"Removed records: {total_race_before - total_race_after}")
    
    return df_with_first_race

def keep_first_visit_marital_status(df_events: pl.DataFrame) -> pl.DataFrame:
    """Keep only the marital status record from the first visit for each patient."""
    
    # Get the first marital status record for each patient based on earliest time
    first_marital_records = (
        df_events
        .filter(pl.col('code').str.contains('MIMIC_IV_Marital_Status'))
        .group_by('subject_id')
        .agg([
            pl.col('time').min().alias('first_time')
        ])
    )
    
    # Join back to get complete records and filter out later marital status records
    df_with_first_marital = (
        df_events
        .join(first_marital_records, on='subject_id', how='left')
        .filter(
            ~pl.col('code').str.contains('MIMIC_IV_Marital_Status') | 
            ((pl.col('code').str.contains('MIMIC_IV_Marital_Status')) & (pl.col('time') == pl.col('first_time')))
        )
        .drop('first_time')
    )
    
    # Print diagnostic information
    total_marital_before = df_events.filter(pl.col('code').str.contains('MIMIC_IV_Marital_Status')).shape[0]
    total_marital_after = df_with_first_marital.filter(pl.col('code').str.contains('MIMIC_IV_Marital_Status')).shape[0]
    log.info(f"Marital Status Record Deduplication:")
    log.info(f"Marital status records before: {total_marital_before}")
    log.info(f"Marital status records after: {total_marital_after}")
    log.info(f"Removed records: {total_marital_before - total_marital_after}")
    
    return df_with_first_marital

def adjust_time_demographics(df_events: pl.DataFrame) -> pl.DataFrame:
    """Set time for demographic records to match the time of SEX record."""
    # Get the time from SEX records for each subject
    sex_times = (
        df_events
        .filter(pl.col('code').str.starts_with('SEX_'))
        .select(['subject_id', 'time'])
        .rename({'time': 'demo_time'})
    )
    
    # Join with original data
    df_events = df_events.join(
        sex_times,
        on='subject_id',
        how='left'
    )
    
    # Update time for demographic records
    df_events = df_events.with_columns([
        pl.when(
            (pl.col('sort_order').is_in([0, 2, 3, 4]))  # AGE, RACE, MARITAL_STATUS, YEAR
        )
        .then(pl.col('demo_time'))
        .otherwise(pl.col('time'))
        .alias('time')
    ])
    # Drop the temporary column
    return df_events.drop('demo_time')

def remove_visit_id_from_demographics(df_events: pl.DataFrame) -> pl.DataFrame:
    """Remove visit_id from all demographic records (sort_order != 5)."""
    df_events = df_events.with_columns([
        pl.when(pl.col('sort_order') != 5)
        .then(pl.lit(None).cast(df_events.schema['visit_id']))
        .otherwise(pl.col('visit_id'))
        .alias('visit_id')
    ])
    
    # Print diagnostic information
    total_records = df_events.shape[0]
    demo_records = df_events.filter(pl.col('sort_order') != 5).shape[0]
    demo_with_visit = df_events.filter(
        (pl.col('sort_order') != 5) & 
        pl.col('visit_id').is_not_null()
    ).shape[0]
    
    log.info(f"Demographic Record Visit ID Removal:")
    log.info(f"Total records: {total_records}")
    log.info(f"Demographic records: {demo_records}")
    log.info(f"Demographic records with visit_id removed: {demo_with_visit}")
    
    return df_events

def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{args.LOG_DIR}/logs/process_mimiciv_meds.log"),  # Add a file
        ]
    )
    # Load data
    os.makedirs(f'{args.LOG_DIR}/cohort_stat', exist_ok=True)
    df_events = load_data()

    # Add sort order
    df_events = add_sort_order(df_events)
    # Remove visit_id from demographic records
    df_events = remove_visit_id_from_demographics(df_events)
    log.info(f'The number of records in the dataset is {df_events.shape[0]}')
    # Process demographics
    df_events = filter_null_visit(df_events)
    log.info('after filter_null_visit')
    log.info(f'Patients: {df_events.select("subject_id").unique().shape[0]}')
    log.info(f'Visits: {df_events.filter(pl.col("sort_order") == 5).select("visit_id").unique().shape[0]}')
    log.info(f'Records: {df_events.shape[0]}')
    
    df_events = remove_unused_columns(df_events)
    
    # Keep only first visit race and marital status records
    df_events = keep_first_visit_race(df_events)
    log.info('after keep_first_visit_race')
    log.info(f'Patients: {df_events.select("subject_id").unique().shape[0]}')
    log.info(f'Visits: {df_events.filter(pl.col("sort_order") == 5).select("visit_id").unique().shape[0]}')
    log.info(f'Records: {df_events.shape[0]}')
    df_events = keep_first_visit_marital_status(df_events)
    log.info('after keep_first_visit_marital_status')
    log.info(f'Patients: {df_events.select("subject_id").unique().shape[0]}')
    log.info(f'Visits: {df_events.filter(pl.col("sort_order") == 5).select("visit_id").unique().shape[0]}')
    log.info(f'Records: {df_events.shape[0]}')
    
    # Calculate and add age
    df_events = calculate_and_add_age(df_events)
    log.info('after calculate_and_add_age')
    log.info(f'Patients: {df_events.select("subject_id").unique().shape[0]}')
    log.info(f'Visits: {df_events.filter(pl.col("sort_order") == 5).select("visit_id").unique().shape[0]}')
    log.info(f'Records: {df_events.shape[0]}')

    # Remove birth records since we now have age tokens
    df_events = remove_birth_records(df_events)
    log.info('after remove_birth_records')
    log.info(f'Patients: {df_events.select("subject_id").unique().shape[0]}')
    log.info(f'Visits: {df_events.filter(pl.col("sort_order") == 5).select("visit_id").unique().shape[0]}')
    log.info(f'Records: {df_events.shape[0]}')
    # Add year from 2000 token
    df_events = add_year_from_2000_token(df_events)
    log.info('after add_year_from_2000_token')
    log.info(f'Patients: {df_events.select("subject_id").unique().shape[0]}')
    log.info(f'Visits: {df_events.filter(pl.col("sort_order") == 5).select("visit_id").unique().shape[0]}')
    log.info(f'Records: {df_events.shape[0]}')
    # Standardize codes
    df_events = standardize_birth_death_gender(df_events)
    log.info('after standardize_birth_death_gender')
    log.info(f'Patients: {df_events.select("subject_id").unique().shape[0]}')
    log.info(f'Visits: {df_events.filter(pl.col("sort_order") == 5).select("visit_id").unique().shape[0]}')
    log.info(f'Records: {df_events.shape[0]}')
    df_events = standardize_race(df_events)
    log.info('after standardize_race')
    log.info(f'Patients: {df_events.select("subject_id").unique().shape[0]}')
    log.info(f'Visits: {df_events.filter(pl.col("sort_order") == 5).select("visit_id").unique().shape[0]}')
    log.info(f'Records: {df_events.shape[0]}')
    df_events = standardize_marital_status(df_events)
    log.info('after standardize_marital_status')
    log.info(f'Patients: {df_events.select("subject_id").unique().shape[0]}')
    log.info(f'Visits: {df_events.filter(pl.col("sort_order") == 5).select("visit_id").unique().shape[0]}')
    log.info(f'Records: {df_events.shape[0]}')
    df_events = adjust_time_demographics(df_events)
    log.info('after adjust_time_demographics')
    log.info(f'Patients: {df_events.select("subject_id").unique().shape[0]}')
    log.info(f'Visits: {df_events.filter(pl.col("sort_order") == 5).select("visit_id").unique().shape[0]}')
    log.info(f'Records: {df_events.shape[0]}')
    # Convert ICD9 to ICD10
    df_events = process_diagnoses(df_events)
    log.info('after process_diagnoses')
    log.info(f'Patients: {df_events.select("subject_id").unique().shape[0]}')
    log.info(f'Visits: {df_events.filter(pl.col("sort_order") == 5).select("visit_id").unique().shape[0]}')
    log.info(f'Records: {df_events.shape[0]}')
    df_events = process_procedures(df_events)
    log.info('after process_procedures')
    log.info(f'Patients: {df_events.select("subject_id").unique().shape[0]}')
    log.info(f'Visits: {df_events.filter(pl.col("sort_order") == 5).select("visit_id").unique().shape[0]}')
    log.info(f'Records: {df_events.shape[0]}')
    # Convert drugs to ATC codes
    df_events = process_medications(df_events)
    log.info('after process_medications')
    log.info(f'Patients: {df_events.select("subject_id").unique().shape[0]}')
    log.info(f'Visits: {df_events.filter(pl.col("sort_order") == 5).select("visit_id").unique().shape[0]}')
    log.info(f'Records: {df_events.shape[0]}')
    # Convert lab tests to meaningful names
    df_events = process_lab_tests(df_events, args)
    log.info('after process_lab_tests')
    log.info(f'Patients: {df_events.select("subject_id").unique().shape[0]}')
    log.info(f'Visits: {df_events.filter(pl.col("sort_order") == 5).select("visit_id").unique().shape[0]}')
    log.info(f'Records: {df_events.shape[0]}')
    # Verify results
    verify_codes(df_events)
    filter_codes(df_events, args)
    log.info('after filter_codes')
    log.info(f'Patients: {df_events.select("subject_id").unique().shape[0]}')
    log.info(f'Visits: {df_events.filter(pl.col("sort_order") == 5).select("visit_id").unique().shape[0]}')
    log.info(f'Records: {df_events.shape[0]}')


if __name__ == "__main__":
    args = parse_args()
    main(args)
    
    
