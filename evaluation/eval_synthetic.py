import warnings
import os, wandb
import random
import torch
import json
import argparse
import numpy as np
import polars as pl
from collections import defaultdict
from model.local_tokenizer import LocalTokenizer

from scripts.utils import load_config
from evaluation.utils import transform_data_matrix
from evaluation.fidelity import fidelity_from_halo_evaluation, fidelity_evaluation, plot_cooccurrence_heatmaps
from evaluation.utility_phenotype import utility_phenotype_pred
from evaluation.utility_mortality import utility_mortality_pred
from evaluation.utility_length_of_stay import utility_length_of_stay_pred
from evaluation.utility_readmission import utility_readmission_pred
from evaluation.privacy import privacy_evaluation


# import pdb; pdb.set_trace()
warnings.filterwarnings('ignore')

# --- define your time-gap buckets -> (lower_hours, upper_hours) ---
# we use the UPPER bound when deciding readmission (conservative).
_TIMEGAP_BOUNDS = {
    "_<=5m":   (0, 5/60),
    "_5m-15m": (5/60, 15/60),
    "_15m-1h": (15/60, 1),
    "_1h-2h":  (1, 2),
    "_2h-6h":  (2, 6),
    "_6h-12h": (6, 12),
    "_12h-1d": (12, 24),
    "_1d-3d":  (24, 72),
    "_3d-1w":  (72, 168),
    "_1w-2w":  (168, 336),
    "_2w-1mt": (336, 720),     # ~30 days
    "_1mt-3mt":(720, 2160),    # ~90 days
    "_3mt-6mt":(2160, 4320),   # ~180 days
    "_>6mt":   (4320, float("inf")),
}

def _build_timegap_id_map(tokenizer):
    """token_id -> (lower_hours, upper_hours) for all time-gap tokens in vocab."""
    m = {}
    for tok, (lo, hi) in _TIMEGAP_BOUNDS.items():
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            m[int(tid)] = (lo, hi)
    return m

def convert_to_structured_format(num_first_tokens, ehrs, tokenizer, source='syn'):
    """Convert raw EHR sequences to structured format."""
    demo_end_idx = num_first_tokens  # ['START_RECORD', 'AGE', 'SEX', 'RACE', 'MARITAL_STATUS', 'YEAR']
    # Define all special tokens
    start_visit_token = tokenizer.convert_tokens_to_ids('START_VISIT')
    end_visit_token = tokenizer.convert_tokens_to_ids('END_VISIT')
    end_record_token = tokenizer.convert_tokens_to_ids('END_RECORD')
    death_token = tokenizer.convert_tokens_to_ids('DEATH')
    demo_fields = ['AGE', 'SEX', 'RACE', 'MARITAL_STATUS', 'YEAR']
    demo_indices = range(1, num_first_tokens)  # Skip START_RECORD token
    # time-gap map: token_id -> (lo_hr, hi_hr)
    tg_map = _build_timegap_id_map(tokenizer)
    
    incomplete_num = 0
    ehr_outputs = []
    for i in range(len(ehrs)):
        if source=='syn' and end_record_token not in ehrs[i]:
            incomplete_num += 1
            continue
        # if source=='test' and len(ehrs[i]) > 2048:
        #     incomplete_num += 1
        #     continue
        demographics = ehrs[i][1:demo_end_idx]  
        visits = []
        gaps_hours = []          # gap from previous visit -> this visit (upper-bound hours)
        
        visit_codes = np.array(ehrs[i][demo_end_idx:], dtype=np.int64)
        flag_death = death_token in ehrs[i]

        starts = np.where(visit_codes == start_visit_token)[0]
        ends = np.where((visit_codes == end_visit_token) | (visit_codes == end_record_token))[0]
        
        for start_idx, end_idx in zip(starts, ends):
            codes = visit_codes[start_idx + 1:end_idx].astype(int).tolist()
            
            gap_hr = None
            if start_idx-1 >= 0:
                prev_tid = int(visit_codes[start_idx - 1])
                if prev_tid in tg_map:
                    _, hi = tg_map[prev_tid]
                    gap_hr = hi
            
            if end_idx == ends[-1] and flag_death:
                codes.append(death_token)
            
            if codes:
                visits.append(codes)
                gaps_hours.append(gap_hr)
            
            if visit_codes[end_idx] == end_record_token:
                break
        if visits:
            ehr_outputs.append({
                'demographics': demographics,  
                'visits': visits,
                "gap_hours": gaps_hours
            })
    # print(f'Incomplete rate in {source}: {incomplete_num} / {len(ehrs)}; skipped incomplete patients for fidelity evaluation')
    return ehr_outputs

def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to lists
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)  # Convert NumPy floats to Python floats
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)  # Convert NumPy integers to Python integers
    return obj  # Return as-is if not a NumPy type


# Global function to set random seed
def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed()
    parser = argparse.ArgumentParser()
    parser.add_argument('--LOG_DIR', required=True,
                        type=str, help='Path dir to the synthetic dataset')
    parser.add_argument('--top_p', type=float, default=0.98)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--fidelity', action="store_true",
                        help='Whether to evaluate fidelity')
    parser.add_argument('--utility', action="store_true",
                        help='Whether to evaluate utility')
    parser.add_argument('--privacy', action="store_true",
                        help='Whether to evaluate privacy')
    parser.add_argument('--heatmap', action="store_true",
                        help='Whether to evaluate heatmap')

    # Parse arguments
    args = parser.parse_args()
    print(args)
    # load real and synthetic data
    train_data = pl.read_parquet(os.path.join(args.LOG_DIR, "tokenized_sequences", "train.parquet"))
    val_data = pl.read_parquet(os.path.join(args.LOG_DIR, "tokenized_sequences", "val.parquet"))
    test_data = pl.read_parquet(os.path.join(args.LOG_DIR, "tokenized_sequences", "test.parquet"))
    synthetic_data_dir = os.path.join(args.LOG_DIR, f"synthetic_data_topp_{args.top_p}_temperature_{args.temperature}")
    synthetic_data = pl.read_parquet(os.path.join(synthetic_data_dir, "synthetic_data.parquet"))
    # Load HuggingFace tokenizer
    tokenizer = LocalTokenizer(os.path.join(args.LOG_DIR, "tokenizer", "vocab.csv"))
    
    evaluation_file = os.path.join(synthetic_data_dir, "synthetic_eval.json")
    structured_syn_data = convert_to_structured_format(num_first_tokens=6,
        ehrs=synthetic_data['synthetic_sequence'].to_list(),
        tokenizer=tokenizer,
        source='syn'
    )
    structured_train_data = convert_to_structured_format(num_first_tokens=6,
        ehrs=train_data['concept_token_ids'].to_list(),
        tokenizer=tokenizer,
        source='train'
    )
    structured_val_data = convert_to_structured_format(num_first_tokens=6,
        ehrs=val_data['concept_token_ids'].to_list(),
        tokenizer=tokenizer,
        source='val'
    )
    structured_test_data = convert_to_structured_format(num_first_tokens=6,
        ehrs=test_data['concept_token_ids'].to_list(),
        tokenizer=tokenizer,
        source='test'
    )

    # Prepare matrices for evaluation
    test_data_matrices = {
        matrix_type: transform_data_matrix(
            structured_test_data, code_vocab_size=tokenizer.vocab_size, matrix_type=matrix_type)
        for matrix_type in ["binary", "count", "probability"]
    }
    synthetic_data_matrices = {
        matrix_type: transform_data_matrix(
            structured_syn_data, code_vocab_size=tokenizer.vocab_size, matrix_type=matrix_type)
        for matrix_type in ["binary", "count", "probability"]
    }
    # Initialize results dictionary
    results = defaultdict(lambda: defaultdict(dict))
    if args.fidelity:
        print("Evaluating fidelity...")
        # Align number of rows
        if test_data_matrices["binary"].shape[0] != synthetic_data_matrices["binary"].shape[0]:
            min_rows = min(
                test_data_matrices["binary"].shape[0], synthetic_data_matrices["binary"].shape[0])
            print(f"Truncating to {min_rows} rows")
            for matrix_type in test_data_matrices.keys():
                test_data_matrices[matrix_type] = test_data_matrices[matrix_type][:min_rows]
                synthetic_data_matrices[matrix_type] = synthetic_data_matrices[matrix_type][:min_rows]
        
        results["fidelity_from_halo"] = fidelity_from_halo_evaluation(
            structured_test_data, structured_syn_data, save_dir=synthetic_data_dir)
        results['fidelity'] = {}
        for matrix_type in ["binary", "count", "probability"]:
            print(f"Evaluating with {matrix_type} matrix...")
            fidelity_results = fidelity_evaluation(
                test_data_matrices[matrix_type],
                synthetic_data_matrices[matrix_type],
                matrix_type
            )
            results['fidelity'].update(fidelity_results)
    if args.utility:
        print("Evaluating utility...")
        results['utility_phenotype_pred'] = utility_phenotype_pred(tokenizer, structured_train_data, structured_val_data, structured_test_data, structured_syn_data, synthetic_data_dir)
        results['utility_mortality_pred'] = utility_mortality_pred(tokenizer, structured_train_data, structured_val_data, structured_test_data, structured_syn_data, synthetic_data_dir)
        results['utility_length_of_stay_pred'] = utility_length_of_stay_pred(tokenizer, structured_train_data, structured_val_data, structured_test_data, structured_syn_data, synthetic_data_dir)
        results['utility_readmission_pred'] = utility_readmission_pred(tokenizer, structured_train_data, structured_val_data, structured_test_data, structured_syn_data, synthetic_data_dir)
        
    if args.privacy:
        print("Evaluating privacy...")
        train_data = pl.read_parquet(os.path.join(args.LOG_DIR, "tokenized_sequences", "train.parquet"))
        structured_train_data = convert_to_structured_format(num_first_tokens=6,
            ehrs=train_data['concept_token_ids'].to_list(),
            tokenizer=tokenizer
        )
        results["privacy"] = privacy_evaluation(structured_train_data, structured_test_data, structured_syn_data)

    if args.heatmap:
        heatmap_dir = os.path.join(synthetic_data_dir, "heatmaps")
        vocab_w_concept_label = pl.read_csv(os.path.join(args.LOG_DIR, "tokenizer", "vocab_w_concept_label.csv"))
        concept_code_labels = dict(zip(vocab_w_concept_label["concept_code"], vocab_w_concept_label["concept_label"]))
        os.makedirs(heatmap_dir, exist_ok=True)
        # Collect all metrics in a dictionary
        res_diagnoses_v = plot_cooccurrence_heatmaps(
            structured_test_data, structured_syn_data, tokenizer,
            prefixes=("ICD10CM_",), topk=150, level="visit",
            title="Diagnoses co-occurrence", 
            out_path=os.path.join(heatmap_dir, "heatmap_diagnoses_visit.png"),
            concept_code_labels=concept_code_labels
        )
        results["diagnoses_visit"] = res_diagnoses_v["metrics"]
        

        res_medications_v = plot_cooccurrence_heatmaps(
            structured_test_data, structured_syn_data, tokenizer,
            prefixes=("ATC_",), topk=150, level="visit",
            title="Medications co-occurrence", 
            out_path=os.path.join(heatmap_dir, "heatmap_medications_visit.png")
        )
        results["medications_visit"] = res_medications_v["metrics"]
        

        res_procedures_v = plot_cooccurrence_heatmaps(
            structured_test_data, structured_syn_data, tokenizer,
            prefixes=("ICD10PCS_",), topk=150, level="visit",
            title="Procedures co-occurrence", 
            out_path=os.path.join(heatmap_dir, "heatmap_procedures_visit.png")
        )
        results["procedures_visit"] = res_procedures_v["metrics"]
        

        res_lab_tests_v = plot_cooccurrence_heatmaps(
            structured_test_data, structured_syn_data, tokenizer,
            prefixes=("LAB_",), topk=-1, level="visit",
            title="Lab tests co-occurrence", 
            out_path=os.path.join(heatmap_dir, "heatmap_lab_tests_visit.png")
        )
        results["lab_tests_visit"] = res_lab_tests_v["metrics"]
        
    with open(evaluation_file, 'w') as f:
        json.dump(results, f, indent=4, default=convert_numpy)
    print(f"Saved evaluation results to: {evaluation_file}")

if __name__ == "__main__":
    main()