"""
This script is used to map the codes in the vocabulary to the codes in MedTok.
"""
import json
import polars as pl
import re
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse
import os
from typing import Dict
import numpy as np
from scripts.token_semantic_desc_mapping import timegap_token_mapping_to_desc, special_tokens_mapping_to_desc
from scripts.utils import load_config
# Argument parsing
parser = argparse.ArgumentParser(description='Map the codes in the vocabulary to the codes in MedTok')
parser.add_argument('--LOG_DIR', type=str, default="output/2025-07-26_17_15_19",
                    help='Directory containing vocab.csv')
args = parser.parse_args()

atc_manual_mapping = {
    "A11EA": "A11DA",
    "L04AA10": "L04AA15",
    "N03AX12": "N03AX11",
    "N03AX16": "N03AX15",
    "V08BA02": "V08AA02",
    "A07FA": "A07XA",
    "A07FA01": "A07XA01",
    "A07FA02": "A07XA02",
    "B05DB": "B05XB",
    "J06BB16": "J06BB15",
    "L01XC06": "L01XD06",
    "L01XC08": "L01XD07",
    "L01XC12": "L01XD",
    "L01XC15": "L01XD",
    "L01XC17": "L01XD",
    "L01XC18": "L01XD",
    "L01XC19": "L01XD",
    "L01XC24": "L01XD",
    "L01XX70": "L01XX72",
    "L04AA13": "L04AA15",
    "L04AA27": "L04AA28",
    "L04AA29": "L04AA28",
    "L04AA31": "L04AA32",
    "L04AA33": "L04AA32",
    "L04AA34": "L04AA32",
    "N05CM19": "N05CM18",
    "R05X": "R05",
    "S03AA23": "S03AA30",
    "V06DF": "V06",
    "V10A": "V10"
}
icd10cm_manual_mapping = {
    "X92.9XXS": "X92-Y09",
    "Z23": "Z23.0",
    "Z56.0": "Z55.9",
    "Z66": "Z66-Z66",
    "Z75.1":"Z69-Z76",
    "Z78.0": "Z77.22",
    "Z78.1": "Z77.22",
    "O94": "O95",
    "W46.0XXA": "W45",
    "W46.1XXA": "W45",
    "Z32.00": "Z31.9",
    "Z32.01": "Z31.9",
    "Z32.02": "Z31.9",
    "Z34.00": "Z35.2",
    "Z34.03": "Z35.2",
    "Z34.80": "Z35.2",
    "Z34.83": "Z35.2",
    "Z34.90": "Z35.2",
    "Z34.91": "Z35.2",
    "Z56.1": "Z57.2",
    "Z56.2": "Z57.2",
    "Z56.3": "Z57.2",
    "Z56.4": "Z57.2",
    "Z56.5": "Z57.2",
    "Z56.6": "Z57.2",
    "Z56.89": "Z57.2",
    "Z56.9": "Z57.2",
    "Z68": "Z68.5",
    "Z69.021": "Z69-Z76",
    "Z69.11": "Z69-Z76",
    "Z75.0":"Z76",
    "Z75.2":"Z76",
    "Z75.4":"Z76",
    "Z75.5":"Z76",
    "Z75.8":"Z76",
    "Z78.9": "Z79.0"
}

def normalize_icd10cm(code: str) -> str:
    """Convert ICD10CM_B955 → B95.5 or ICD10CM_K9509 → K95.09"""
    code = code.replace("ICD10CM_", "")
    if len(code) > 3:
        return code[:3] + '.' + code[3:]
    else:
        return code

def find_best_icd10cm_code(code: str, codes_medtok: set) -> str:
    """
    Find the best approximate code for a given ICD-10-CM code.
    Prefers siblings with same prefix before falling back to truncation.
    """
    orig_code = code
    if orig_code in codes_medtok:
        return orig_code
    if orig_code in icd10cm_manual_mapping:
        return icd10cm_manual_mapping[orig_code]
    if orig_code.startswith("Z3A"):
        return "Z30-Z39"
    if orig_code.startswith("K95"):
        return "K90-K95"
    if '.' in code:
        prefix, suffix = code.split('.')
        # 1. Try other codes starting with the same prefix
        pattern = re.compile(rf"^{prefix}\.\d+$")
        sibling_candidates = [c for c in codes_medtok if pattern.match(c)]
        if sibling_candidates:
            # Compute numerical closeness of suffix
            try:
                suffix_val = int(suffix)
                closest = min(
                    sibling_candidates,
                    key=lambda x: abs(int(x.split('.')[1]) - suffix_val)
                )
                return closest
            except ValueError:
                pass  # fallback if suffix is non-numeric

        # 2. Try to find similar pattern codes (e.g. T36.8X5A → T36.8X2A)
        pattern = re.compile(rf"^{prefix}\.[^.]+$")
        pattern_candidates = [c for c in codes_medtok if pattern.match(c)]
        if pattern_candidates:
            # Find the most similar code by character position
            def similarity_score(candidate):
                # Count matching characters at same positions
                return sum(1 for a, b in zip(suffix, candidate.split('.')[1]) if a == b)
            
            closest = max(pattern_candidates, key=similarity_score)
            return closest

        # 3. Fall back to truncating suffix (E08.91 → E08.9 → E08)
        for i in range(len(suffix) - 1, -1, -1):
            candidate = f"{prefix}.{suffix[:i]}"
            if candidate in codes_medtok:
                return candidate
        code = prefix  # E08

    # 4. Try progressively shorter prefixes (E08 → E0 → E)
    for i in range(len(code), 0, -1):
        candidate = code[:i]
        if candidate in codes_medtok:
            return candidate
    return orig_code
    # raise ValueError(f"No match found for {orig_code}")

def find_best_icd10pcs_code(code: str, codes_medtok: set) -> str:
    """
    Find the best matching ICD10PCS code by finding codes with most common initial characters.
    ICD10PCS structure: each character position has specific meaning, so earlier positions are more important.
    For example: 037G3ZZ -> 037G0GZ (matching 037G is better than matching scattered positions)
    """
    if code in codes_medtok:
        return code
        
    # Find all codes that share at least the first 2 characters (more strict initial match)
    prefix = code[:2]
    candidates = [c for c in codes_medtok if c.startswith(prefix) and '.' not in c]
    
    if not candidates:
        # Fall back to single char prefix if no matches
        candidates = [c for c in codes_medtok if c.startswith(code[0])]
    
    if not candidates:
        raise ValueError(f"No match found for {code}")
        
    # Find longest matching prefix
    def similarity_score(candidate):
        # Count matching characters from start until first mismatch
        for i, (a, b) in enumerate(zip(code, candidate)):
            if a != b:
                return i
        return min(len(code), len(candidate))
        
    best_match = max(candidates, key=similarity_score)    
    return best_match

def main():
    args = parser.parse_args()
    cfg = load_config(os.path.join(args.LOG_DIR, "config.yaml"))
    vocab = pl.read_csv(os.path.join(args.LOG_DIR, "tokenizer", "vocab.csv"))
    codes_medtok = set(
        json.load(open("artifacts/medtok_files/code2tokens.json")).keys())
    print(f"Loaded {len(codes_medtok)} MedTok codes")
    
    # Load and convert to proper dictionary
    df = pl.read_parquet("artifacts/medtok_files/all_codes_mappings.parquet").filter(
        pl.col('code_system') == 'snomed').select(['med_code', 'desc'])
    medtok_code_to_desc = dict(zip(df['med_code'].to_list(), df['desc'].to_list()))
    print(f"Loaded {len(medtok_code_to_desc)} MedTok descriptions")
    additional_tokens_descs = {**special_tokens_mapping_to_desc, **timegap_token_mapping_to_desc}

    # Initialize ClinicalBERT
    model_name = cfg["aux_embeddings"]["semantic"]["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    # Pre-compute all MedTok with descriptions embeddings used for finding the best lab test match
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    all_descs = list(medtok_code_to_desc.values())
    print(f"Computed embeddings for {len(all_descs)} descriptions")
    
    # load the code embeddings in medtok
    code2embeddings_medtok = json.load(open("artifacts/medtok_files/code2embeddings.json"))
    print(f"Loaded {len(code2embeddings_medtok)} MedTok code embeddings")
    mappings = []  # List to store (our_code, medtok_code) pairs
    # embedding_dict: Dict[str, np.ndarray] = {}  # will save embeddings by token index (as strings)
    total_rows = len(vocab)
    for token_id, concept_code in tqdm(vocab.iter_rows(), total=total_rows, desc="Processing tokens", position=0):
        mapped_code = None
        if concept_code in additional_tokens_descs:
            pass
        elif concept_code.startswith("ATC_"):
            normalized_code = concept_code.split("ATC_")[1]
            if normalized_code in atc_manual_mapping:
                mapped_code = atc_manual_mapping[normalized_code]
            else:
                mapped_code = normalized_code
        elif concept_code.startswith("ICD10CM_"):
            normalized_code = normalize_icd10cm(concept_code)
            mapped_code = find_best_icd10cm_code(normalized_code, codes_medtok)
        elif concept_code.startswith("ICD10PCS_"):
            normalized_code = concept_code.split("ICD10PCS_")[1]
            mapped_code = find_best_icd10pcs_code(normalized_code, codes_medtok)
        elif concept_code.startswith("LAB_"):
            # ! do nothing for lab tests, low quality when mapping to medtok
            pass
        else:
            raise ValueError(f"Unknown concept_code: {concept_code}")
        mappings.append({"token_id": token_id, "concept_code": concept_code, "medtok_code": mapped_code})
    
    # Create DataFrame with two columns and save
    os.makedirs(f'{args.LOG_DIR}/knowledge_embd', exist_ok=True)
    df = pl.DataFrame(mappings)
    df.write_csv(os.path.join(args.LOG_DIR, "knowledge_embd", "vocab_mapped_to_medtok.csv"))

if __name__ == "__main__":
    main()