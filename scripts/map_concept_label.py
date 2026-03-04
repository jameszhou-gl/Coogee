# This script will get the concept_label for each concept_code in the tokenizer/vocab.csv file, especially ICD10CM, ICD10PCS, LAB, ATC codes.
# e.g, the concept_code of ATC_A01AB23 is corresponding to the concept_label of "minocycline" based on scripts/maps/atc_coding.csv.gz

import os
import polars as pl
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--LOG_DIR", type=str, required=True)
    return parser.parse_args()

ATC_CODE_DESC = {
    "D11AX27": "oxymetazoline",
    "L01FA01": "rituximab",
    "L01FD01": "Trastuzumab",
    "L01FD02": "Pertuzumab",
    "L01FG01": "Bevacizumab",
    "N06AX28": "Levomilnacipran",
    "S03AA23": "Polymyxin B, Neomycin und Gramicidin"
}

ICD10CM_CODE_DESC = {
    "NoDx": "No Diagnosis",
    "T406X": "Poisoning by, adverse effect of and underdosing of other and unspecified narcotics",
    "T432X1A": "Poisoning by, adverse effect of, and underdosing of other and unspecified antidepressants",
    "W19XXX": "Unspecified Fall",
    "Z919": "Patient's noncompliance with other medical treatment and regimen"
}

ICD10PCS_CODE_DESC = {
    "0HT0XZZ": "Release Left Lower Arm Skin, External Approach",
    "NoPCS": "No Procedure"
}

def main():
    args = parse_args()
    vocab = pl.read_csv(os.path.join(args.LOG_DIR, "tokenizer", "vocab.csv"))
    atc_coding = pl.read_csv(os.path.join('scripts', 'maps', "atc_coding.csv.gz"))
    atc_code_to_desc_mapping = dict(zip(atc_coding["atc_code"], atc_coding["atc_name"]))
    atc_code_to_desc_mapping.update(ATC_CODE_DESC)
    
    d_icd_diagnoses = pl.read_csv("dataset/mimiciv/2.2/hosp/d_icd_diagnoses.csv.gz", schema_overrides={"icd_code": pl.Utf8}).filter(pl.col("icd_version").eq(10))
    icd10cm_code_to_desc_mapping = dict(zip(d_icd_diagnoses["icd_code"], d_icd_diagnoses["long_title"]))
    icd10cm_code_to_desc_mapping.update(ICD10CM_CODE_DESC)
    d_icd_procedures = pl.read_csv("dataset/mimiciv/2.2/hosp/d_icd_procedures.csv.gz", schema_overrides={"icd_code": pl.Utf8}).filter(pl.col("icd_version").eq(10))
    icd10pcs_code_to_desc_mapping = dict(zip(d_icd_procedures["icd_code"], d_icd_procedures["long_title"]))
    icd10pcs_code_to_desc_mapping.update(ICD10PCS_CODE_DESC)
    concept_label = [None] * len(vocab)
    for index, concept_code in enumerate(vocab["concept_code"].to_list()):
        if concept_code.startswith("ATC_"):
            concept_label[index] = atc_code_to_desc_mapping[concept_code.split("ATC_")[1]]
        elif concept_code.startswith("ICD10CM_"):
            concept_label[index] = icd10cm_code_to_desc_mapping[concept_code.split("ICD10CM_")[1]]
        elif concept_code.startswith("ICD10PCS_"):
            concept_label[index] = icd10pcs_code_to_desc_mapping[concept_code.split("ICD10PCS_")[1]]
    vocab = vocab.with_columns([
        pl.Series(name="concept_label", values=concept_label).cast(pl.Utf8)
    ])
    vocab.write_csv(os.path.join(args.LOG_DIR, "tokenizer", "vocab_w_concept_label.csv"))

if __name__ == "__main__":
    main()