# From Statistical Fidelity to Clinical Consistency: Scalable Generation and Auditing of Synthetic Patient Trajectories

---

## Data

Download the MIMIC-IV v2.2 dataset from [Physionet](https://physionet.org/content/mimiciv/2.2/) and save it to `dataset/mimiciv/2.2`.
The `dataset/mimiciv/2.2` directory should contain `hosp` and `icu` directories.


## Install
1. We need to setup a separate environment to convert MIMIC-IV to [MEDS format](https://github.com/Medical-Event-Data-Standard) first:
```bash
conda create -n meds-etl-env python==3.10.0 -y
conda activate meds-etl-env
git submodule update --init --recursive
pip install -e external/meds_etl
meds_etl_mimic dataset/mimiciv/2.2 dataset/mimiciv/2.2_meds 2.2
```
There'll be a `dataset/mimiciv/2.2_meds` directory containing the MEDS-formatted data. You will then use the data for the following steps in [Quick Start](#quick-start).


2. Setup and change to the main python environment:
```bash
conda create -n coogee python==3.13.0 -y
conda activate coogee
pip install -r requirements.txt
```

---

## Quick Start 

### 1. tokenization
```bash
TIMESTAMP=$(date '+%Y-%m-%d_%H_%M_%S')
LOG_DIR="output/coogee-final-sanity-check/${TIMESTAMP}"
mkdir -p "$LOG_DIR"
mkdir -p "$LOG_DIR/logs"
python -m scripts.process_mimiciv_meds --LOG_DIR $LOG_DIR --icd10cm_top_n -1 --icd10pcs_top_n -1 --atc_top_n -1 --lab_top_n -1
python -m scripts.token_processor_complete \
  --config "configs/coogee.yaml" \
  --LOG_DIR $LOG_DIR
python -m scripts.map_concept_label --LOG_DIR $LOG_DIR
```
The pre-processing and tokenization steps need 80GB RAM. Some files are generated as below:
1. `cohort_stat`: contains the statistics of the cohort;
2. `logs`: `process_mimiciv_meds.log` and `token_processor_complete.log`;
3. `raw_sequences`: contains the untokenized medical event sequences splited into train/val/test;
4. `tokenized_sequences`: contains the tokenized sequences splited into train/val/test;
5. `tokenizer`: contains vocab.csv;

### 2. knowledge-aware representation
We construct two types of knowledge-aware representations for each token in our vocabulary, leveraging external biomedical knowledge sources.


#### 1. some preparation from MedTok
We need to download `code2tokens.json`, `all_codes_mappings.parquet`, `code2embeddings.json`, `kg.csv` and `all_codes_mappings.parquet` from [MedTok](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/7XNT3M) and save them to `artifacts/medtok_files`.
```bash
python -m scripts.map_vocab_to_medtok --LOG_DIR $LOG_DIR
```
<details>
<summary>This script maps each vocabulary code to MedTok concepts.</summary>

- ICD10CM/PCS codes: normalized and matched to the closest MedToK code by prefix/character matching;
- ATC codes: direct mapping with manual corrections for deprecated codes;
- Lab test codes: semantically matched to SNOMED concepts using ClinicalBERT similarity;
- Special tokens: assigned zero embeddings;
</details>

#### 2. construct knowledge-aware embeddings for each token
```bash
python -m scripts.construct_knowledge_embd --LOG_DIR $LOG_DIR
```

<details>
<summary>This script produces two embeddings per token.</summary>

**Hierarchy Embeddings** capture structural relationships from PrimeKG, a comprehensive biomedical knowledge graph:
1. For each medical code, extract its local subgraph from PrimeKG (including connected diseases, drugs, proteins, etc.)
2. Train a Relational Graph Convolutional Network (RGCN) using link prediction loss to learn node representations that preserve graph structure
3. Pool subgraph node embeddings via mean pooling to obtain a single per-code hierarchy embedding

**Semantic Embeddings** capture textual meaning from code descriptions:
1. Look up the natural language description for each medical code (e.g., "Type 2 diabetes mellitus" for ICD-10 E11)
2. Encode descriptions using ClinicalBERT
3. Special tokens use predefined descriptions (e.g., "This token represents the death of the patient" for DEATH)

</details>

The GNN training might take a few hours to complete. Here's the final checkpoint: [artifacts/coogee_final/knowledge_embd/gnn_checkpoint_172.pt](https://drive.google.com/file/d/1iThpwp2Ines0w1QLtrYTxvTX5t7Onb26/view?usp=share_link). 

### 3. training
```bash
RUN_DIR="$LOG_DIR/sweeps/w_knowledge_embd"
n_embd=384
hidden_dim=1024
n_layers=6
n_heads=6
n_kv=2
batch_size=32
python -m model.train \
  --LOG_DIR "$RUN_DIR" \
  --n_embd "$n_embd" \
  --hidden_dim "$hidden_dim" \
  --n_layers "$n_layers" \
  --num_attention_heads "$n_heads" \
  --num_key_value_heads "$n_kv" \
  --batch_size "$batch_size" \
  --wandb_log \
  --use_hierarchy_embd \
  --use_semantic_embd
```
### 4. generation
```bash
python -m model.generate --LOG_DIR "$LOG_DIR" --wandb_log
```

### 5. construct final patient timelines (used for clinical evaluation in our study)
```bash
python -m scripts.construct_final_patient_timelines --LOG_DIR "$LOG_DIR"
python -m scripts.post_process_icd10cm_syn --LOG_DIR "$LOG_DIR"
```

### 6. LLM-as-a-judge exp

```bash
# split in order to use more gpus to process at the same time
python -m scripts.split_synthetic_timelines_into_parts --LOG_DIR "$LOG_DIR"

qsub pbs_cmd/pbs_llm_as_judge/llm_as_judge_1.sh
python -m scripts.compute_realism_stats --LOG_DIR "$LOG_DIR" --NUM 7
```

### 7. evaluation

```bash
python -m evaluation.eval_synthetic --LOG_DIR "$LOG_DIR/synthetic_data_topp_0.98_temperature_1.0_realism_geq_7_w_reasoning" --fidelity --utility --privacy --heatmap
# plot time fidelity in Figure1 g-i
python -m evaluation.eval_fidelity_time --LOG_DIR "$LOG_DIR"
```

## Citation (to be updated)

If you find our work useful in your research, please consider citing:

```tex
@misc{zhou2026statisticalfidelityclinicalconsistency,
      title={From Statistical Fidelity to Clinical Consistency: Scalable Generation and Auditing of Synthetic Patient Trajectories}, 
      author={Guanglin Zhou and Armin Catic and Motahare Shabestari and Matthew Young and Chaiquan Li and Katrina Poppe and Sebastiano Barbieri},
      year={2026},
      eprint={2603.06720},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.06720}, 
}
```

## License
![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)

**Coogee** © 2026 by Guanglin Zhou is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License](https://creativecommons.org/licenses/by-nc/4.0/). 

---


