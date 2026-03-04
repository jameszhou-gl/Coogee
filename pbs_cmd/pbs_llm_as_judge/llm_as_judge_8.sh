#!/bin/bash
#PBS -N patient_timelines_8
#PBS -l select=1:ncpus=6:ngpus=1:mem=100gb:gpu_model=H200
#PBS -l walltime=12:00:00
#PBS -j oe

cd "$PBS_O_WORKDIR"
source /srv/scratch/z3523916/miniconda3/etc/profile.d/conda.sh
conda activate facilitate-meds

echo "Job started at: $(date '+%Y-%m-%d-%H_%M_%S')"

python -m scripts.llm_as_a_judge \
    --PATIENT_TIMELINES_DIR output/coogee-final-sanity-check/2025-12-19_10_58_00-rm_know_emb_labtest_w_n_embd_factor/final_patient_timelines/synthetic_post_processed_icd10cm_8

echo "Job finished at: $(date '+%Y-%m-%d-%H_%M_%S')"