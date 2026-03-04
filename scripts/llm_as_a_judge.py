from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import json
import os
import time
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--PATIENT_TIMELINES_DIR', type=str, required=True, help='Path that stores the final patient timelines')
args = parser.parse_args()

model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

def build_prompt(csv_content: str) -> str:
    return f"""
You are an expert Chief Medical Officer and Data Quality Auditor. You are reviewing a patient record for clinical and logical validity.

### DATA FORMAT
You will be provided with a CSV content representing a patient's timeline.
- **time**: ISO 8601 timestamp. Events are chronological.
- **code**: The raw clinical identifier (e.g., ICD10CM, ICD10PCS, ATC, LAB test, Demographic tags like 'SEX_F').
- **numerical_value**: The result for lab tests. NOTE: This field is empty/NaN for non-lab rows. Ignore empty values.
- **code_label**: Human-readable description of the code.

### EVALUATION CRITERIA
Analyze the record across three dimensions:
1. **Clinical Plausibility**
2. **Logical Consistency**
3. **Temporal Coherence**

### SCORING RUBRIC
1–2: Clearly artificial  
3–4: Largely synthetic  
5–6: Plausible but inconsistent  
7–8: Mostly realistic  
9–10: Indistinguishable from real-world EHR

### PATIENT RECORD from the csv
{csv_content}

### OUTPUT FORMAT
You must respond in the following JSON format and don't output anything else:
{{
"realism_score": <INTEGER 1-10>,
"reasoning": "A brief summary of your analysis, highlighting realistic vs. synthetic elements."
}}
"""

result_json = []
realism_scores = []
loop_start = time.time()
review_files = os.listdir(args.PATIENT_TIMELINES_DIR)
result_json_name = f'{args.PATIENT_TIMELINES_DIR}/Qwen3_30B_with_reasoning.json'

def save_results():
    """Save current results to JSON file"""
    pd.DataFrame(result_json).to_json(result_json_name, orient='records', indent=2)
    print(f"\nSaved {len(result_json)} results to {result_json_name}")

for idx, review_file in enumerate(tqdm(review_files, desc="Processing patient files"), 1):
    csv_content = pd.read_csv(
        f"{args.PATIENT_TIMELINES_DIR}/{review_file}"
    ).to_markdown(index=False)
    prompt = build_prompt(csv_content)
    try:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        model_inputs = tokenizer(
            [text],
            return_tensors="pt"
        ).to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=16384
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        content = tokenizer.decode(
            output_ids,
            skip_special_tokens=True
        )
        parsed = json.loads(content.strip())

        result_json.append({
            "file": review_file,
            "realism_score": parsed["realism_score"],
            "reasoning": parsed['reasoning']
        })
        realism_scores.append(parsed["realism_score"])
        
        # Save every 100 patients
        if idx % 100 == 0:
            save_results()
            
    except Exception as e:
        print(f"Error processing {review_file}: {e}")

# Final save for any remaining results
save_results()
print(f"Realism scores: {realism_scores}")

