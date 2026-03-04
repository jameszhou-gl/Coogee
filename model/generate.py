import os, argparse, random, wandb
from pathlib import Path

import numpy as np
import polars as pl
import torch
from tqdm import tqdm
from model.local_tokenizer import LocalTokenizer

from model.model import Transformer, ModelArgs
from scripts.utils import load_config
SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------------- Dataset ----------------
def load_tokens(parquet_path):
    df = pl.read_parquet(parquet_path)
    print(f"[Test Dataset] loaded {len(df)} sequences")
    flat = np.concatenate(df["concept_token_ids"].to_list()).astype(np.int64)
    print(f"[Test Dataset] loaded {len(flat)} tokens")
    return torch.from_numpy(flat)

# --------------- Utilities ----------------
def get_latest_checkpoint(log_dir):
    ckpt_dir = Path(log_dir) / "checkpoints"
    if not ckpt_dir.exists(): return None
    
    # First check for checkpoint_best.pt
    best_ckpt = ckpt_dir / "checkpoint_best.pt"
    if best_ckpt.exists():
        return str(best_ckpt)
    else:
        return None

# --------------- Testing ----------------
def main():
    # CLI parser
    ap = argparse.ArgumentParser()
    ap.add_argument('--LOG_DIR', type=str, required=True)
    ap.add_argument('--top_p', type=float, default=0.98)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--quick_test', action="store_true", default=False)
    ap.add_argument('--wandb_log', action="store_true", default=False)
    
    args = ap.parse_args()
    
    cfg = load_config(os.path.join(args.LOG_DIR, "config.yaml"))
    seq_cfg = cfg.get('sequence')
    max_len = int(seq_cfg.get('max_seq_length', 2048))
    if args.wandb_log:
        wandb.init(project=cfg['wandb']['project'], config={**cfg, **vars(args)})
    # tokenizer
    tokenizer = LocalTokenizer(os.path.join(args.LOG_DIR, "tokenizer", "vocab.csv"))
    
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        print('CUDA:', torch.cuda.get_device_name())
    
    # ModelArgs
    margs = ModelArgs.from_json(os.path.join(args.LOG_DIR, "checkpoints", "margs.json"))
        
    model = Transformer(margs).to(device)
    print(f"[Model] loaded {sum(p.numel() for p in model.parameters())} parameters")
    
    # data
    test_path = os.path.join(args.LOG_DIR, "tokenized_sequences", "test.parquet")
    if args.quick_test:
        df = pl.read_parquet(test_path)
        df = df[:len(df)//10]
    else:
        df = pl.read_parquet(test_path)
    print(f"[Test Dataset] loaded {len(df)} sequences")
    
    ckpt_path = get_latest_checkpoint(args.LOG_DIR)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"[ckpt] resumed from {ckpt_path}")
    
    model.eval().to(dtype=torch.bfloat16)
    torch.backends.cuda.matmul.allow_tf32 = True
    synthetic_dir = os.path.join(args.LOG_DIR, f"synthetic_data_topp_{args.top_p}_temperature_{args.temperature}")
    os.makedirs(synthetic_dir, exist_ok=True)
    synthetic_path = os.path.join(synthetic_dir, "synthetic_data.parquet")
    # Resume logic
    if os.path.exists(synthetic_path):
        existing_df = pl.read_parquet(synthetic_path)
        done_subjects = set(existing_df["subject_id"].to_list())
        print(f"[Resume] Found existing synthetic data with {len(done_subjects)} subjects")
    else:
        existing_df = None
        done_subjects = set()
    
    gen_seqs = []
    decoded_seqs = []
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for idx, row in enumerate(tqdm(df.iter_rows(), total=len(df), desc="Generating sequences")):
            # Get column indices first
            subject_id_idx = df.get_column_index("subject_id")
            concept_ids_idx = df.get_column_index("concept_token_ids")
            
            # Access columns by index
            subject_id = row[subject_id_idx]
            if subject_id in done_subjects:
                continue  # skip if already generated
            concept_ids = torch.tensor(row[concept_ids_idx], dtype=torch.long, device=device).unsqueeze(0)
            # Take only the first 6 tokens
            input_demographic = concept_ids[:, :6]
            
            gen_seq = model.generate(
                input_demographic,
                max_length=max_len,
                temperature=args.temperature,
                top_p=args.top_p,
                end_token_id=tokenizer.eos_token_id,
                tokenizer=tokenizer
            ).squeeze(0).tolist()
            
            gen_seqs.append({
                "subject_id": subject_id,
                "synthetic_sequence": gen_seq
            })
            if idx <10:
                decoded_seqs.append(f"{subject_id}:\n")
                decoded_test_seq = tokenizer.decode(concept_ids.squeeze(0).tolist())
                decoded_seqs.append(f"ground truth:\n{decoded_test_seq}\n")
                decoded_gen_seq = tokenizer.decode(gen_seq)
                decoded_seqs.append(f"generated:\n{decoded_gen_seq}\n")
            
            # Save every 100 patients
            if len(gen_seqs) >= 100:
                new_df = pl.DataFrame(gen_seqs)
                if existing_df is not None:
                    existing_df = pl.concat([existing_df, new_df])
                else:
                    existing_df = new_df
                existing_df.write_parquet(synthetic_path)
                print(f"[Synthetic Data] Saved {len(existing_df)} sequences so far.")
                
                # reset buffer
                gen_seqs = []

    # Final flush (if any left)
    if gen_seqs:
        new_df = pl.DataFrame(gen_seqs)
        if existing_df is not None:
            existing_df = pl.concat([existing_df, new_df])
        else:
            existing_df = new_df
        existing_df.write_parquet(synthetic_path)
        print(f"[Final Save] Written {len(existing_df)} sequences total.")

        txt_path = os.path.join(synthetic_dir, "generated_sequences.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(decoded_seqs))
            f.write("\n")
        
if __name__ == "__main__":
    main()