import os, argparse, random, wandb, math
from pathlib import Path

import numpy as np
import polars as pl
import torch
from tqdm import tqdm
from model.local_tokenizer import LocalTokenizer

from model.model import Transformer, ModelArgs
from scripts.utils import load_config
# import pdb; pdb.set_trace()
SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------------- Dataset ----------------
def load_tokens(parquet_path, split_name):
    df = pl.read_parquet(parquet_path)
    print(f"[{split_name} Dataset] loaded {len(df):,} sequences")
    flat = np.concatenate(df["concept_token_ids"].to_list()).astype(np.int64)
    print(f"[{split_name} Dataset] loaded {len(flat):,} tokens")
    return torch.from_numpy(flat)

def get_batch(data, block_size, batch_size, device):
    # data: 1D tensor of tokens
    N = data.size(0)
    ix = torch.randint(low=0, high=N - block_size - 1, size=(batch_size,))
    x = torch.stack([data[i : i+block_size] for i in ix])
    y = torch.stack([data[i+1 : i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

def save_checkpoint(model, optim, sched, epoch, tokens_seen, log_dir):
    ckpt_dir = Path(log_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # keep only last
    for f in ckpt_dir.glob("checkpoint_*.pt"): f.unlink()
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "scheduler_state_dict": sched.state_dict(),
        "tokens_seen": tokens_seen,
    }, ckpt_dir / f"checkpoint_best.pt")
    print(f"[ckpt] saved epoch {epoch}, tokens_seen {tokens_seen:,}")
        
# --------------- Training ----------------
def main():
    # CLI parser
    ap = argparse.ArgumentParser()
    ap.add_argument('--LOG_DIR', type=str, required=True)
    ap.add_argument('--n_embd', type=int, default=384)
    ap.add_argument('--n_layers', type=int, default=6)
    ap.add_argument('--hidden_dim', type=int, default=768)
    ap.add_argument('--num_attention_heads', type=int, default=9)
    ap.add_argument('--num_key_value_heads', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--target_tokens', type=float, default=None,
                help='Total training tokens budget (e.g., 2.1035e10). Overrides 200-epochs if set')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--patience', type=int, default=10,
                help='stop if val loss does not improve for N epochs')
    ap.add_argument('--wandb_log', action="store_true")
    ap.add_argument('--wandb_name', type=str, default=None)
    ap.add_argument('--debug', action="store_true")
    ap.add_argument('--use_hierarchy_embd', action="store_true")
    ap.add_argument('--use_semantic_embd', action="store_true")
    ap.add_argument('--n_embd_factor', type=int, default=None)
    args = ap.parse_args()
    
    cfg = load_config(os.path.join(args.LOG_DIR, "config.yaml"))
    seq_cfg = cfg.get('sequence')
    # tokenizer
    tokenizer = LocalTokenizer(os.path.join(args.LOG_DIR, "tokenizer", "vocab.csv"))
    
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        print('CUDA:', torch.cuda.get_device_name())
        
    # ModelArgs
    margs = ModelArgs(LOG_DIR=args.LOG_DIR)  # Initialize with required LOG_DIR
    margs.n_embd = args.n_embd
    margs.n_embd_factor = args.n_embd_factor
    margs.n_layers = args.n_layers
    margs.hidden_dim = args.hidden_dim
    margs.num_attention_heads = args.num_attention_heads
    margs.num_key_value_heads = args.num_key_value_heads
    margs.drop_out = args.dropout
    n_ctx = int(seq_cfg.get('max_seq_length', 2048))
    margs.n_ctx = n_ctx
    margs.vocab_size = tokenizer.vocab_size
    margs.use_hierarchy_embd = cfg['aux_embeddings']['hierarchy']['enabled'] if args.use_hierarchy_embd else False
    margs.hierarchy_dim = cfg['aux_embeddings']['hierarchy']['dim']
    margs.use_semantic_embd = cfg['aux_embeddings']['semantic']['enabled'] if args.use_semantic_embd else False
    margs.semantic_dim = cfg['aux_embeddings']['semantic']['dim']
    if tokenizer.pad_token_id is not None: margs.pad_token_id = tokenizer.pad_token_id
    if tokenizer.bos_token_id is not None: margs.bos_token_id = tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None: margs.eos_token_id = tokenizer.eos_token_id
    margs_path = os.path.join(args.LOG_DIR, "checkpoints", "margs.json")
    os.makedirs(os.path.dirname(margs_path), exist_ok=True)
    margs.to_json(margs_path)
    if args.wandb_log:
        wandb_config = {**vars(args), **cfg, **vars(margs)}
        if args.wandb_name:
            wandb.init(project=cfg.get('wandb').get('project'), config=wandb_config, name=args.wandb_name)
        else:
            wandb.init(project=cfg.get('wandb').get('project'), config=wandb_config)
        
    model = Transformer(margs).to(device=device, dtype=torch.bfloat16)
    # optional but recommeded on NVIDIA
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    scaler = None # ! do not use GradScaler for bf16
    
    model.print_model_params()
    # data
    train_path = os.path.join(args.LOG_DIR, "tokenized_sequences", "train.parquet")
    val_path = os.path.join(args.LOG_DIR, "tokenized_sequences", "val.parquet")
    train_data = load_tokens(train_path, 'Train').to(device)
    val_data   = load_tokens(val_path, 'Val').to(device)
    
    optim = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1
    )
    # Compute steps
    tokens_per_step = args.batch_size * n_ctx
    steps_per_epoch = max(1, len(train_data) // (args.batch_size * n_ctx))
    if args.target_tokens:
        print(f"[Train] Using target tokens: {args.target_tokens:,}")
        target_steps = int(math.ceil(float(args.target_tokens) / tokens_per_step))
    else:
        print(f"[Train] Using 200 epochs to set target steps")
        target_steps = int(math.ceil(200 * steps_per_epoch))
    print(f"[Train] target_steps: {target_steps:,}")
    warmup_steps = max(1, int(0.01 * target_steps))  # ~1% warmup
    # Chinchilla-style warmup + cosine to 0.1x
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            # linear warmup to 1.0
            return float(current_step) / float(warmup_steps)
        # cosine decay from 1.0 -> 0.1
        progress = (current_step - warmup_steps) / max(1, (target_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    # resume
    epochs_done, best_loss, global_step, tokens_seen = 0, float('inf'), 0, 0
    epochs_no_improve = 0
    ckpt_path = os.path.join(args.LOG_DIR, "checkpoints", "checkpoint_best.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optim.load_state_dict(ckpt['optimizer_state_dict'])
        epochs_done = ckpt['epoch']
        tokens_seen = ckpt['tokens_seen']
        global_step = epochs_done * steps_per_epoch
        print(f"[ckpt] resumed from {ckpt_path}, epoch {epochs_done}, step {global_step:,}, tokens_seen {tokens_seen:,}")
    
    # Initialize scheduler with current global_step to match optimizer state
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda, last_epoch=global_step-1)
    # Manually set the last_epoch to avoid the initial step
    sched._step_count = global_step
    sched._last_lr = [lr_lambda(global_step - 1) * pg['lr'] for pg in optim.param_groups]
    if os.path.exists(ckpt_path) and 'scheduler_state_dict' in ckpt:
        sched.load_state_dict(ckpt['scheduler_state_dict'])
    
    print("[Train] Start")
    pbar = tqdm(total=target_steps, desc="Training (by token budget)", initial=global_step)
    while global_step < target_steps:
        model.train()
        train_loss = 0.0
        steps_this_epoch = min(steps_per_epoch, target_steps - global_step)
        for _ in range(steps_this_epoch):
            train_x, train_y = get_batch(train_data, n_ctx, args.batch_size, device)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(train_x)  # [B, L, V]
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    train_y.reshape(-1),
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step()
            
            global_step += 1
            tokens_seen += tokens_per_step
            train_loss += loss.item()    
            pbar.update(1)      
            
            if global_step >= target_steps:
                break   
        epochs_done += 1
        train_loss /= max(1, steps_per_epoch)
        if args.wandb_log:
            wandb.log({
                "cuda/alloc_GB": torch.cuda.memory_allocated(device) / 1024**3,
                "cuda/reserved_GB": torch.cuda.memory_reserved(device) / 1024**3,
                "cuda/max_allocated_GB": torch.cuda.max_memory_allocated(device) / 1024**3,
                "cuda/max_reserved_GB": torch.cuda.max_memory_reserved(device) / 1024**3,
            }, step=global_step)
        # validation
        model.eval()
        val_loss, val_steps = 0.0, max(1, len(val_data) // (args.batch_size*n_ctx))
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for _ in range(val_steps):
                val_x, val_y = get_batch(val_data, n_ctx, args.batch_size, device)
                val_logits = model(val_x)
                loss = torch.nn.functional.cross_entropy(
                    val_logits.reshape(-1, val_logits.size(-1)),
                    val_y.reshape(-1),
                )
                val_loss += loss.item()
        val_loss /= val_steps
        if args.wandb_log:
            wandb.log({
                "epoch_float": epochs_done-1+steps_this_epoch/steps_per_epoch,
                "global_step": global_step,
                "tokens_seen": tokens_seen,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": optim.param_groups[0]['lr']
            })
            alpha_h, alpha_s = model.print_alpha_values()
            wandb.log({
                "alpha_h": alpha_h,
                "alpha_s": alpha_s
            })
        # ----- early stopping on val loss -----
        improved = val_loss < best_loss - 1e-6  # small tolerance to avoid float noise
        if improved:
            best_loss = val_loss
            epochs_no_improve = 0
            print(f"[Train] Val loss improved to {val_loss:.4f}")
            save_checkpoint(model, optim, sched, epochs_done, tokens_seen, args.LOG_DIR)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping: no val improvement for {args.patience} epochs "
                    f"(best val loss={best_loss:.4f} at step {global_step}/{target_steps})")
                break

    pbar.close()
    print(f"[Done] steps={global_step}/{target_steps}  tokens_seen≈{tokens_seen:,}")
        
if __name__ == "__main__":
    main()