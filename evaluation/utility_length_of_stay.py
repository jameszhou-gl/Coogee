# evaluation/utility_length_of_stay.py

import os
import json
import math
import random
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from sklearn import metrics
from evaluation.utils import plot_real_vs_syn

SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Hyperparameters
# ----------------------------
LR = 1e-3
EPOCHS = 40
BATCH_SIZE = 512
EMBED_DIM = 128
HIDDEN_DIM = 256
DROPOUT = 0.5
N_CTX = 1                 # single visit per sample (current visit only)
NUM_TRAIN_EXAMPLES = 5000
NUM_VAL_EXAMPLES   =  500
NUM_TEST_EXAMPLES  = 1000
NUM_CLASSES = 10         # LOS categories 0..9

# ----------------------------
# Helpers: compact vocab over Dx/Proc/Drug
# ----------------------------
def build_dx_proc_drug_vocab(tokenizer) -> Tuple[Dict[int, int], int]:
    keep_prefixes = ("ICD10CM_", "ICD10PCS_", "ATC_")
    tok2idx: Dict[int, int] = {}
    nxt = 0
    for tid in range(tokenizer.vocab_size):
        try:
            t = tokenizer.decode([tid])
        except Exception:
            continue
        if any(t.startswith(p) for p in keep_prefixes):
            tok2idx[tid] = nxt; nxt += 1
    return tok2idx, nxt

# ----------------------------
# Time-gap mapping (minutes)
# ----------------------------
# We approximate with **bin midpoints** (in minutes). Adjust if you prefer lower/upper bounds.
_TIME_BIN_MINS = {
    "_<=5m":     2.5,
    "_5m-15m":   10.0,
    "_15m-1h":   37.5,
    "_1h-2h":    90.0,
    "_2h-6h":    240.0,
    "_6h-12h":   540.0,
    "_12h-1d":   1080.0,      # 18h midpoint
    "_1d-3d":    2.0 * 24 * 60,
    "_3d-1w":    5.0 * 24 * 60,
    "_1w-2w":    10.5 * 24 * 60,
    "_2w-1mt":   21.5 * 24 * 60,
    "_1mt-3mt":  61.0 * 24 * 60,   # months approximated as 30.5d
    "_3mt-6mt":  106.0 * 24 * 60,
    "_>6mt":     180.0 * 24 * 60,  # conservative large value
}

def _decode_cache(tokenizer):
    cache = {}
    def dec(tid: int) -> str:
        if tid not in cache:
            cache[tid] = tokenizer.decode([tid])
        return cache[tid]
    return dec

def _minutes_to_los_category(total_minutes: float) -> int:
    """Buckets into 0..9 following PyHealth/Harutyunyan et al."""
    days = total_minutes / (60.0 * 24.0)
    if days < 1:  # < 1 day
        return 0
    if 1 <= days <= 7:
        # 1..7 is integer day
        # floor to nearest day in [1..7]
        d = int(math.floor(days + 1e-6))
        return min(max(d, 1), 7)
    if 7 < days <= 14:
        return 8
    return 9  # > 14 days

# ----------------------------
# Sample construction
# ----------------------------
def construct_los_samples(
    structured_data: List[Dict],
    tokenizer,
    tok2compact: Dict[int, int],
) -> List[Dict]:
    """
    Build per-visit samples:
      X = BoW of Dx/Proc/Drug from the current visit only
      y = LOS category inferred by summing time-gap tokens inside the same visit
    """
    dec = _decode_cache(tokenizer)
    samples: List[Dict] = []

    for patient in structured_data:
        visits: List[List[int]] = patient.get("visits", [])
        for v in visits:
            # ---- features: current visit only (Dx/Proc/Drug) ----
            bow_indices = []
            total_mins = 0.0

            for tid in v:
                # accumulate time gaps (inside visit)
                tok = dec(int(tid))
                if tok in _TIME_BIN_MINS:
                    total_mins += _TIME_BIN_MINS[tok]
                # keep Dx/Proc/Drug as features
                if tid in tok2compact:
                    bow_indices.append(tok2compact[tid])

            if not bow_indices:
                continue  # skip visits without any Dx/Proc/Drug codes

            y = _minutes_to_los_category(total_mins)

            # deduplicate within visit for BoW
            bow_indices = sorted(set(bow_indices))
            samples.append({
                "visit_codes": bow_indices,
                "label": y
            })

    return samples

# ----------------------------
# Model: simple MLP on visit BoW
# ----------------------------
class VisitMLP(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.emb = nn.Linear(vocab_size, EMBED_DIM, bias=False)
        self.dropout = nn.Dropout(DROPOUT)
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(EMBED_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, num_classes),
        )

    def forward(self, x_bow):
        # x_bow: [B, V]
        h = self.emb(x_bow)
        h = self.dropout(h)
        logits = self.mlp(h)
        return logits

# ----------------------------
# Batching / training / eval
# ----------------------------
def visits_to_bow(samples: List[Dict], vocab_size: int, start: int, bs: int):
    batch = samples[start:start+bs]
    x = np.zeros((len(batch), vocab_size), dtype=np.float32)
    y = np.array([b["label"] for b in batch], dtype=np.int64)
    for i, s in enumerate(batch):
        for idx in s["visit_codes"]:
            x[i, idx] = 1.0
    return x, y

def _class_hist(samples: List[Dict]) -> Dict[int, int]:
    hist = {k: 0 for k in range(NUM_CLASSES)}
    for s in samples:
        hist[s["label"]] += 1
    return hist

def _stratified_subset(samples: List[Dict], n_total: int) -> List[Dict]:
    if n_total is None or len(samples) <= n_total:
        return samples
    by_c = {k: [] for k in range(NUM_CLASSES)}
    for s in samples:
        by_c[s["label"]].append(s)
    k = n_total // NUM_CLASSES
    out = []
    for c in range(NUM_CLASSES):
        pool = by_c[c]
        if len(pool) == 0:
            continue
        take = np.random.choice(pool, k, replace=(len(pool) < k)).tolist()
        out.extend(take)
    random.shuffle(out)
    return out

def train(vocab_size: int, model: nn.Module, train_s: List[Dict], val_s: List[Dict], save_path: str):
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best = float("inf"); best_state = None

    for _ in tqdm(range(EPOCHS)):
        np.random.shuffle(train_s)
        # train
        model.train()
        losses = []
        for i in range(0, len(train_s), BATCH_SIZE):
            xb, yb = visits_to_bow(train_s, vocab_size, i, BATCH_SIZE)
            xb = torch.tensor(xb, device=device)
            yb = torch.tensor(yb, device=device)
            opt.zero_grad()
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()
            losses.append(loss.detach().cpu().item())
        # val
        model.eval()
        with torch.no_grad():
            vlosses = []
            for i in range(0, len(val_s), BATCH_SIZE):
                xb, yb = visits_to_bow(val_s, vocab_size, i, BATCH_SIZE)
                xb = torch.tensor(xb, device=device)
                yb = torch.tensor(yb, device=device)
                logits = model(xb)
                vloss = ce(logits, yb)
                vlosses.append(vloss.detach().cpu().item())
        cur = float(np.mean(vlosses)) if vlosses else float("inf")
        if cur < best:
            best = cur
            best_state = {"model": model.state_dict(), "opt": opt.state_dict()}
            torch.save(best_state, save_path)

    if best_state is not None:
        model.load_state_dict(best_state["model"])

def evaluate(vocab_size: int, model: nn.Module, test_s: List[Dict]) -> Dict:
    model.eval()
    ce = nn.CrossEntropyLoss()
    losses = []; preds = []; trues = []
    with torch.no_grad():
        for i in range(0, len(test_s), BATCH_SIZE):
            xb, yb = visits_to_bow(test_s, vocab_size, i, BATCH_SIZE)
            xb = torch.tensor(xb, device=device)
            yb = torch.tensor(yb, device=device)
            logits = model(xb)
            loss = ce(logits, yb)
            losses.append(loss.detach().cpu().item())
            pr = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            preds.append(pr)
            trues.append(yb.detach().cpu().numpy())
    P = np.concatenate(preds, axis=0)
    y = np.concatenate(trues, axis=0)

    yhat = np.argmax(P, axis=1)
    report = {
        "Test Loss": float(np.mean(losses)) if losses else float("nan"),
        "Accuracy": metrics.accuracy_score(y, yhat),
        "Precision": metrics.precision_score(y, yhat, average='macro', zero_division=0),
        "Recall": metrics.recall_score(y, yhat, average='macro', zero_division=0),
        "F1_score": metrics.f1_score(y, yhat, average='macro', zero_division=0),
        "ROC_AUC": metrics.roc_auc_score(y, P, multi_class='ovr', average='macro') if (y.min() != y.max()) else float("nan"),
    }
    # --- NEW: 95% CI for ROC_AUC via bootstrap over test examples ---
    def _auc_ci_bootstrap(y_true, prob, n_boot=2000, seed=1337):
        rng = np.random.RandomState(seed)
        N = len(y_true); aucs = []
        for _ in range(n_boot):
            idx = rng.randint(0, N, size=N)  # sample indices with replacement
            y_b = y_true[idx]
            P_b = prob[idx]
            # skip if bootstrap sample collapses to 1 class
            if len(np.unique(y_b)) < 2:
                continue
            try:
                auc = metrics.roc_auc_score(y_b, P_b, multi_class='ovr', average='macro')
                aucs.append(auc)
            except Exception:
                # rare numerical issues; skip this draw
                continue
        if len(aucs) == 0:
            return (float("nan"), float("nan"))
        lo = float(np.percentile(aucs, 2.5))
        hi = float(np.percentile(aucs, 97.5))
        return (lo, hi)

    report["ROC_AUC_CI"] = _auc_ci_bootstrap(y, P)
    
    return report

# ----------------------------
# Public entry point
# ----------------------------
def utility_length_of_stay_pred(
    tokenizer,
    structured_train_data,
    structured_val_data,
    structured_test_data,
    structured_syn_data,
    synthetic_data_dir,
):
    """
    TSTR (train-on-synth, test-on-real) with validation on real-val.
    Also trains a baseline on real for reference.
    """
    out_dir = os.path.join(synthetic_data_dir, "utility_models")
    os.makedirs(out_dir, exist_ok=True)

    # build compact vocab over Dx/Proc/Drug
    tok2compact, V = build_dx_proc_drug_vocab(tokenizer)

    # construct per-visit samples
    syn_samples  = construct_los_samples(structured_syn_data,  tokenizer, tok2compact)
    tr_samples   = construct_los_samples(structured_train_data, tokenizer, tok2compact)
    val_samples  = construct_los_samples(structured_val_data,   tokenizer, tok2compact)
    test_samples = construct_los_samples(structured_test_data,  tokenizer, tok2compact)

    # debug histograms
    def _print_hist(name, s):
        hist = _class_hist(s)
        print(f"[{name}] total={len(s)} class_hist={hist}")

    # _print_hist("train_real_raw", tr_samples)
    # _print_hist("train_syn_raw",  syn_samples)
    # _print_hist("val_raw",        val_samples)
    # _print_hist("test_raw",       test_samples)

    # balanced/limited subsets (stratified across 10 classes)
    tr_real  = _stratified_subset(tr_samples,  NUM_TRAIN_EXAMPLES)
    tr_syn   = _stratified_subset(syn_samples, NUM_TRAIN_EXAMPLES)
    val_set  = _stratified_subset(val_samples, NUM_VAL_EXAMPLES)
    test_set = _stratified_subset(test_samples, NUM_TEST_EXAMPLES)

    # _print_hist("train_real", tr_real)
    # _print_hist("train_syn",  tr_syn)
    # _print_hist("val",        val_set)
    # _print_hist("test",       test_set)

    # --- Train on REAL ---
    print("Training LOS model on real...")
    m_real = VisitMLP(V).to(device)
    ckpt_real = os.path.join(out_dir, "los_real.pt")
    if not os.path.exists(ckpt_real):
        train(V, m_real, tr_real, val_set, ckpt_real)
    state = torch.load(ckpt_real, map_location=device)
    m_real.load_state_dict(state["model"])
    res_real = evaluate(V, m_real, test_set)

    # --- Train on SYNTHETIC (TSTR) ---
    print("Training LOS model on synthetic (TSTR)...")
    m_syn = VisitMLP(V).to(device)
    ckpt_syn = os.path.join(out_dir, "los_syn.pt")
    if not os.path.exists(ckpt_syn):
        train(V, m_syn, tr_syn, val_set, ckpt_syn)
    state = torch.load(ckpt_syn, map_location=device)
    m_syn.load_state_dict(state["model"])
    res_syn = evaluate(V, m_syn, test_set)

    results = {
        "counts": {
            "Vocab": V,
            "train_real": len(tr_real),
            "train_syn": len(tr_syn),
            "val": len(val_set),
            "test": len(test_set),
        },
        "Real": res_real,
        "Syn":  res_syn,
    }
    plot_real_vs_syn(synthetic_data_dir, results, task_name="Length-of-stay prediction")
    return results