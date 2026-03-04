# evaluation/utility_readmission.py
import os
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm
from sklearn import metrics
from evaluation.utils import plot_real_vs_syn
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ----------------------------
# Seeds / device
# ----------------------------
SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Hyperparams (aligned to your other utilities)
# ----------------------------
LR = 1e-3
EPOCHS = 50
BATCH_SIZE = 512
N_CTX = 48              # max visits per sample
EMBEDDING_DIM = 64
LSTM_HIDDEN_DIM = 32

NUM_TRAIN_EXAMPLES = 5000
NUM_VAL_EXAMPLES   =  500
NUM_TEST_EXAMPLES  = 1000

# ----------------------------
# Compact vocab over Dx/Proc/Drug
# ----------------------------
def build_dx_proc_drug_vocab(tokenizer) -> Tuple[Dict[int, int], int]:
    """Map token IDs whose decoded text starts with ICD10CM_, ICD10PCS_, or ATC_ to a compact [0..V-1] index."""
    keep = ("ICD10CM_", "ICD10PCS_", "ATC_")
    tok2idx: Dict[int, int] = {}
    nxt = 0
    for tid in range(tokenizer.vocab_size):
        try:
            t = tokenizer.decode([tid])
        except Exception:
            continue
        if any(t.startswith(p) for p in keep):
            tok2idx[tid] = nxt
            nxt += 1
    return tok2idx, nxt

# ----------------------------
# Build readmission samples
# ----------------------------
def construct_readmission_samples(
    structured_data: List[Dict],
    tok2compact: Dict[int, int],
    sigma_days: int = 15,
) -> List[Dict]:
    """
    Build visit-level readmission samples consistent with:
        f: (v1, ..., v_{t-1}) -> y[ τ(v_t) - τ(v_{t-1}) ],
    where y=1 iff τ(v_t) - τ(v_{t-1}) <= σ (σ in days).
    We use patient["gap_hours"][t] as τ(v_t) - τ(v_{t-1}) in hours.
    """
    sigma_hours = float(sigma_days * 24)
    samples: List[Dict] = []

    for p in structured_data:
        visits_raw: List[List[int]] = p.get("visits", [])
        gaps_hours: List[Optional[float]] = p.get("gap_hours", [])
        if not visits_raw or len(visits_raw) != len(gaps_hours):
            continue  # malformed

        # Convert each visit to compact bag of codes (Dx/Proc/Drug only)
        visits_compact: List[List[int]] = []
        for v in visits_raw:
            bag = [tok2compact[tid] for tid in v if tid in tok2compact]
            if bag:
                visits_compact.append(sorted(set(bag)))  # de-dup within visit
            else:
                visits_compact.append([])

        T = len(visits_compact)
        # We can form samples for t = 1..T-1 (since gap_hours[0] is None)
        for t in range(1, T):
            # history excludes v_t itself
            history = [visits_compact[k] for k in range(0, t) if len(visits_compact[k]) > 0]
            if len(history) == 0:
                continue
            gap = gaps_hours[t]  # hours between v_{t-1} and v_t
            y = 1 if (gap is not None and gap <= sigma_hours) else 0
            samples.append({"visits": history, "label": y})

    return samples

# ----------------------------
# Model (BiLSTM over visit BoWs)
# ----------------------------
class VisitLSTM(nn.Module):
    def __init__(self, code_vocab_size: int):
        super().__init__()
        self.embedding = nn.Linear(code_vocab_size, EMBEDDING_DIM, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=LSTM_HIDDEN_DIM,
            num_layers=2,
            dropout=0.5,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(2 * LSTM_HIDDEN_DIM, 1)

    def forward(self, visit_bow_seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # visit_bow_seq: [B, T, V] (bag-of-codes per visit)
        x = self.embedding(visit_bow_seq)          # [B, T, D]
        x = self.dropout(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)  # [B, T, 2H]

        # forward last step + reverse first step pooling
        idx = (lengths - 1).long()
        out_fwd = out[torch.arange(out.size(0)), idx, :LSTM_HIDDEN_DIM]
        out_rev = out[:, 0, LSTM_HIDDEN_DIM:]
        h = torch.cat([out_fwd, out_rev], dim=-1)
        logits = self.fc(h).squeeze(-1)
        prob = torch.sigmoid(logits)
        return prob

# ----------------------------
# Batching / train / eval
# ----------------------------
def visits_to_bow_tensor(samples, vocab_size, start, batch_size):
    batch = samples[start:start + batch_size]
    bows = np.zeros((len(batch), N_CTX, vocab_size), dtype=np.float32)
    lens = np.zeros(len(batch), dtype=np.int64)
    labels = np.array([b["label"] for b in batch], dtype=np.float32)
    for i, s in enumerate(batch):
        seq = s["visits"][:N_CTX]
        lens[i] = min(len(seq), N_CTX)
        for t, visit in enumerate(seq[:N_CTX]):
            for c in visit:
                bows[i, t, c] = 1.0
    return bows, labels, lens

def train_model(vocab_size, model, train_samples, val_samples, save_path):
    bce = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best = float("inf"); best_state = None

    for _ in tqdm(range(EPOCHS)):
        np.random.shuffle(train_samples)
        # train
        model.train(); losses = []
        for i in range(0, len(train_samples), BATCH_SIZE):
            bows, labels, lens = visits_to_bow_tensor(train_samples, vocab_size, i, BATCH_SIZE)
            bows = torch.tensor(bows, device=device)
            labels = torch.tensor(labels, device=device)
            lens_t = torch.tensor(lens)
            opt.zero_grad()
            prob = model(bows, lens_t)
            loss = bce(prob, labels)
            loss.backward()
            opt.step()
            losses.append(loss.detach().cpu().item())

        # validate
        model.eval(); vlosses = []
        with torch.no_grad():
            for i in range(0, len(val_samples), BATCH_SIZE):
                bows, labels, lens = visits_to_bow_tensor(val_samples, vocab_size, i, BATCH_SIZE)
                bows = torch.tensor(bows, device=device)
                labels = torch.tensor(labels, device=device)
                lens_t = torch.tensor(lens)
                prob = model(bows, lens_t)
                vloss = bce(prob, labels)
                vlosses.append(vloss.detach().cpu().item())
        cur = float(np.mean(vlosses)) if vlosses else float("inf")
        if cur < best:
            best = cur
            best_state = {"model": model.state_dict(), "opt": opt.state_dict()}
            torch.save(best_state, save_path)

    if best_state is not None:
        model.load_state_dict(best_state["model"])

def test_model(vocab_size, model, test_samples):
    model.eval()
    bce = nn.BCELoss()
    losses, probs, trues = [], [], []
    with torch.no_grad():
        for i in range(0, len(test_samples), BATCH_SIZE):
            bows, labels, lens = visits_to_bow_tensor(test_samples, vocab_size, i, BATCH_SIZE)
            bows = torch.tensor(bows, device=device)
            labels = torch.tensor(labels, device=device)
            lens_t = torch.tensor(lens)
            prob = model(bows, lens_t)
            loss = bce(prob, labels)
            losses.append(loss.detach().cpu().item())
            probs.extend(prob.detach().cpu().numpy().tolist())
            trues.extend(labels.detach().cpu().numpy().tolist())

    y = np.array(trues).astype(int)
    pred = np.array(probs)
    yhat = (pred >= 0.5).astype(int)

    out = {
        "Test Loss": float(np.mean(losses)) if losses else float("nan"),
        "Confusion Matrix": metrics.confusion_matrix(y, yhat),
        "Accuracy": metrics.accuracy_score(y, yhat),
        "Precision": metrics.precision_score(y, yhat, zero_division=0),
        "Recall": metrics.recall_score(y, yhat, zero_division=0),
        "F1_score": metrics.f1_score(y, yhat, zero_division=0),
        "ROC_AUC": metrics.roc_auc_score(y, pred) if (y.min() != y.max()) else float("nan"),
    }
    pr, rc, _ = metrics.precision_recall_curve(y, pred)
    out["AUPRC"] = metrics.auc(rc, pr)

    # --- NEW: 95% CI for ROC_AUC via bootstrap over test examples ---
    def _auc_ci_bootstrap_binary(y_true, y_score, n_boot=2000, seed=1337):
        rng = np.random.RandomState(seed)
        N = len(y_true); aucs = []
        if len(np.unique(y_true)) < 2:
            return (float("nan"), float("nan"))
        for _ in range(n_boot):
            idx = rng.randint(0, N, size=N)  # sample with replacement
            y_b = y_true[idx]
            s_b = y_score[idx]
            # skip draws that collapse to a single class
            if len(np.unique(y_b)) < 2:
                continue
            try:
                aucs.append(metrics.roc_auc_score(y_b, s_b))
            except Exception:
                continue
        if not aucs:
            return (float("nan"), float("nan"))
        lo = float(np.percentile(aucs, 2.5))
        hi = float(np.percentile(aucs, 97.5))
        return (lo, hi)

    out["ROC_AUC_CI"] = _auc_ci_bootstrap_binary(y, pred)

    return out

# ----------------------------
# TSTR entry point
# ----------------------------
def utility_readmission_pred(
    tokenizer,
    structured_train_data: List[Dict],
    structured_val_data: List[Dict],
    structured_test_data: List[Dict],
    structured_syn_data: List[Dict],
    synthetic_data_dir: str,
    sigma_days: int = 15,
):
    """
    Build visit-level readmission samples with σ days (default 15),
    train on synthetic (validate on real-val), and test on real-test.
    Also train a real-only baseline for comparison.
    """
    utility_dir = os.path.join(synthetic_data_dir, "utility_models")
    os.makedirs(utility_dir, exist_ok=True)

    # Compact vocab (Dx/Proc/Drug)
    tok2compact, V = build_dx_proc_drug_vocab(tokenizer)

    # Samples
    syn_samples  = construct_readmission_samples(structured_syn_data,  tok2compact, sigma_days)
    tr_samples   = construct_readmission_samples(structured_train_data, tok2compact, sigma_days)
    val_samples  = construct_readmission_samples(structured_val_data,   tok2compact, sigma_days)
    test_samples = construct_readmission_samples(structured_test_data,  tok2compact, sigma_days)

    # Filter empties
    syn_samples  = [s for s in syn_samples  if len(s["visits"]) > 0]
    tr_samples   = [s for s in tr_samples   if len(s["visits"]) > 0]
    val_samples  = [s for s in val_samples  if len(s["visits"]) > 0]
    test_samples = [s for s in test_samples if len(s["visits"]) > 0]

    # Stratified balancing (50/50) like your other utilities
    def stratified_subset(samples, n_total, tag="set"):
        if n_total is None or len(samples) <= n_total:
            return samples
        pos = [s for s in samples if s["label"] == 1]
        neg = [s for s in samples if s["label"] == 0]
        k = n_total // 2
        pos_pick = np.random.choice(pos, k, replace=(len(pos) < k)).tolist() if pos else []
        neg_pick = np.random.choice(neg, k, replace=(len(neg) < k)).tolist()
        out = pos_pick + neg_pick
        random.shuffle(out)
        # print(f"[{tag}] total={len(samples)} → subset={len(out)} "
        #       f"(pos avail={len(pos)}, neg avail={len(neg)}, "
        #       f"pos picked={len(pos_pick)}, neg picked={len(neg_pick)}, "
        #       f"replace_pos={len(pos) < k}, replace_neg={len(neg) < k})")
        return out

    tr_real  = stratified_subset(tr_samples,  NUM_TRAIN_EXAMPLES, "train_real")
    tr_syn   = stratified_subset(syn_samples, NUM_TRAIN_EXAMPLES, "train_syn")
    val_set  = stratified_subset(val_samples, NUM_VAL_EXAMPLES,   "val")
    test_set = stratified_subset(test_samples, NUM_TEST_EXAMPLES, "test")

    # --- Train baseline on REAL ---
    print("Training readmission model on real...")
    m_real = VisitLSTM(V).to(device)
    ck_real = os.path.join(utility_dir, f"readmit_real_sigma{sigma_days}.pt")
    if not os.path.exists(ck_real):
        train_model(V, m_real, tr_real, val_set, ck_real)
    state = torch.load(ck_real, map_location=device)
    m_real.load_state_dict(state["model"])
    res_real = test_model(V, m_real, test_set)

    # --- Train TSTR on SYNTHETIC ---
    print("Training readmission model on synthetic (TSTR)...")
    m_syn = VisitLSTM(V).to(device)
    ck_syn = os.path.join(utility_dir, f"readmit_syn_sigma{sigma_days}.pt")
    if not os.path.exists(ck_syn):
        train_model(V, m_syn, tr_syn, val_set, ck_syn)
    state = torch.load(ck_syn, map_location=device)
    m_syn.load_state_dict(state["model"])
    res_syn = test_model(V, m_syn, test_set)
    results = {"Real": res_real, "Syn": res_syn}
    plot_real_vs_syn(synthetic_data_dir, results, task_name="Readmission prediction")
    return {
        "Real": res_real,
        "Syn":  res_syn,
        "counts": {
            "Vocab": V,
            "train_real": len(tr_real),
            "train_syn": len(tr_syn),
            "val": len(val_set),
            "test": len(test_set),
            "sigma_days": sigma_days,
        },
    }
    