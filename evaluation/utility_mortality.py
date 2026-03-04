# evaluation/utility_mortality.py

import os
import json
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from sklearn import metrics
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from evaluation.utils import plot_real_vs_syn
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Model & training hyperparams
# ----------------------------
LR = 1e-3
EPOCHS = 50
BATCH_SIZE = 512
LSTM_HIDDEN_DIM = 32
EMBEDDING_DIM = 64
N_CTX = 48  # max # visits fed to the LSTM
NUM_TRAIN_EXAMPLES = 5000
NUM_VAL_EXAMPLES   =  500
NUM_TEST_EXAMPLES  = 1000


# ----------------------------
# Utilities: code vocabulary
# ----------------------------
def build_dx_proc_drug_vocab(tokenizer) -> Tuple[Dict[int, int], int]:
    """
    Build a compact {token_id -> compact_index} mapping over tokens whose text
    starts with: ICD10CM_, ICD10PCS_, ATC_. Returns (map, size).
    """
    keep_prefixes = ("ICD10CM_", "ICD10PCS_", "ATC_")
    tok2compact: Dict[int, int] = {}
    next_idx = 0
    # tokenizer.vocab_size ensures we iterate all ids safely
    for tid in range(tokenizer.vocab_size):
        try:
            t = tokenizer.decode([tid])
        except Exception:
            continue
        if any(t.startswith(p) for p in keep_prefixes):
            tok2compact[tid] = next_idx
            next_idx += 1
    return tok2compact, next_idx


# ----------------------------
# Sample construction (visit-level)
# ----------------------------
def _decode_cache(tokenizer):
    cache = {}
    def dec(tid: int) -> str:
        if tid not in cache:
            cache[tid] = tokenizer.decode([tid])
        return cache[tid]
    return dec

def construct_mortality_samples(
    structured_data: List[Dict],
    tokenizer,
    tok2compact: Dict[int, int],
) -> List[Dict]:
    """
    Build visit-level mortality samples:
      Input  X_t  = [v1, v2, ..., v_{t-1}]  (each v is a list of compact code indices)
      Label  y_t  = 1 if visit v_t ends in death, else 0
    Drop the last "no-next-visit" sample by construction (we never build a sample for t=1).
    We infer 'death during v_t' by checking whether the patient sequence contains a DEATH token,
    and if so, we assume it applies to the final visit (common with your serialization).
    """
    dec = _decode_cache(tokenizer)
    samples: List[Dict] = []

    for patient in structured_data:
        visits_raw: List[List[int]] = patient.get("visits", [])
        if not visits_raw:
            continue

        # Keep only Dx/Proc/Drug tokens and remap to compact ids
        visits_compact: List[List[int]] = []
        for v in visits_raw:
            bag = []
            for tid in v:
                if tid in tok2compact:
                    bag.append(tok2compact[tid])
            if bag:
                visits_compact.append(sorted(set(bag)))  # de-dup within visit
        if len(visits_compact) < 2:
            # Need at least 2 visits to create one (history -> next) sample
            continue

        # Does this patient die in their last recorded visit?
        # Your serialization places 'DEATH' between last END_VISIT and END_RECORD.
        has_death = False
        for v in visits_raw:  # quick scan decoded only if needed
            # Skip; death is outside visits per your format. We'll check whole concatenated text.
            pass
        # Cheaper check: concatenate tokens of the whole record once.
        # We only need to know if 'DEATH' appears anywhere in the record.
        # Demographics are not in visits_raw; but death sits after END_VISIT and before END_RECORD.
        record_has_death = False
        # Reconstruct the flattened token ids for this patient from visits + headers is not provided here;
        # instead decode a few tokens in each visit and check for literal 'DEATH' tokens among tids.
        # Safer: search for 'DEATH' by decoding any token equal to tokenizer.convert_tokens_to_ids('DEATH') if exists.
        try:
            tid_DEATH = tokenizer.convert_tokens_to_ids("DEATH")
            # Search in visits and (optionally) demographics if they contained token ids (they don't in your dict).
            for v in visits_raw:
                if tid_DEATH in v:
                    record_has_death = True
                    break
        except Exception:
            # Fallback: decode a small subset (rarely needed)
            record_has_death = False

        # Assign the mortality label to the final visit if death is present.
        last_visit_is_mortal = bool(record_has_death)

        # Build samples for t = 2..T (history -> visit t label)
        T = len(visits_compact)
        for t in range(1, T):  # 1..T-1 (0-based), predicting visit t (1-based index)
            history = visits_compact[:t]  # v1..v_{t}
            # label for visit (t+1 in 1-based) equals 1 if (t == T-1 and last_visit_is_mortal)
            y = 1 if (t == T - 1 and last_visit_is_mortal) else 0
            samples.append({
                "visits": history,
                "label": y
            })

    return samples


# ----------------------------
# Model (same style as phenotype)
# ----------------------------
class MortalityModel(nn.Module):
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

    def forward(self, visit_bow_seq, lengths):
        # visit_bow_seq: [B, T, V] one-hot/bow per visit
        x = self.embedding(visit_bow_seq)         # [B, T, D]
        x = self.dropout(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)  # [B, T, 2H]

        # last valid step per sequence (forward) & first step (reverse)
        idx = (lengths - 1).long()
        out_fwd = out[torch.arange(out.size(0)), idx, :LSTM_HIDDEN_DIM]
        out_rev = out[:, 0, LSTM_HIDDEN_DIM:]
        h = torch.cat([out_fwd, out_rev], dim=-1)
        logits = self.fc(h).squeeze(-1)
        prob = torch.sigmoid(logits)
        return prob


# ----------------------------
# Batching / training / eval
# ----------------------------
def visits_to_bow_tensor(samples, vocab_size, start, batch_size):
    batch = samples[start:start+batch_size]
    bows = np.zeros((len(batch), N_CTX, vocab_size), dtype=np.float32)
    lens = np.zeros(len(batch), dtype=np.int64)
    labels = np.array([b["label"] for b in batch], dtype=np.float32)
    for i, s in enumerate(batch):
        seq = s["visits"][:N_CTX]
        lens[i] = min(len(seq), N_CTX)
        for t, visit in enumerate(seq[:N_CTX]):
            for cidx in visit:
                bows[i, t, cidx] = 1.0
    return bows, labels, lens

def train_model(vocab_size, model, train_samples, val_samples, save_path):
    bce = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best = float("inf"); best_state = None

    for _ in tqdm(range(EPOCHS)):
        np.random.shuffle(train_samples)
        # train
        model.train()
        losses = []
        for i in range(0, len(train_samples), BATCH_SIZE):
            bows, labels, lens = visits_to_bow_tensor(train_samples, vocab_size, i, BATCH_SIZE)
            bows = torch.tensor(bows, device=device)
            labels = torch.tensor(labels, device=device)
            opt.zero_grad()
            prob = model(bows, torch.tensor(lens))
            loss = bce(prob, labels)
            loss.backward()
            opt.step()
            losses.append(loss.detach().cpu().item())

        # val
        model.eval()
        with torch.no_grad():
            vlosses = []
            for i in range(0, len(val_samples), BATCH_SIZE):
                bows, labels, lens = visits_to_bow_tensor(val_samples, vocab_size, i, BATCH_SIZE)
                bows = torch.tensor(bows, device=device)
                labels = torch.tensor(labels, device=device)
                prob = model(bows, torch.tensor(lens))
                vloss = bce(prob, labels)
                vlosses.append(vloss.detach().cpu().item())
        cur = float(np.mean(vlosses)) if vlosses else float("inf")
        if cur < best:
            best = cur
            best_state = {"model": model.state_dict(), "opt": opt.state_dict()}
            torch.save(best_state, save_path)

    if best_state is not None:
        model.load_state_dict(best_state["model"])


def test_model(vocab_size, model, test_samples, n_bootstrap=1000, seed=42):
    model.eval()
    bce = nn.BCELoss()
    losses, probs, trues = [], [], []
    with torch.no_grad():
        for i in range(0, len(test_samples), BATCH_SIZE):
            bows, labels, lens = visits_to_bow_tensor(test_samples, vocab_size, i, BATCH_SIZE)
            bows = torch.tensor(bows, device=device)
            labels = torch.tensor(labels, device=device)
            prob = model(bows, torch.tensor(lens))
            loss = bce(prob, labels)
            losses.append(loss.detach().cpu().item())
            probs.extend(prob.detach().cpu().numpy().tolist())
            trues.extend(labels.detach().cpu().numpy().tolist())

    preds = np.array(probs)
    y = np.array(trues).astype(int)
    yhat = (preds >= 0.5).astype(int)

    out = {
        "Test Loss": float(np.mean(losses)) if losses else float("nan"),
        "Confusion Matrix": metrics.confusion_matrix(y, yhat),
        "Accuracy": metrics.accuracy_score(y, yhat),
        "Precision": metrics.precision_score(y, yhat, zero_division=0),
        "Recall": metrics.recall_score(y, yhat, zero_division=0),
        "F1_score": metrics.f1_score(y, yhat, zero_division=0),
    }

    # ROC-AUC and bootstrap CI
    if y.min() != y.max():
        roc_auc = metrics.roc_auc_score(y, preds)
        bootstrapped_scores = []
        rng = np.random.RandomState(seed)
        for _ in range(n_bootstrap):
            indices = rng.randint(0, len(y), len(y))
            if len(np.unique(y[indices])) < 2:
                continue  # skip if not both classes
            score = metrics.roc_auc_score(y[indices], preds[indices])
            bootstrapped_scores.append(score)
        ci_lower = np.percentile(bootstrapped_scores, 2.5)
        ci_upper = np.percentile(bootstrapped_scores, 97.5)
        out["ROC_AUC"] = roc_auc
        out["ROC_AUC_CI"] = (ci_lower, ci_upper)
    else:
        out["ROC_AUC"] = float("nan")
        out["ROC_AUC_CI"] = (float("nan"), float("nan"))

    # AUPRC
    pr, rc, _ = metrics.precision_recall_curve(y, preds)
    out["AUPRC"] = metrics.auc(rc, pr)

    return out


# ----------------------------
# Public entry point
# ----------------------------
def utility_mortality_pred(
    tokenizer,
    structured_train_data,
    structured_val_data,
    structured_test_data,
    structured_syn_data,
    synthetic_data_dir,
):
    """
    TSTR: train on synthetic, validate on real-val, test on real-test.
    Also report a baseline trained on real.
    """
    utility_dir = os.path.join(synthetic_data_dir, "utility_models")
    os.makedirs(utility_dir, exist_ok=True)

    # Build compact vocab over Dx/Proc/Drug
    tok2compact, V = build_dx_proc_drug_vocab(tokenizer)

    # Construct samples (visit-level)
    syn_samples  = construct_mortality_samples(structured_syn_data,  tokenizer, tok2compact)
    tr_samples   = construct_mortality_samples(structured_train_data, tokenizer, tok2compact)
    val_samples  = construct_mortality_samples(structured_val_data,   tokenizer, tok2compact)
    test_samples = construct_mortality_samples(structured_test_data,  tokenizer, tok2compact)

    # Filter empty
    syn_samples  = [s for s in syn_samples  if len(s["visits"]) > 0]
    tr_samples   = [s for s in tr_samples   if len(s["visits"]) > 0]
    val_samples  = [s for s in val_samples  if len(s["visits"]) > 0]
    test_samples = [s for s in test_samples if len(s["visits"]) > 0]

    def stratified_subset(samples, n_total, tag=""):
        if n_total is None or len(samples) <= n_total:
            print(f"[{tag}] using all {len(samples)} samples (no subsampling)")
            return samples
        pos = [s for s in samples if s["label"] == 1]
        neg = [s for s in samples if s["label"] == 0]
        k = n_total // 2
        pos_pick = np.random.choice(pos, k, replace=(len(pos) < k)).tolist() if pos else []
        neg_pick = np.random.choice(neg, k, replace=(len(neg) < k)).tolist()
        out = pos_pick + neg_pick
        random.shuffle(out)
        # print(f"[{tag}] total={len(samples)} → subset={len(out)} "
        #     f"(pos avail={len(pos)}, neg avail={len(neg)}, "
        #     f"pos picked={len(pos_pick)}, neg picked={len(neg_pick)}, "
        #     f"replace_pos={len(pos)<k}, replace_neg={len(neg)<k})")
        return out

    tr_real  = stratified_subset(tr_samples,  NUM_TRAIN_EXAMPLES, tag="train_real")
    tr_syn   = stratified_subset(syn_samples, NUM_TRAIN_EXAMPLES, tag="train_syn")
    val_set  = stratified_subset(val_samples, NUM_VAL_EXAMPLES,   tag="val")
    test_set = stratified_subset(test_samples, NUM_TEST_EXAMPLES, tag="test")

    # --- Train on REAL ---
    print("Training mortality model on real...")
    m_real = MortalityModel(V).to(device)
    real_ckpt = os.path.join(utility_dir, "mortality_real.pt")
    if not os.path.exists(real_ckpt):
        train_model(V, m_real, tr_real, val_set, real_ckpt)
    state = torch.load(real_ckpt, map_location=device)
    m_real.load_state_dict(state["model"])
    res_real = test_model(V, m_real, test_set)

    # --- Train on SYNTHETIC (TSTR) ---
    print("Training mortality model on synthetic (TSTR)...")
    m_syn = MortalityModel(V).to(device)
    syn_ckpt = os.path.join(utility_dir, "mortality_syn.pt")
    if not os.path.exists(syn_ckpt):
        train_model(V, m_syn, tr_syn, val_set, syn_ckpt)
    state = torch.load(syn_ckpt, map_location=device)
    m_syn.load_state_dict(state["model"])
    res_syn = test_model(V, m_syn, test_set)

    results = {"Real": res_real, "Syn": res_syn, "counts": {
        "Vocab": V,
        "train_real": len(tr_real),
        "train_syn": len(tr_syn),
        "val": len(val_set),
        "test": len(test_set),
    }}
    plot_real_vs_syn(synthetic_data_dir, results, task_name="Mortality prediction")
    return results