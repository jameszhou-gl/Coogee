import os
import torch
import pickle
import json
import random
import itertools
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from sklearn import metrics
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from evaluation.utils import plot_real_vs_syn

SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.001
EPOCHS = 50
LABEL_IDX_LIST = list(range(25))
BATCH_SIZE = 512
LSTM_HIDDEN_DIM = 32
EMBEDDING_DIM = 64
NUM_TRAIN_EXAMPLES = 5000
NUM_TEST_EXAMPLES = 1000
NUM_VAL_EXAMPLES = 500
# CODE_VOCAB_SIZE = 6984
N_CTX = 48


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class DiagnosisModel(nn.Module):
    def __init__(self, code_vocab_size):
        super(DiagnosisModel, self).__init__()
        self.embedding = nn.Linear(code_vocab_size, EMBEDDING_DIM, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(input_size=EMBEDDING_DIM,
                            hidden_size=LSTM_HIDDEN_DIM,
                            num_layers=2,
                            dropout=0.5,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(2*LSTM_HIDDEN_DIM, 1)

    def forward(self, input_visits, lengths):
        visit_emb = self.embedding(input_visits)
        visit_emb = self.dropout(visit_emb)
        packed_input = pack_padded_sequence(
            visit_emb, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), lengths - 1, :LSTM_HIDDEN_DIM]
        out_reverse = output[:, 0, LSTM_HIDDEN_DIM:]
        out_combined = torch.cat((out_forward, out_reverse), 1)

        patient_embedding = self.fc(out_combined)
        patient_embedding = torch.squeeze(patient_embedding, 1)
        prob = torch.sigmoid(patient_embedding)

        return prob


def get_batch(code_vocab_size, ehr_dataset, loc, batch_size, label_idx):
    ehr = ehr_dataset[loc:loc+batch_size]
    batch_ehr = np.zeros((len(ehr), N_CTX, code_vocab_size))
    batch_labels = np.array([p['labels'][label_idx] for p in ehr])
    batch_lens = np.zeros(len(ehr))
    for i, p in enumerate(ehr):
        visits = p['visits'][:N_CTX]
        batch_lens[i] = min(len(visits), N_CTX)
        for j, v in enumerate(visits):
            batch_ehr[i, j][v] = 1

    return batch_ehr, batch_labels, batch_lens


def train_model(code_vocab_size, model, train_dataset, val_dataset, save_name, label_idx):
    global_loss = 1e10
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCELoss()
    for e in tqdm(range(EPOCHS)):
        np.random.shuffle(train_dataset)
        train_losses = []
        for i in range(0, len(train_dataset), BATCH_SIZE):
            model.train()
            batch_ehr, batch_labels, batch_lens = get_batch(code_vocab_size,
                train_dataset, i, BATCH_SIZE, label_idx)
            batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
            batch_labels = torch.tensor(
                batch_labels, dtype=torch.float32).to(device)
            optimizer.zero_grad()
            prob = model(batch_ehr, batch_lens)
            loss = bce(prob, batch_labels)
            train_losses.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
        cur_train_loss = np.mean(train_losses)
        # print("Epoch %d Training Loss:%.5f" % (e, cur_train_loss))

        model.eval()
        with torch.no_grad():
            val_losses = []
            for v_i in range(0, len(val_dataset), BATCH_SIZE):
                batch_ehr, batch_labels, batch_lens = get_batch(code_vocab_size,
                    val_dataset, v_i, BATCH_SIZE, label_idx)
                batch_ehr = torch.tensor(
                    batch_ehr, dtype=torch.float32).to(device)
                batch_labels = torch.tensor(
                    batch_labels, dtype=torch.float32).to(device)
                prob = model(batch_ehr, batch_lens)
                val_loss = bce(prob, batch_labels)
                val_losses.append(val_loss.cpu().detach().numpy())
            cur_val_loss = np.mean(val_losses)
            # print("Epoch %d Validation Loss:%.5f" % (e, cur_val_loss))
            if cur_val_loss < global_loss:
                global_loss = cur_val_loss
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(state, f'{save_name}')
                # print('------------ Save best model ------------')

    model.load_state_dict(state['model'])


def test_model(code_vocab_size, model, test_dataset, label_idx):
    loss_list = []
    pred_list = []
    true_list = []
    bce = nn.BCELoss()
    model.eval()
    with torch.no_grad():
        for i in range(0, len(test_dataset), BATCH_SIZE):
            batch_ehr, batch_labels, batch_lens = get_batch(code_vocab_size,
                test_dataset, i, BATCH_SIZE, label_idx)
            batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
            batch_labels = torch.tensor(
                batch_labels, dtype=torch.float32).to(device)
            prob = model(batch_ehr, batch_lens)
            val_loss = bce(prob, batch_labels)
            loss_list.append(val_loss.cpu().detach().numpy())
            pred_list += list(prob.cpu().detach().numpy())
            true_list += list(batch_labels.cpu().detach().numpy())

    round_list = np.around(pred_list)

    # Extract, save, and display test metrics
    avg_loss = np.mean(loss_list)
    cmatrix = metrics.confusion_matrix(true_list, round_list)
    acc = metrics.accuracy_score(true_list, round_list)
    prc = metrics.precision_score(true_list, round_list)
    rec = metrics.recall_score(true_list, round_list)
    f1 = metrics.f1_score(true_list, round_list)
    auroc = metrics.roc_auc_score(true_list, pred_list)
    (precisions, recalls, _) = metrics.precision_recall_curve(true_list, pred_list)
    auprc = metrics.auc(recalls, precisions)

    metrics_dict = {}
    metrics_dict['Test Loss'] = avg_loss
    metrics_dict['Confusion Matrix'] = cmatrix
    metrics_dict['Accuracy'] = acc
    metrics_dict['Precision'] = prc
    metrics_dict['Recall'] = rec
    metrics_dict['F1 Score'] = f1
    metrics_dict['AUROC'] = auroc
    metrics_dict['AUPRC'] = auprc

    return metrics_dict

# ---- helpers to build a stable 0..K-1 phenotype index mapping ----
def build_pheno_index(icd10_to_pheno_path="evaluation/icd10_to_pheno/icd10_to_pheno_mapping.json"):
    """
    Reads evaluation/icd10_to_pheno/icd10_to_pheno_mapping.json and returns:
      - pheno2id: dict {phenotype_name -> idx}
      - id2pheno: dict {idx -> phenotype_name}
    Indices are assigned in sorted(name) order for reproducibility.
    """
    with open(icd10_to_pheno_path, "r") as f:
        icd10_to_pheno = json.load(f)

    all_phenos = set()
    for phenos in icd10_to_pheno.values():
        all_phenos.update(phenos)

    pheno_list = sorted(all_phenos)
    pheno2id = {p: i for i, p in enumerate(pheno_list)}
    id2pheno = {i: p for p, i in pheno2id.items()}
    return pheno2id, id2pheno


def construct_label(
    structured_data: List[Dict],
    icd10_to_pheno_mapping: str = "evaluation/icd10_to_pheno/icd10_to_pheno_mapping.json",
    tokenizer=None,
    pheno2id: Optional[Dict[str, int]] = None,
    drop_no_label: bool = False,
) -> List[Dict]:
    """
    Convert each structured patient into a sample with a multi-hot phenotype label vector.
    - structured_data: list of {"demographics": [...], "visits": [[int,...], ...]}
    - icd10_to_pheno_mapping: JSON mapping {"A021": ["Septicemia (except in labor)", ...], ...}
    - tokenizer: LocalTokenizer to decode token IDs -> text (must decode ICD10CM_* tokens)
    - pheno2id: optional external mapping {phenotype_name -> idx}. If None, build from the JSON.
    - drop_no_label: if True, skip patients that map to no phenotype (all-zero vector)

    Returns a new list of dicts; each dict includes:
        - 'demographics'
        - 'visits'
        - 'labels' : np.ndarray of shape (K,), dtype=int, with 0/1
        - 'label_indices' : sorted list of indices that are 1
    """
    # --- load JSON mapping ICD-10 -> phenotype names ---
    with open(icd10_to_pheno_mapping, "r") as f:
        icd10_to_pheno = json.load(f)

    K = len(pheno2id)

    if tokenizer is None:
        raise ValueError("construct_label requires a tokenizer to decode token IDs (ICD10CM_*).")

    out = []
    # cache decode for speed
    _decode_cache: Dict[int, str] = {}

    def dec(tid: int) -> str:
        if tid not in _decode_cache:
            _decode_cache[tid] = tokenizer.decode([tid])
        return _decode_cache[tid]

    for patient in structured_data:
        labels = np.zeros(K, dtype=int)

        # gather all ICD10CM codes across all visits
        for v in patient.get("visits", []):
            for tid in v:
                tok = dec(int(tid))
                # keep only diagnosis tokens like "ICD10CM_Z8571"
                if tok.startswith("ICD10CM_"):
                    icd10 = tok.split("_", 1)[1]  # "Z8571"
                    phenos = icd10_to_pheno.get(icd10)
                    if not phenos:
                        continue
                    for pname in phenos:
                        idx = pheno2id.get(pname)
                        if idx is not None:
                            labels[idx] = 1

        if drop_no_label and labels.sum() == 0:
            continue
        # Instead of keeping all visits tokens:
        visits_diag_only = []
        for v in patient.get("visits", []):
            diag_tokens = []
            for tid in v:
                tok = dec(int(tid))
                if tok.startswith("ICD10CM_"):  # keep only diagnosis tokens
                    diag_tokens.append(tid)
            if diag_tokens:
                visits_diag_only.append(diag_tokens)

        # replace patient visits
        sample = {
            "demographics": patient.get("demographics", []),
            "visits": visits_diag_only,  # now restricted to diagnoses
            "labels": labels
        }
        out.append(sample)

    return out

# --- helper: 95% CI via bootstrap over labels ---
def mean_and_ci(values: List[float], n_boot: int = 2000, seed: int = 1337):
    """Return (mean, (lo95, hi95)) for a 1D list/array using bootstrap over entries."""
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    if len(arr) == 0:
        return mean, (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(arr), size=len(arr))
        boots.append(np.mean(arr[idx]))
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    return mean, (lo, hi)

def utility_phenotype_pred(tokenizer, structured_train_data, structured_val_data,
                           structured_test_data, structured_syn_data, synthetic_data_dir):
    code_vocab_size = 20024
    utility_model_dir = os.path.join(synthetic_data_dir, "utility_models")
    os.makedirs(utility_model_dir, exist_ok=True)

    pheno2id, id2pheno = build_pheno_index()
    synthetic_dataset = construct_label(structured_syn_data, tokenizer=tokenizer, pheno2id=pheno2id)
    train_ehr_dataset = construct_label(structured_train_data, tokenizer=tokenizer, pheno2id=pheno2id)
    val_ehr_dataset   = construct_label(structured_val_data, tokenizer=tokenizer, pheno2id=pheno2id)
    test_ehr_dataset  = construct_label(structured_test_data, tokenizer=tokenizer, pheno2id=pheno2id)

    results = {}

    # collect per-label metrics to bootstrap CIs later
    f1_syn, acc_syn, prec_syn, rec_syn, auroc_syn, auprc_syn = [], [], [], [], [], []
    f1_real, acc_real, prec_real, rec_real, auroc_real, auprc_real = [], [], [], [], [], []

    for i in LABEL_IDX_LIST:
        print(f"processing label {str(i)}: {id2pheno[i]}")
        label_results = {}

        # Prepare datasets (same as your original)
        syn_pos = [p for p in synthetic_dataset if p['labels'][i] == 1]
        syn_neg = [p for p in synthetic_dataset if p['labels'][i] == 0]
        tr_pos  = [p for p in train_ehr_dataset  if p['labels'][i] == 1]
        tr_neg  = [p for p in train_ehr_dataset  if p['labels'][i] == 0]
        val_pos = [p for p in val_ehr_dataset    if p['labels'][i] == 1]
        val_neg = [p for p in val_ehr_dataset    if p['labels'][i] == 0]
        te_pos  = [p for p in test_ehr_dataset   if p['labels'][i] == 1]
        te_neg  = [p for p in test_ehr_dataset   if p['labels'][i] == 0]

        val_dataset  = list(np.random.choice(val_pos,  NUM_VAL_EXAMPLES//2,  replace=(len(val_pos)  < NUM_VAL_EXAMPLES//2))) + \
                       list(np.random.choice(val_neg,  NUM_VAL_EXAMPLES//2,  replace=False))
        test_dataset = list(np.random.choice(te_pos,   NUM_TEST_EXAMPLES//2, replace=(len(te_pos)   < NUM_TEST_EXAMPLES//2))) + \
                       list(np.random.choice(te_neg,   NUM_TEST_EXAMPLES//2, replace=False))
        tr_real_set  = list(np.random.choice(tr_pos,   NUM_TRAIN_EXAMPLES//2, replace=(len(tr_pos)   < NUM_TRAIN_EXAMPLES//2))) + \
                       list(np.random.choice(tr_neg,   NUM_TRAIN_EXAMPLES//2, replace=False))
        tr_syn_set   = list(np.random.choice(syn_pos,  NUM_TRAIN_EXAMPLES//2, replace=(len(syn_pos)  < NUM_TRAIN_EXAMPLES//2))) + \
                       list(np.random.choice(syn_neg,  NUM_TRAIN_EXAMPLES//2, replace=False))

        tr_real_set = [p for p in tr_real_set if len(p['visits']) > 0]
        tr_syn_set  = [p for p in tr_syn_set  if len(p['visits']) > 0]
        val_dataset = [p for p in val_dataset if len(p['visits']) > 0]
        test_dataset= [p for p in test_dataset if len(p['visits']) > 0]

        # Train/eval REAL
        print("Training on real data...")
        model_real = DiagnosisModel(code_vocab_size).to(device)
        real_ckpt = f"{utility_model_dir}/utility_real_{i}.pt"
        if not os.path.exists(real_ckpt):
            train_model(code_vocab_size, model_real, tr_real_set, val_dataset, real_ckpt, i)
        state = torch.load(real_ckpt, map_location=device)
        model_real.load_state_dict(state['model'])
        res_real = test_model(code_vocab_size, model_real, test_dataset, i)  # keys: 'F1 Score','Accuracy','Precision','Recall','AUROC','AUPRC'
        label_results['Real'] = res_real

        # Train/eval SYN
        print("Training on synthetic data...")
        model_syn = DiagnosisModel(code_vocab_size).to(device)
        syn_ckpt = f"{utility_model_dir}/utility_syn_{i}.pt"
        if not os.path.exists(syn_ckpt):
            train_model(code_vocab_size, model_syn, tr_syn_set, val_dataset, syn_ckpt, i)
        state = torch.load(syn_ckpt, map_location=device)
        model_syn.load_state_dict(state['model'])
        res_syn = test_model(code_vocab_size, model_syn, test_dataset, i)
        label_results['Syn'] = res_syn

        results[id2pheno[i]] = label_results

        # accumulate per-label metrics (Syn)
        f1_syn.append(res_syn['F1 Score'])
        acc_syn.append(res_syn['Accuracy'])
        prec_syn.append(res_syn['Precision'])
        rec_syn.append(res_syn['Recall'])
        auroc_syn.append(res_syn['AUROC'])
        auprc_syn.append(res_syn['AUPRC'])
        # accumulate per-label metrics (Real)
        f1_real.append(res_real['F1 Score'])
        acc_real.append(res_real['Accuracy'])
        prec_real.append(res_real['Precision'])
        rec_real.append(res_real['Recall'])
        auroc_real.append(res_real['AUROC'])
        auprc_real.append(res_real['AUPRC'])

    # --- summarize with mean and 95% CI (bootstrap across labels) ---
    final_results = {}
    final_results['Syn'] = {}
    final_results['Real'] = {}

    for split, F1s, Accs, Precs, Recs, AUCs, AUPRCs in [
        ('Syn',  f1_syn,  acc_syn,  prec_syn,  rec_syn,  auroc_syn,  auprc_syn),
        ('Real', f1_real, acc_real, prec_real, rec_real, auroc_real, auprc_real)
    ]:
        mean_f1,   ci_f1   = mean_and_ci(F1s)
        mean_acc,  ci_acc  = mean_and_ci(Accs)
        mean_pr,   ci_pr   = mean_and_ci(Precs)
        mean_rec,  ci_rec  = mean_and_ci(Recs)
        mean_auc,  ci_auc  = mean_and_ci(AUCs)
        mean_aupr, ci_aupr = mean_and_ci(AUPRCs)

        final_results[split]['F1_score'] = mean_f1
        final_results[split]['F1_score_CI'] = ci_f1
        final_results[split]['Accuracy'] = mean_acc
        final_results[split]['Accuracy_CI'] = ci_acc
        final_results[split]['Precision'] = mean_pr
        final_results[split]['Precision_CI'] = ci_pr
        final_results[split]['Recall'] = mean_rec
        final_results[split]['Recall_CI'] = ci_rec
        final_results[split]['ROC_AUC'] = mean_auc
        final_results[split]['ROC_AUC_CI'] = ci_auc
        final_results[split]['AUPRC'] = mean_aupr
        final_results[split]['AUPRC_CI'] = ci_aupr

    # optional: keep per-label entries as you already do
    final_results['per_label'] = results

    # plot (still uses means)
    plot_real_vs_syn(synthetic_data_dir, final_results, task_name="Phenotype prediction")

    return final_results