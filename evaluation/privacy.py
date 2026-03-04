import random, numpy as np
from collections import Counter
from tqdm import tqdm
from sklearn import metrics

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)

# ------------------------------
# Helpers on the structured form
# ------------------------------

def _union_of_codes(patient):
    """Set of all codes across all visits (ints)."""
    s = set()
    for v in patient["visits"]:
        s.update(v)
    return s

def _visit_sets(patient):
    """List of per-visit sets (for distance)."""
    return [set(v) for v in patient["visits"]]

def _hamming_dist(a, b):
    """
    Symmetric difference size (Generalized Hamming):
    |A union B| - |A intersect B|
    Equivalently: |A| + |B| - 2 * |A intersect B|
    """
    return len(a) + len(b) - 2 * len(a & b)

def _record_distance(p, q):
    """
    Visit-aware Hamming distance between two records.
    Sum of symmetric differences for aligned visits.
    """
    Pv = _visit_sets(p)
    Qv = _visit_sets(q)
    
    # 1. Penalty for mismatch in number of visits (Hamming logic: size of extra sets)
    # Note: If one patient has more visits, those extra visits contribute their full size
    # to the symmetric difference. However, your previous script simplified this 
    # to just counting the *number* of mismatched visits (d += 1 or d += len(v)).
    # To strictly match your previous `find_hamming` logic which did:
    # "d += len(v)" for extra visits, we should sum the lengths of the extra visits.
    
    # STRICT ALIGNMENT WITH PREVIOUS WORK:
    L_min = min(len(Pv), len(Qv))
    d = 0
    
    # Distance for aligned visits
    for i in range(L_min):
        d += _hamming_dist(Pv[i], Qv[i])
        
    # Distance for extra visits (add full size of codes in extra visits)
    if len(Pv) > L_min:
        for i in range(L_min, len(Pv)):
            d += len(Pv[i])
    elif len(Qv) > L_min:
        for i in range(L_min, len(Qv)):
            d += len(Qv[i])
            
    return d

def _nearest_distance_to_pool(target, pool):
    """Min distance from one real record to any record in the pool."""
    mind = float("inf")
    for s in pool:
        d = _record_distance(target, s)
        if d < mind:
            mind = d
    return mind

# ------------------------------------
# 1) Membership Inference Attack (MIA)
# ------------------------------------
def membership_inference_attack(train_data, test_data, synthetic_data, n_eval=500):
    # Remove empties
    syn_nonempty = [p for p in synthetic_data if len(p["visits"]) > 0]
    tr_nonempty  = [p for p in train_data     if len(p["visits"]) > 0]
    te_nonempty  = [p for p in test_data      if len(p["visits"]) > 0]

    # Sample balanced evaluation set of members/non-members
    pos = [(p, 1) for p in random.sample(tr_nonempty, min(n_eval, len(tr_nonempty)))]
    neg = [(p, 0) for p in random.sample(te_nonempty, min(n_eval, len(te_nonempty)))]
    n = min(len(pos), len(neg))
    pos, neg = pos[:n], neg[:n]
    eval_set = pos + neg
    random.shuffle(eval_set)

    # Distances to nearest synthetic
    dists, labels = [], []
    for p, y in tqdm(eval_set, desc="MIA: distances"):
        d = _nearest_distance_to_pool(p, syn_nonempty)
        dists.append(d)
        labels.append(y)

    # Threshold at median distance (same as your previous code)
    med = float(np.median(dists))
    preds = [1 if d < med else 0 for d in dists]

    return {
        "Accuracy":  metrics.accuracy_score(labels, preds),
        "Precision": metrics.precision_score(labels, preds, zero_division=0),
        "Recall":    metrics.recall_score(labels, preds, zero_division=0),
        "F1":        metrics.f1_score(labels, preds, zero_division=0),
        "MedianThreshold": med,
        "PosN": int(n), "NegN": int(n)
    }

# ------------------------------------
# 2) Attribute Inference Attack (AIA)
# ------------------------------------
def _most_common_codes(patients, top_k=100):
    cnt = Counter()
    for p in patients:
        for v in p["visits"]:
            cnt.update(v)
    return set([c for c, _ in cnt.most_common(top_k)])

def _known_secret_split(patient, common_codes):
    u = _union_of_codes(patient)
    known  = u & common_codes   # attacker background knowledge
    secret = u - common_codes   # target attributes to infer
    return known, secret

def _knn_predict_secret(target_known, ref_known_secret, k=5):
    """
    ref_known_secret: list of tuples (known_set, secret_set)
    distance: Hamming distance (Symmetric Difference) on KNOWN sets
    prediction: majority vote of SECRET codes among k nearest
    """
    dists = []
    for known, secret in ref_known_secret:
        # MODIFICATION HERE: Use Hamming instead of Jaccard
        d = _hamming_dist(target_known, known) 
        dists.append((d, secret))
    
    dists.sort(key=lambda x: x[0])
    neighbors = [sec for _, sec in dists[:k]]
    vote = Counter()
    for s in neighbors:
        vote.update(s)
    # strict majority vote
    return set([c for c, cnt in vote.items() if cnt > k/2])

def attribute_inference_attack(train_data, test_data, synthetic_data,
                               top_common=100, k=5):
    """
    Attack set: randomly choose |test_data| targets from train_data (non-sensitive cohort).
    Compare attacker using SYNTHETIC vs BASELINE using held-out TEST as references.
    """
    # choose same number across sets
    n = min(len(train_data), len(test_data), len(synthetic_data))
    tr = random.sample(train_data, n)
    te = random.sample(test_data, n)
    sy = [p for p in synthetic_data if len(p["visits"]) > 0]
    sy = random.sample(sy, min(n, len(sy)))

    # Define attacker background: top-K frequent codes (can also add demographics if desired)
    common = _most_common_codes(tr + te, top_k=top_common)

    # Build (known, secret) pairs
    def build_known_secret(ps):
        out = []
        for p in ps:
            known, secret = _known_secret_split(p, common)
            out.append((known, secret))
        return out

    tr_ks = build_known_secret(tr)
    te_ks = build_known_secret(te)
    sy_ks = build_known_secret(sy)

    # Evaluate F1 for attacker using synthetic as reference vs baseline using test as reference
    def eval_against(ref_ks):
        tp = fp = fn = 0
        for known, true_secret in tqdm(tr_ks, desc="AIA"):
            pred = _knn_predict_secret(known, ref_ks, k=k)
            tpos = len(pred & true_secret)
            fpos = len(pred - true_secret)
            fneg = len(true_secret - pred)
            tp += tpos; fp += fpos; fn += fneg
        # Sørensen–Dice/F1 on set union
        return tp / (tp + 0.5 * (fp + fn) + 1e-12)

    f1_attack   = eval_against(sy_ks)  # attacker with synthetic reference
    f1_baseline = eval_against(te_ks)  # same attack, but reference is held-out test

    return {
        "Attribute Attack F1 (synthetic ref)": float(f1_attack),
        "Baseline F1 (test ref)": float(f1_baseline),
        "TopCommon": int(top_common),
        "kNN": int(k)
    }

# ------------------------------
# Entry point (same signature)
# ------------------------------
def privacy_evaluation(train_data, test_data, synthetic_data,
                       n_eval_mia=500, top_common=100, k=1):
    return {
        "MIA": membership_inference_attack(
            train_data, test_data, synthetic_data, n_eval=n_eval_mia
        ),
        "AIA": attribute_inference_attack(
            train_data, test_data, synthetic_data, top_common=top_common, k=k
        ),
    }