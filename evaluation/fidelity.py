import os
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy import sparse
from tqdm import tqdm
import seaborn as sns
import scipy.cluster.hierarchy as sch
import matplotlib.gridspec as gridspec

SEED = 1337
np.random.seed(SEED)

# Function to calculate mean lengths
def calculate_mean_lengths(data):
    record_lengths = [len(patient['visits']) for patient in data]
    visit_lengths = [len(visit)
                     for patient in data for visit in patient['visits']]
    return {
        'mean_record_length': np.mean(record_lengths),
        'std_record_length': np.std(record_lengths),
        'mean_visit_length': np.mean(visit_lengths),
        'std_visit_length': np.std(visit_lengths),
    }

def _as_int_visits(data):
    """Yield visits as np.int64 arrays (unique within visit), plus vocab size."""
    Vmax = 0
    seqs = []
    for p in data:
        visits_int = []
        for v in p['visits']:
            vi = np.asarray(v, dtype=np.int64)
            Vmax = max(Vmax, int(vi.max()) if vi.size else 0)
            visits_int.append(vi)
        seqs.append(visits_int)
    return seqs, Vmax + 1  # vocab size

def calculate_code_probabilities(
    data
):
    """
    Returns (unigram_probs, same_visit_bigram_probs, sequential_bigram_probs)
    """
    patients, V = _as_int_visits(data)

    # ----- Unigrams -----
    uni_counts = np.zeros(V, dtype=np.int64)
    for visits in patients:
        for v in visits:
            uni_counts += np.bincount(v, minlength=V)

    # ----- Same-visit bigrams (unordered, upper triangle only) -----
    coo_rows, coo_cols, coo_vals = [], [], []
    # ----- Sequential bigrams (ordered) -----
    seq_rows, seq_cols, seq_vals = [], [], []

    for visits in tqdm(patients, desc="Calculate code probabilities"):
        # per-visit one-hot as sparse row vectors:
        # Build once and reuse.
        row_vecs = []
        for v in visits:
            if v.size == 0:
                row_vecs.append(None)
                continue
            # one-hot row
            data_ = np.ones_like(v, dtype=np.float32)
            x = sparse.csr_matrix((data_, (np.zeros_like(v), v)), shape=(1, V))
            row_vecs.append(x)

        # same-visit: x^T @ x (unordered; take upper triangle)
        for x in row_vecs:
            if x is None:
                continue
            M = (x.T @ x).tocoo()  # very sparse
            # remove diagonal (i==j) and keep only upper triangle (i<j)
            keep = (M.row < M.col)
            coo_rows.append(M.row[keep])
            coo_cols.append(M.col[keep])
            coo_vals.append(M.data[keep])

        # sequential: x_t^T @ x_{t+1} (ordered)
        for t in range(len(row_vecs) - 1):
            x = row_vecs[t]
            y = row_vecs[t+1]
            if x is None or y is None:
                continue
            M = (x.T @ y).tocoo()
            seq_rows.append(M.row)
            seq_cols.append(M.col)
            seq_vals.append(M.data)

    # stack into big COO matrices, then sum duplicates
    def _stack_sum(rows_list, cols_list, vals_list, shape):
        if not rows_list:
            return sparse.coo_matrix(shape, dtype=np.float64)
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        vals = np.concatenate(vals_list).astype(np.float64)
        return sparse.coo_matrix((vals, (rows, cols)), shape=shape).tocsr()

    same_mat = _stack_sum(coo_rows, coo_cols, coo_vals, shape=(V, V))  # upper triangle filled
    seq_mat  = _stack_sum(seq_rows,  seq_cols,  seq_vals,  shape=(V, V))

    # ----- Convert to probabilities -----
    uni_total = uni_counts.sum()
    unigram_probs = {int(i): float(c) / float(uni_total)
                     for i, c in enumerate(uni_counts) if c > 0}

    same_total = same_mat.sum()
    # same-visit: keys as (i, j) with i<j
    same_visit_bigram_probs = {}
    if same_total > 0:
        S = same_mat.tocoo()
        inv_total = 1.0 / float(same_total)
        for i, j, v in zip(S.row, S.col, S.data):
            same_visit_bigram_probs[(int(i), int(j))] = float(v) * inv_total

    seq_total = seq_mat.sum()
    sequential_bigram_probs = {}
    if seq_total > 0:
        T = seq_mat.tocoo()
        inv_total = 1.0 / float(seq_total)
        for i, j, v in zip(T.row, T.col, T.data):
            sequential_bigram_probs[(int(i), int(j))] = float(v) * inv_total

    return unigram_probs, same_visit_bigram_probs, sequential_bigram_probs


# Function to compare probabilities with vectorized operations
def compare_probabilities(real_probs, synthetic_probs, label, save_dir):
    # Convert dictionaries to arrays efficiently
    # Ensure all keys are tuples of strings or strings
    all_keys = list(set(real_probs.keys()).union(synthetic_probs.keys()))
    
    # Convert values to arrays while maintaining key types
    real_values = np.array([real_probs.get(k, 0) for k in all_keys])
    synthetic_values = np.array([synthetic_probs.get(k, 0) for k in all_keys])
    
    # Calculate R2 score
    r2 = r2_score(real_values, synthetic_values)
    
    # Calculate deviation vectorized
    deviation = np.abs(real_values - synthetic_values)
    
    # Normalize deviation vectorized
    if deviation.max() != deviation.min():
        deviation_normalized = (deviation - deviation.min()) / (deviation.max() - deviation.min())
        deviation_normalized = 1 - deviation_normalized
    else:
        deviation_normalized = np.ones_like(deviation)
    
    # Create plot
    plt.figure(figsize=(10, 8))  # Larger figure for better visibility
    
    # Use hexbin for large datasets to improve performance and visibility
    # if len(all_keys) > 40000:
    #     plt.hexbin(real_values, synthetic_values, gridsize=50, 
    #               cmap='viridis', mincnt=1, bins='log')
    # else:
    scatter = plt.scatter(real_values, synthetic_values, 
                        c=deviation_normalized, cmap="RdBu_r", 
                        alpha=0.7, s=20, label="Data points")
    plt.colorbar(scatter)
    
    # Add reference line
    max_val = max(real_values.max(), synthetic_values.max()) * 1.1
    plt.plot([0, max_val], [0, max_val], color="red", 
             linestyle="--", linewidth=1, label="Reference line (y=x)")
    
    # Beautify plot
    # plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.xlabel("Real Data", fontsize=24)
    plt.ylabel("Synthetic Data", fontsize=24)
    plt.title(f"{label}", fontsize=22)
    
    # Set tick label font sizes
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    
    plt.xlim([0, max_val])
    plt.ylim([0, max_val])
    plt.legend(fontsize=22)
    
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    # Save plot
    plot_path = os.path.join(save_dir, f"{label.replace(' ', '_')}_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return r2

def calculate_bland_altman(real_probs, syn_probs, title="Bland-Altman Plot", save_path=None):
    all_keys = list(set(real_probs.keys()).union(syn_probs.keys()))
    
    real_values = np.array([real_probs.get(k, 0) for k in all_keys])
    syn_values = np.array([syn_probs.get(k, 0) for k in all_keys])
    
    # 1. Core BA Calculations
    differences = syn_values - real_values
    averages = (syn_values + real_values) / 2
    n = len(differences)
    
    mean_bias = np.mean(differences)
    sd_differences = np.std(differences, ddof=1) 
    
    lower_loa = mean_bias - (1.96 * sd_differences)
    upper_loa = mean_bias + (1.96 * sd_differences)
    
    # 2. Precision of the Estimates (Confidence Intervals)
    # Based on Bland & Altman 1986 calculations
    se_bias = sd_differences / np.sqrt(n)
    ci_bias_upper = mean_bias + (1.96 * se_bias)
    ci_bias_lower = mean_bias - (1.96 * se_bias)
    
    se_loa = np.sqrt(3 * sd_differences**2 / n)
    ci_upper_loa_upper = upper_loa + (1.96 * se_loa)
    ci_upper_loa_lower = upper_loa - (1.96 * se_loa)
    ci_lower_loa_upper = lower_loa + (1.96 * se_loa)
    ci_lower_loa_lower = lower_loa - (1.96 * se_loa)
    
    # 3. Proportion within LoA
    within_loa_count = np.sum((differences >= lower_loa) & (differences <= upper_loa))
    percent_within_loa = (within_loa_count / n) * 100
    
    # 4. Mean Absolute Error (MAE)
    mae = np.mean(np.abs(differences))
    
    # --- PLOTTING (style matched to correlation figures) ---
    if save_path:
        plt.figure(figsize=(10, 8))
        
        # Pre-transform x to log10 for spread across orders of magnitude
        log_avg = np.log10(np.maximum(averages, 1e-20))
        
        # Symlog parameters (computed early, needed for density estimation)
        loa_max = max(abs(upper_loa), abs(lower_loa))
        linthresh = loa_max * 3
        
        # --- Density-based coloring ---
        # Bin in display-coordinate space (log x, arcsinh y) so bins are fine
        # near y=0 where the LoA structure matters, coarser for outliers.
        y_display = np.arcsinh(differences / linthresh)
        n_bins = 200
        hist, xedges, yedges = np.histogram2d(log_avg, y_display, bins=n_bins)
        xi = np.clip(np.digitize(log_avg, xedges) - 1, 0, n_bins - 1)
        yi = np.clip(np.digitize(y_display, yedges) - 1, 0, n_bins - 1)
        point_density = hist[xi, yi]
        
        # Log-normalize (density spans orders of magnitude)
        log_dens = np.log10(point_density + 1)
        density_norm = log_dens / log_dens.max()
        
        # Sort so dense points (red) are drawn last, on top
        order = np.argsort(density_norm)
        
        # Adaptive marker size for large datasets
        n_points = len(differences)
        if n_points > 500_000:
            s, alpha = 1, 0.5
        elif n_points > 50_000:
            s, alpha = 5, 0.6
        else:
            s, alpha = 20, 0.7
        
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        from matplotlib.cm import ScalarMappable
        ba_cmap = LinearSegmentedColormap.from_list(
            'ba_density', ['#4291C2', '#FFFFFF', '#D7634F'])
        
        scatter = plt.scatter(log_avg[order], differences[order],
                              c=density_norm[order], cmap=ba_cmap,
                              vmin=0, vmax=1,
                              alpha=alpha, s=s, rasterized=True)
        sm = ScalarMappable(cmap=ba_cmap, norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label('Point Density (log scale)', fontsize=14)
        
        # Relabel x-axis to show original probability values
        lo = np.floor(log_avg.min())
        hi = np.ceil(log_avg.max())
        x_ticks = np.arange(lo, hi + 1)
        plt.xticks(x_ticks, [f'$10^{{{int(t)}}}$' for t in x_ticks])
        
        # BA reference lines
        plt.axhline(mean_bias, color='black', linestyle='-', linewidth=1.5)
        plt.axhline(upper_loa, color='red', linestyle='--', linewidth=1.5)
        plt.axhline(lower_loa, color='red', linestyle='--', linewidth=1.5)
        
        ax = plt.gca()
        ax.set_yscale('symlog', linthresh=linthresh)
        
        plt.xlabel("Average Probability", fontsize=24)
        plt.ylabel("Difference (Synthetic - Real)", fontsize=24)
        plt.title(f"{title}", fontsize=22)
        
        plt.tick_params(axis='x', labelsize=20)
        plt.tick_params(axis='y', labelsize=20)
        
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#D7634F', markersize=8,
                   label='Data points'),
            Line2D([0], [0], color='black', linestyle='-', linewidth=1.5,
                   label=f'Mean bias: {mean_bias:.2e}'),
            Line2D([0], [0], color='red', linestyle='--', linewidth=1.5,
                   label=f'+1.96 SD: {upper_loa:.2e}'),
            Line2D([0], [0], color='red', linestyle='--', linewidth=1.5,
                   label=f'$-$1.96 SD: {lower_loa:.2e}'),
        ]
        plt.legend(handles=legend_handles, fontsize=14, loc='upper left')
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    metrics_dict = {
        "mean_bias": mean_bias,
        "ci_bias": (ci_bias_lower, ci_bias_upper),
        "lower_loa": lower_loa,
        "ci_lower_loa": (ci_lower_loa_lower, ci_lower_loa_upper),
        "upper_loa": upper_loa,
        "ci_upper_loa": (ci_upper_loa_lower, ci_upper_loa_upper),
        "percent_within_loa": percent_within_loa,
        "mae": mae
    }
    
    return metrics_dict

# Run this for your unigrams and bigrams

def fidelity_from_halo_evaluation(real_data, synthetic_data, save_dir):
    # Generate statistics for real and synthetic data
    real_lengths = calculate_mean_lengths(real_data)
    synthetic_lengths = calculate_mean_lengths(synthetic_data)

    real_unigrams, real_same_bigrams, real_seq_bigrams = calculate_code_probabilities(
        real_data
    )
    synthetic_unigrams, synthetic_same_bigrams, synthetic_seq_bigrams = calculate_code_probabilities(
        synthetic_data
    )
    # Compare probabilities and save plots
    r2_unigrams = compare_probabilities(
        real_unigrams, synthetic_unigrams, "Correlation between Unigram Probabilities", save_dir)
    r2_same_bigrams = compare_probabilities(
        real_same_bigrams, synthetic_same_bigrams, "Correlation between Same-visit Bigram Probabilities", save_dir)
    r2_seq_bigrams = compare_probabilities(
        real_seq_bigrams, synthetic_seq_bigrams, "Correlation between Sequential-visit Bigram Probabilities", save_dir)

    unigrams_metrics = calculate_bland_altman(real_unigrams, synthetic_unigrams, title="Bland-Altman: Unigram Probabilities", save_path=os.path.join(save_dir, "bland_altman_unigrams.png"))
    same_bigrams_metrics = calculate_bland_altman(real_same_bigrams, synthetic_same_bigrams, title="Bland-Altman: Same-visit Bigram Probabilities", save_path=os.path.join(save_dir, "bland_altman_same_bigrams.png"))
    seq_bigrams_metrics = calculate_bland_altman(real_seq_bigrams, synthetic_seq_bigrams, title="Bland-Altman: Sequential-visit Bigram Probabilities", save_path=os.path.join(save_dir, "bland_altman_seq_bigrams.png"))
    # Save metrics to JSON
    metrics = {
        "real_record_mean_length": real_lengths['mean_record_length'],
        "real_record_std_length": real_lengths['std_record_length'],
        "real_visit_mean_length": real_lengths['mean_visit_length'],
        "real_visit_std_length": real_lengths['std_visit_length'],
        "synthetic_record_mean_length": synthetic_lengths['mean_record_length'],
        "synthetic_record_std_length": synthetic_lengths['std_record_length'],
        "synthetic_visit_mean_length": synthetic_lengths['mean_visit_length'],
        "synthetic_visit_std_length": synthetic_lengths['std_visit_length'],
        "r2_unigrams": r2_unigrams,
        "r2_same_bigrams": r2_same_bigrams,
        "r2_seq_bigrams": r2_seq_bigrams,
        "unigrams_metrics": unigrams_metrics,
        "same_bigrams_metrics": same_bigrams_metrics,
        "seq_bigrams_metrics": seq_bigrams_metrics,
    }
    return metrics



def compute_prevalence(data_matrix, matrix_type):
    """
    Computes prevalence for the given matrix type.

    Args:
        data_matrix: Transformed data matrix (binary, count, or probability).
        matrix_type: Type of matrix ('binary', 'count', or 'probability').

    Returns:
        numpy.ndarray: Prevalence values for each code.
    """
    if matrix_type == "binary":
        # Proportion of patients with each code
        return np.mean(data_matrix > 0, axis=0)
    elif matrix_type in ["count", "probability"]:
        # Mean frequency or normalized probabilities
        return np.mean(data_matrix, axis=0)
    else:
        raise ValueError(f"Unsupported matrix type: {matrix_type}")


def fidelity_evaluation(real_data, synthetic_data, matrix_type):
    """
    Evaluates fidelity metrics using the specified matrix type.

    Args:
        real_data: Real dataset matrix.
        synthetic_data: Synthetic dataset matrix.
        matrix_type: The type of matrix used ('binary', 'count', or 'probability').

    Returns:
        dict: Fidelity metrics.
    """
    results = {}

    # Prevalence-based metrics
    real_prevalence = compute_prevalence(real_data, matrix_type)
    synthetic_prevalence = compute_prevalence(synthetic_data, matrix_type)
    results[f"{matrix_type}_mmd"] = np.abs(
        real_prevalence - synthetic_prevalence).max()
    # R² score
    results[f"{matrix_type}_r2_score"] = r2_score(
        real_prevalence, synthetic_prevalence)
    

    return results

# ======== HEATMAP CO-OCCURRENCE UTILITIES ========
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.sparse import coo_matrix, csr_matrix
from scipy.stats import pearsonr, spearmanr

def _token_texts_from_ids(tokenizer, token_ids):
    # fast decode of single IDs
    return [tokenizer.decode([tid]) for tid in token_ids]

def _code_filter_fn(prefixes):
    """return a predicate(token_text) that keeps codes whose text starts with any of prefixes."""
    prefixes = tuple(prefixes)
    def keep(s):
        return s.startswith(prefixes)
    return keep

def _topk_by_global_freq(real_data, synth_data, tokenizer, keep_fn, topk=200):
    """
    real_data/synth_data: list of patients; each patient is {'visits': [[int codes], ...]}
    keep_fn: predicate over decoded token text
    Return: kept_token_ids (list), kept_token_texts (list), id2col (dict)
    """
    freq = Counter()
    for source in (real_data, synth_data):
        for p in source:
            for v in p['visits']:
                for tid in v:
                    freq[tid] += 1
    # decode (lazy) and filter
    # (decode only the top ~N most frequent first to avoid decoding entire vocab)
    most = [tid for tid, _ in freq.most_common(200_000)]
    texts = _token_texts_from_ids(tokenizer, most)
    kept = [(tid, t) for tid, t in zip(most, texts) if keep_fn(t)]
    kept = kept[:topk]
    kept_ids  = [k for k,_ in kept]
    kept_text = [t for _,t in kept]
    id2col = {k:i for i,k in enumerate(kept_ids)}
    return kept_ids, kept_text, id2col

def _build_baskets(structured, id2col, level="patient"):
    """
    level='patient': one row per patient (union of codes across all visits)
    level='visit'  : one row per visit
    Returns sparse binary incidence X (rows=baskets, cols=codes)
    """
    rows, cols, data = [], [], []
    if level == "patient":
        for r, patient in enumerate(structured):
            seen = set()
            for v in patient['visits']:
                for tid in v:
                    c = id2col.get(tid, None)
                    if c is not None:
                        seen.add(c)
            for c in seen:
                rows.append(r); cols.append(c); data.append(1)
    elif level == "visit":
        r = 0
        for patient in structured:
            for v in patient['visits']:
                seen = set()
                for tid in v:
                    c = id2col.get(tid, None)
                    if c is not None:
                        seen.add(c)
                for c in seen:
                    rows.append(r); cols.append(c); data.append(1)
                r += 1
    else:
        raise ValueError("level must be 'patient' or 'visit'")
    if len(rows)==0:
        return csr_matrix((0, len(id2col)))
    nrows = max(rows)+1 if rows else 0
    ncols = len(id2col)
    return coo_matrix((data, (rows, cols)), shape=(nrows, ncols), dtype=np.float32).tocsr()

def _cooc_from_incidence(X):
    """
    Co-occurrence counts = X^T X (binary baskets).
    Also returns diag counts (occurrence per code).
    """
    C = (X.T @ X).astype(np.float64)    # [K,K]
    d = np.asarray(C.diagonal()).copy() # occ per code
    return C, d

def _cosine_from_cooc(C, d, eps=1e-8):
    """Cosine similarity from co-occurrence counts: C_ij / sqrt(d_i d_j)."""
    s = np.sqrt(np.outer(d, d)) + eps
    S = C.toarray() if hasattr(C, "toarray") else np.asarray(C)
    return (S / s)

def _upper_triangle_flat(M):
    idx = np.triu_indices_from(M, k=1)
    return M[idx]

def _matrix_similarity(M_real, M_syn):
    a = _upper_triangle_flat(M_real)
    b = _upper_triangle_flat(M_syn)
    # if all-zero edge case
    if np.all(a==0) and np.all(b==0):
        return {"pearson_r": 1.0, "spearman_r": 1.0, "rel_Frob": 0.0}
    pr = pearsonr(a, b).statistic if np.std(a)>0 and np.std(b)>0 else np.nan
    sr = spearmanr(a, b).correlation if np.std(a)>0 and np.std(b)>0 else np.nan
    rel_frob = np.linalg.norm(a-b) / (np.linalg.norm(a)+1e-12)
    return {"pearson_r": float(pr), "spearman_r": float(sr), "rel_Frob": float(rel_frob)}

def plot_cooccurrence_heatmaps(
    real_structured, synth_structured, tokenizer,
    prefixes=("ICD10CM_",),
    topk=200,
    level="patient",                   # 'patient' or 'visit'
    norm="cosine",                     # only 'cosine' implemented
    title="Co-occurrence (patient-level, diagnoses)",
    out_path="cooccurrence_heatmap.png",
    concept_code_labels=None
    ):
    """
    Build Real/Synth co-occurrence, normalize, plot heatmaps + Δ, and return metrics.
    Panels have identical widths; colorbars are added at the figure level.
    """
    keep_fn = _code_filter_fn(prefixes)
    kept_ids, kept_text, id2col = _topk_by_global_freq(
        real_structured, synth_structured, tokenizer, keep_fn, topk=topk
    )
    K = len(kept_ids)
    if K < 2:
        print("[heatmap] Not enough codes selected.")
        return None

    # --- Build incidence + co-occurrence ---
    Xr = _build_baskets(real_structured,  id2col, level=level)
    Xs = _build_baskets(synth_structured, id2col, level=level)

    Cr, dr = _cooc_from_incidence(Xr)
    Cs, ds = _cooc_from_incidence(Xs)

    if norm == "cosine":
        Mr = _cosine_from_cooc(Cr, dr)
        Ms = _cosine_from_cooc(Cs, ds)
        Y = sch.linkage(1-Mr, method="ward")
        Z = sch.dendrogram(Y, no_plot=True)
        cluster_idx = Z['leaves']
        Mr = Mr[cluster_idx][:, cluster_idx]
        Ms = Ms[cluster_idx][:, cluster_idx]
    else:
        raise NotImplementedError("Only 'cosine' normalization implemented here.")

    # Similarity numbers
    sim = _matrix_similarity(Mr, Ms)

    for name in ["Real", "Synthetic"]:
        fig, ax = plt.subplots(figsize=(6.2, 5.8))
        # consistent color limits for Real/Synth
        vmax = np.nanpercentile(np.concatenate([Mr.ravel(), Ms.ravel()]), 99)
        if name == "Real":
            hm = sns.heatmap(Mr, ax=ax, vmin=0, vmax=vmax, cmap="Reds",
                            cbar=False, square=True)
        else:
            hm = sns.heatmap(Ms, ax=ax, vmin=0, vmax=vmax, cmap="Reds",
                            cbar=False, square=True)
        # ax.set_title(name)
        cbar = fig.colorbar(hm.collections[0], ax=ax, location="right", fraction=0.046, pad=0.02)
        # --- Ticks: subsample to avoid clutter ---
        step = max(1, K // 30)
        tick_idx = np.arange(0, K, step)
        # Reorder labels by cluster order, then subsample by tick_idx
        reordered_text = [kept_text[i] for i in cluster_idx]
        print(reordered_text)
        if concept_code_labels is not None:
            print([concept_code_labels[i] for i in reordered_text])
        labels = [reordered_text[i].split("_", 1)[-1] for i in tick_idx] 

        ax.set_xticks(tick_idx + 0.5)
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticks(tick_idx + 0.5)
        ax.set_yticklabels(labels, fontsize=8)

        # fig.suptitle(title, y=1.03, fontsize=14)
        fig.tight_layout()
        fig.savefig( f"{out_path[:-4]}_{name.lower()}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)



    return {
        "metrics": sim,
        "kept_ids": kept_ids,
        "kept_text": kept_text,
        "out_path": out_path
    }