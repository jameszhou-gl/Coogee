# evaluation.py
import os, json, argparse
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from model.local_tokenizer import LocalTokenizer

# evaluation.py
# Time-gap fidelity: pooled time-gap ECDF, per-visit LOS, inter-visit gaps.
# Minimal dependencies; pure matplotlib (no seaborn).
import os
import json
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, ks_2samp

# --------------------------- config ---------------------------

# Range bounds (min, max) in minutes for each time-gap token
GAP_RANGES = {
    '_<=5m': (0, 5),
    '_5m-15m': (5, 15),
    '_15m-1h': (15, 60),
    '_1h-2h': (60, 120),
    '_2h-6h': (120, 360),
    '_6h-12h': (360, 720),
    '_12h-1d': (720, 1440),
    '_1d-3d': (1440, 4320),
    '_3d-1w': (4320, 10080),
    '_1w-2w': (10080, 20160),
    '_2w-1mt': (20160, 43200),
    '_1mt-3mt': (43200, 129600),
    '_3mt-6mt': (129600, 259200),
    '_>6mt': (259200, 525600*5),  # up to 5 years
}
GAP_TOKENS = set(GAP_RANGES.keys())


# --------------------------- helpers ---------------------------

def _decode_cache(tokenizer):
    """Return a small closure to decode ids with memoization."""
    cache: Dict[int, str] = {}
    def dec(tid: int) -> str:
        if tid not in cache:
            cache[tid] = tokenizer.decode([tid])
        return cache[tid]
    return dec


def _is_gap(tok: str) -> bool:
    return tok in GAP_TOKENS


def _gap_minutes(tok: str) -> float:
    lo, hi = GAP_RANGES[tok]
    return np.random.uniform(lo, hi)


def ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Empirical CDF (x, y)."""
    if len(values) == 0:
        return np.array([]), np.array([])
    x = np.sort(values)
    y = np.arange(1, len(values) + 1) / float(len(values))
    return x, y


def _summary_1d(x: np.ndarray) -> Dict[str, float]:
    if len(x) == 0:
        return dict(n=0, median=np.nan, iqr=np.nan, mean=np.nan, std=np.nan)
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    return dict(
        n=int(len(x)),
        median=float(q50),
        q25=float(q25),
        q75=float(q75),
        iqr=float(q75 - q25),
        mean=float(np.mean(x)),
        std=float(np.std(x, ddof=1))
    )


# --------------------------- core parsing ---------------------------
        
def extract_time_quantities(
    sequences: List[List[int]],
    tokenizer,
    num_first_tokens: int = 6,
) -> Dict[str, np.ndarray]:
    """
    Parse raw token-id sequences to:
      - pooled_gaps: all gap tokens anywhere (minutes)
      - los_per_visit: sum of gap tokens within each visit (minutes)
      - intervisit_gaps: sum of gap tokens between END_VISIT and next START_VISIT (minutes)
    Assumes standard markers START_VISIT / END_VISIT / END_RECORD and that
    time-gap tokens appear as standalone tokens between events.
    """
    dec = _decode_cache(tokenizer)
    tid_START_VISIT = tokenizer.convert_tokens_to_ids('START_VISIT')
    tid_END_VISIT   = tokenizer.convert_tokens_to_ids('END_VISIT')
    tid_END_RECORD  = tokenizer.convert_tokens_to_ids('END_RECORD')

    pooled_gaps = []
    los_per_visit = []
    intervisit_gaps = []

    for seq in sequences:
        # skip demographics header
        toks = seq[num_first_tokens:]

        in_visit = False
        curr_los = 0.0
        pending_intervisit = 0.0  # sums gaps after END_VISIT, before next START_VISIT

        i = 0
        while i < len(toks):
            tok_id = int(toks[i])
            tok = dec(tok_id)

            if tok_id == tid_START_VISIT:
                # starting a new visit; if any pending inter-visit time accumulated, store it
                if pending_intervisit > 0:
                    intervisit_gaps.append(pending_intervisit)
                    pending_intervisit = 0.0
                in_visit = True
                curr_los = 0.0
                i += 1
                continue

            if tok_id == tid_END_VISIT or tok_id == tid_END_RECORD:
                if in_visit:
                    if curr_los > 1:
                        los_per_visit.append(curr_los) # e.g., 1 min
                    curr_los = 0.0
                    in_visit = False
                # if this was END_RECORD, we will exit the loop after handling any stray gaps
                i += 1
                continue

            # normal token; if it's a time-gap token, attribute accordingly
            if _is_gap(tok):
                gap_min = _gap_minutes(tok)
                pooled_gaps.append(gap_min)
                if in_visit:
                    curr_los += gap_min
                else:
                    # we're outside a visit (e.g., between visits)
                    pending_intervisit += gap_min

            i += 1

        # flush any trailing inter-visit time at record end (optional; usually 0)
        if pending_intervisit > 0:
            intervisit_gaps.append(pending_intervisit)

    return dict(
        pooled_gaps=np.asarray(pooled_gaps, dtype=float),
        los_per_visit=np.asarray(los_per_visit, dtype=float),
        intervisit_gaps=np.asarray(intervisit_gaps, dtype=float),
    )


# --------------------------- plotting ---------------------------

def plot_ecdf(real_vals, synth_vals, xlabel, title, out_png, use_log10=True):
    """Simple ECDF overlay plot."""
    rv = np.asarray(real_vals, float)
    sv = np.asarray(synth_vals, float)

    if use_log10:
        rv = np.log10(np.clip(rv, 1e-6, None))
        sv = np.log10(np.clip(sv, 1e-6, None))

    x_r, y_r = ecdf(rv)
    x_s, y_s = ecdf(sv)

    plt.figure(figsize=(7, 4.2))
    if len(x_r):
        plt.step(x_r, y_r, where="post", linewidth=2)
    if len(x_s):
        plt.step(x_s, y_s, where="post", linewidth=2)

    if use_log10:
        plt.xlabel(f"log10({xlabel})", fontsize=12)
    else:
        plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("ECDF", fontsize=12)
    plt.title(title, fontsize=12)
    plt.legend(["Real", "Synthetic"])
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# --------------------------- metrics ---------------------------

def distribution_metrics(real_vals, synth_vals, for_emd_log10=True) -> Dict[str, float]:
    """
    Summary stats + distances for two 1D distributions.
    - EMD on log10 scale (safer for heavy-tailed times)
    - KS statistic on original scale (complements EMD)
    - median / IQR differences on original scale
    """
    rv = np.asarray(real_vals, float)
    sv = np.asarray(synth_vals, float)

    # summaries on original scale (minutes)
    summ_r = _summary_1d(rv)
    summ_s = _summary_1d(sv)

    # KS on original scale
    ks = ks_2samp(rv, sv).statistic if len(rv) and len(sv) else np.nan

    # EMD on log scale (more stable for many orders of magnitude)
    if for_emd_log10:
        rv_log = np.log10(np.clip(rv, 1e-6, None))
        sv_log = np.log10(np.clip(sv, 1e-6, None))
        # Histogram at common bins for EMD
        bins = np.linspace(min(rv_log.min(), sv_log.min()),
                           max(rv_log.max(), sv_log.max()), 256)
        pr, _ = np.histogram(rv_log, bins=bins, density=True)
        ps, _ = np.histogram(sv_log, bins=bins, density=True)
        centers = 0.5 * (bins[1:] + bins[:-1])
        emd = wasserstein_distance(centers, centers, pr, ps)
    else:
        emd = np.nan

    # deltas on original scale
    med_delta = (summ_s["median"] - summ_r["median"]) if summ_r["n"] and summ_s["n"] else np.nan
    iqr_delta = (summ_s["iqr"] - summ_r["iqr"]) if summ_r["n"] and summ_s["n"] else np.nan

    return {
        # distances
        "emd_log10": float(emd),
        "ks_stat": float(ks),
        # summary (real)
        "real_n": float(summ_r["n"]),
        "real_median": float(summ_r["median"]),
        "real_q25": float(summ_r["q25"]),     # <-- Added
        "real_q75": float(summ_r["q75"]),     # <-- Added
        "real_iqr": float(summ_r["iqr"]),
        "real_mean": float(summ_r["mean"]),
        "real_std": float(summ_r["std"]),
        # summary (synthetic)
        "synth_n": float(summ_s["n"]),
        "synth_median": float(summ_s["median"]),
        "synth_q25": float(summ_s["q25"]),    # <-- Added
        "synth_q75": float(summ_s["q75"]),    # <-- Added
        "synth_iqr": float(summ_s["iqr"]),
        "synth_mean": float(summ_s["mean"]),
        "synth_std": float(summ_s["std"]),
        # deltas
        "delta_median": float(med_delta),
        "delta_iqr": float(iqr_delta)
    }


# --------------------------- top-level API ---------------------------

def run_timegap_evaluation(
    real_sequences: List[List[int]],
    synth_sequences: List[List[int]],
    tokenizer,
    num_first_tokens: int,
    out_dir: str,
) -> Dict[str, Dict[str, float]]:
    """
    Main entry: extracts time quantities and saves three ECDF plots:
        - pooled time-gap tokens (across all events)  -> timegaps_ecdf.png
        - length-of-stay per visit                    -> los_ecdf.png
        - inter-visit gaps                            -> intervisit_ecdf.png
    Returns a dict with metrics for each of those distributions.
    """
    os.makedirs(out_dir, exist_ok=True)

    real_q = extract_time_quantities(real_sequences, tokenizer, num_first_tokens)
    synth_q = extract_time_quantities(synth_sequences, tokenizer, num_first_tokens)

    # # 1) pooled time-gap tokens (minutes), ECDF on log10
    # plot_ecdf(
    #     real_q["pooled_gaps"], synth_q["pooled_gaps"],
    #     xlabel="minutes between events",
    #     title="ECDF of time gaps (all events)",
    #     out_png=os.path.join(out_dir, "timegaps_ecdf.png"),
    #     use_log10=True,
    # )
    pooled_metrics = distribution_metrics(real_q["pooled_gaps"], synth_q["pooled_gaps"])

    # # 2) length of stay per visit (minutes)
    # plot_ecdf(
    #     real_q["los_per_visit"], synth_q["los_per_visit"],
    #     xlabel="minutes per visit (LOS)",
    #     title="ECDF of length-of-stay per visit",
    #     out_png=os.path.join(out_dir, "los_ecdf.png"),
    #     use_log10=True,
    # )
    los_metrics = distribution_metrics(real_q["los_per_visit"], synth_q["los_per_visit"])

    # # 3) inter-visit gaps (minutes)
    # plot_ecdf(
    #     real_q["intervisit_gaps"], synth_q["intervisit_gaps"],
    #     xlabel="minutes between visits",
    #     title="ECDF of inter-visit gaps",
    #     out_png=os.path.join(out_dir, "intervisit_ecdf.png"),
    #     use_log10=True,
    # )
    intervisit_metrics = distribution_metrics(real_q["intervisit_gaps"], synth_q["intervisit_gaps"])

    results = {
        "pooled_timegaps": pooled_metrics,
        "visit_los": los_metrics,
        "intervisit_gaps": intervisit_metrics,
    }

    with open(os.path.join(out_dir, "timegap_fidelity.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results

# =========================
# 0) Small utils
# =========================
def _load_config_yaml(path):
    # minimal reader that does not depend on your utils
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

# def _load_tokenizer(log_dir):
#     from transformers import AutoTokenizer
#     cfg = _load_config_yaml(os.path.join(log_dir, "config.yaml"))
#     repo = cfg.get("huggingface", {}).get("repo")
#     if not repo:
#         raise ValueError("Missing huggingface.repo in config.yaml")
#     return AutoTokenizer.from_pretrained(repo)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

def plot_ecdf_with_ci_and_fit_two(
    real_minutes,
    synth_minutes,
    title="Empirical CDF",
    out_path="ecdf_with_ci.png",
    log10_x=True,
    alpha=0.05,
    colors=dict(real="#4291C2", synth="#D7634F", fit="#2ca02c"),
    method="dkw",
    n_boot=500,
    show_fit=False
):
    """
    Overlay Real and Synthetic ECDFs with 95% DKW bands.
    Optionally also fit log-normal CDF to Real data for reference.
    """
    def _prep(data):
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data) & (data > 0)]
        return data

    def _ecdf(x):
        x = np.sort(x)
        y = np.arange(1, len(x)+1) / len(x)
        return x, y

    def _dkw_band(n, y, alpha):
        eps = np.sqrt(np.log(2.0/alpha) / (2.0 * n))
        return np.clip(y - eps, 0, 1), np.clip(y + eps, 0, 1)
    
    def _bootstrap_band(x, x_grid, alpha, B=500, rng=None):
        # pointwise bootstrap band for ECDF at x_grid
        if rng is None: rng = np.random.default_rng(1337)
        n = x.size
        boot_vals = np.empty((B, x_grid.size), float)
        for b in range(B):
            xb = rng.choice(x, size=n, replace=True)
            xb.sort()
            boot_vals[b] = np.searchsorted(xb, x_grid, side="right") / n
        lo = np.percentile(boot_vals, 100*(alpha/2), axis=0)
        hi = np.percentile(boot_vals, 100*(1-alpha/2), axis=0)
        return lo, hi

    real = _prep(real_minutes)
    synth = _prep(synth_minutes)
    if len(real) == 0 or len(synth) == 0:
        print("[plot_ecdf_with_ci_and_fit_two] Empty data; skipped.")
        return

    # ECDFs
    xr, yr = _ecdf(real)
    xs, ys = _ecdf(synth)

    # common x grid for bootstrap bands (quantile grid is stable)
    x_all = np.concatenate([real, synth])
    x_grid = np.quantile(x_all, np.linspace(0.01, 0.99, 200))

    # choose CI method
    if method == "dkw":
        lower_r, upper_r = _dkw_band(real.size, yr, alpha)
        lower_s, upper_s = _dkw_band(synth.size, ys, alpha)
        x_band_r, x_band_s = xr, xs
    elif method == "bootstrap":
        lower_r, upper_r = _bootstrap_band(np.sort(real), x_grid, alpha, B=n_boot)
        lower_s, upper_s = _bootstrap_band(np.sort(synth), x_grid, alpha, B=n_boot)
        x_band_r = x_band_s = x_grid
    else:
        raise ValueError("method must be 'dkw' or 'bootstrap'")
    
        # optional log scale on x
    if log10_x:
        xr, xs = np.log10(xr), np.log10(xs)
        x_band_r = np.log10(x_band_r)
        x_band_s = np.log10(x_band_s)
        xlabel = "minutes (log10 scale)"
    else:
        xlabel = "minutes"

    # log-normal fit to Real
    if show_fit:
        s, loc, scale = lognorm.fit(real, floc=0)
        x_fit = np.linspace(real.min(), real.max(), 1000)
        cdf_fit = lognorm.cdf(x_fit, s, loc=loc, scale=scale)
        if log10_x: x_fit = np.log10(x_fit)

    plt.figure(figsize=(10, 8))

    # Real ECDF + band
    plt.step(xr, yr, where="post", color=colors["real"], lw=2, label="Real ECDF")
    # plt.fill_between(x_band_r, lower_r, upper_r, step=None if method=="bootstrap" else "post",
    #                  color=colors["real"], alpha=0.15, label="Real 95% CI")

    # Synthetic ECDF + band
    plt.step(xs, ys, where="post", color=colors["synth"], lw=2, label="Synthetic ECDF")
    # plt.fill_between(x_band_s, lower_s, upper_s, step=None if method=="bootstrap" else "post",
    #                  color=colors["synth"], alpha=0.15, label="Synthetic 95% CI")

    # Fitted CDF (to Real)
    if show_fit:
        plt.plot(x_fit, cdf_fit, color=colors["fit"], lw=2, ls="--", label="Log-normal fit (Real)")
    # Set tick label font sizes
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel("Cumulative probability", fontsize=24)
    plt.title(title, fontsize=22)
    plt.ylim(0, 1.02)
    plt.legend(frameon=False, fontsize=22)

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--LOG_DIR", required=True, type=str)
    ap.add_argument("--top_p", type=float, default=0.98)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    # paths
    syn_dir = os.path.join(args.LOG_DIR, f"synthetic_data_topp_{args.top_p}_temperature_{args.temperature}")
    save_dir = os.path.join(f'{syn_dir}', "time_fidelity")
    os.makedirs(save_dir, exist_ok=True)

    # load data
    tokenizer = LocalTokenizer(os.path.join(args.LOG_DIR, "tokenizer", "vocab.csv"))
    real_df = pl.read_parquet(os.path.join(args.LOG_DIR, "tokenized_sequences", "test.parquet"))
    syn_df  = pl.read_parquet(os.path.join(syn_dir, "synthetic_data.parquet"))
    
    # real test sequences (ids) and synthetic sequences (ids)
    real_sequences = real_df['concept_token_ids'].to_list()
    synth_sequences = syn_df['synthetic_sequence'].to_list()

    _ = run_timegap_evaluation(
        real_sequences=real_sequences,
        synth_sequences=synth_sequences,
        tokenizer=tokenizer,          # HF tokenizer you already loaded
        num_first_tokens=6,           # START_RECORD + 5 demographics
        out_dir=save_dir,   # where to save the three plots + json
    )
        # run timegap extraction
    real_eval  = extract_time_quantities(real_sequences, tokenizer, num_first_tokens=6)
    synth_eval = extract_time_quantities(synth_sequences, tokenizer, num_first_tokens=6)

    # now you have the arrays
    real_minutes        = real_eval["pooled_gaps"]
    real_los_minutes    = real_eval["los_per_visit"]
    real_intervisit_min = real_eval["intervisit_gaps"]
    # print(real_intervisit_min)
    synth_minutes        = synth_eval["pooled_gaps"]
    synth_los_minutes    = synth_eval["los_per_visit"]
    synth_intervisit_min = synth_eval["intervisit_gaps"]

    # save ECDF plots with CI + fitted curve
    plot_ecdf_with_ci_and_fit_two(
        real_minutes,
        synth_minutes,
        title="Empirical CDF of pooled event-to-event time gaps",
        out_path=os.path.join(save_dir, "ecdf_event_to_event_time_gaps.png")
    )

    plot_ecdf_with_ci_and_fit_two(
        real_los_minutes,
        synth_los_minutes,
        title="Empirical CDF of hospital length-of-stay per visit",
        out_path=os.path.join(save_dir, "ecdf_hospital_length_of_stay_per_visit.png")
    )

    plot_ecdf_with_ci_and_fit_two(
        real_intervisit_min,
        synth_intervisit_min,
        title="Empirical CDF of inter-visit time gaps",
        out_path=os.path.join(save_dir, "ecdf_inter_visit_time_gaps.png")
    )


if __name__ == "__main__":
    main()
    
# python -m evaluation.eval_fidelity_time --LOG_DIR output/coogee-final-sanity-check/2025-12-19_10_58_00-rm_know_emb_labtest_w_n_embd_factor