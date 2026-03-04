import numpy as np
import matplotlib.pyplot as plt
import os


def data_to_binary_matrix(data, code_vocab_size):
    """
    Converts data to a binary matrix.
    Each row represents a patient, each column represents a unique code, and a 1 indicates the presence of the code.
    """
    binary_matrix = np.zeros((len(data), code_vocab_size))
    
    for i, patient in enumerate(data):
        unique_codes = set(
            code for visit in patient['visits'] for code in visit)
        for code in unique_codes:
            binary_matrix[i, code] = 1
    return binary_matrix


def data_to_count_matrix(data, code_vocab_size):
    """
    Converts data to a count matrix.
    Each row represents a patient, each column represents a unique code, and the value indicates the count of the code.
    """
    count_matrix = np.zeros((len(data), code_vocab_size), dtype=int)
    for i, patient in enumerate(data):
        for visit in patient['visits']:
            for code in visit:
                count_matrix[i, code] += 1
    return count_matrix


def transform_data_matrix(data, code_vocab_size, matrix_type):
    """
    Transforms data into binary, count, or probability matrix.

    Args:
        data: Input data with the shape of [num_patients, code_vocab_size].
        code_vocab_size: Vocabulary size of codes.
        matrix_type: Type of matrix ('binary', 'count', 'probability').

    Returns:
        numpy.ndarray: Transformed matrix with the shape of [num_patients, code_vocab_size].
    """
    if matrix_type == "binary":
        return data_to_binary_matrix(data, code_vocab_size)
    elif matrix_type == "count":
        return data_to_count_matrix(data, code_vocab_size)
    elif matrix_type == "probability":
        count_matrix = data_to_count_matrix(data, code_vocab_size)
        row_sums = count_matrix.sum(axis=1, keepdims=True)
        return count_matrix / (row_sums + 1e-5)  # Avoid division by zero
    else:
        raise ValueError(f"Unsupported matrix type: {matrix_type}")
    
    

def plot_real_vs_syn(out_dir, results=None, task_name="Readmission"):
    metrics = ["Precision", "Recall", "F1_score", "ROC_AUC"]
    real_vals = [results["Real"][m] for m in metrics]
    syn_vals  = [results["Syn"][m]  for m in metrics]

    x = range(len(metrics))
    width = 0.3

    # fig, ax = plt.subplots(figsize=(7, 4))
    plt.figure(figsize=(7, 4))
    plt.style.use('default')

    # Create the main plot
    ax = plt.gca()
    ax.set_facecolor('white')  # White background
    ax.yaxis.set_minor_locator(plt.NullLocator())
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # grouped bars
    real_color = '#4291C2'
    syn_color = '#D7634F'
    bars_syn  = ax.bar([i - width/2 for i in x], syn_vals,  width, label="Synthetic", color=syn_color)
    bars_real = ax.bar([i + width/2 for i in x], real_vals, width, label="Real", color=real_color)

    # axes/labels
    # ax.set_ylim(0, 1.5)
    ax.set_ylim(0, 1.2)   # extend y-limit
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])  # only show ticks up to 1.0
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    # ax.set_ylabel("Score", fontsize=18)
    ax.set_title(f"{task_name}", fontsize=18)
    ax.legend(loc="upper right", fontsize=10)
    # ax.legend(loc="center left", bbox_to_anchor=(0.5, 1.1), frameon=False, fontsize=12)

    plt.tight_layout()
    print(f"Saving plot to {os.path.join(out_dir, f'{task_name}_real_vs_syn.png')}")
    plt.savefig(os.path.join(out_dir, f"{task_name}_real_vs_syn.png"), dpi=300)
    plt.show()
    
    