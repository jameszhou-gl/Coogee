import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt
# ==========================================
# 1. INPUT DATA (Paste your values here)
# ==========================================
ROUND_NUM = 1
if ROUND_NUM == 1:
    # first round
    # Ground Truth Labels (Order must match the scores below)
    # '0' = Synthetic, '1' = Real (or use strings 'synthetic', 'real')
    mapping_df = pd.read_csv("/srv/scratch/z3523916/facilitate-meds-files/output/coogee-final-sanity-check/2025-12-12_14_38_00-rm_know_emb_labtest/final_patient_timelines/clinical_review_round1/clinical_review_round1_mapping.csv")
    ground_truth = mapping_df.type.values.tolist()
    print("Ground Truth: ", ground_truth)
    # ground_truth = ['synthetic', 'real', 'synthetic', 'real', 'real', 'real', 'synthetic', 'synthetic', 'synthetic', 'real', 'real', 'real', 'real', 'real', 'synthetic', 'real', 'real', 'synthetic', 'synthetic', 'real', 'real', 'synthetic', 'synthetic', 'real', 'real', 'real', 'synthetic', 'synthetic', 'real', 'synthetic', 'real', 'real', 'synthetic', 'synthetic', 'synthetic', 'real', 'synthetic', 'synthetic', 'synthetic', 'synthetic']
    # Clinician Scores (Lists of length 40)
    armin_scores    = [4,9,5,9,9,9,6,3,9,8,9,9,8,9,4,8,8,8,5,8,9,8,4,7,6,6,8,6,8,2,7,7,8,5,8,9,5,4,5,5]
    matthew_scores  = [6,9,8,3,3,8,3,3,8,8,9,9,9,7,5,8,6,9,5,6,8,9,7,9,9,6,2,5,7,4,9,9,9,7,9,7,9,8,2,5]
    motahare_scores = [4,7,8,9,6,4,6,4,7,9,8,3,7,8,6,1,9,4,5,10,6,9,2,9,4,2,2,8,8,1,7,2,4,2,9,7,9,2,2,2]

    # LLMs
    gemini_3_0_pro_chat =  [3,6,2,4,3,4,1,2,5,3,9,5,4,6,7,5,5,6,1,4,8,5,7,3,4,5,7,4,3,6,7,5,5,2,4,3,6,2,8,6]
    gpt_5_api = [3, 6, 3, 5, 6, 7, 3, 4, 4, 6, 6, 8, 6, 6, 7, 4, 8, 4, 7, 7, 6, 6, 3, 6, 5, 6, 7, 5, 7, 3, 8, 6, 5, 4, 5, 6, 6, 4, 7, 6]
    gemini_3_pro_preview_api = [3,7,3,6,7,5,2,1,4,10,10,10,8,10,6,5,8,6,6,8,9,5,1,9,8,10,9,9,10,1,10,9,9,3,7,6,8,6,3,5]
    qwen_3_max_api = [4, 9, 4, 7, 8, 6, 2, 4, 6, 5, 9, 9, 6, 9, 8, 4, 9, 4, 5, 9, 9, 9, 6, 9, 6, 3, 9, 7, 8, 2, 9, 9, 7, 4, 7, 7, 4, 4, 4, 6]
    medgemma = [10, None, 1,10,10, 9,9,7,3,9,6,None, 2, None, 6,9,7,2,None,9]
    # Qwen3_30B_A3B_Instruct_2507 = [6, 9, 5, 7, 8, 7, 8, 7, 7, 7, 8, 8, 7, 8, 8, 7, 8, 6, 6, 9, 8, 7, 6, 7, 7, 6, 7, 8, 8, 6, 9, 8, 7, 5, 7, 8, 4, 6, 7, 7]
    Qwen3_30B_A3B_Instruct_2507 = [6, 8, 5, 7, 8, 7, 9, 7, 7, 7, 8, 8, 8, 7, 8, 8, 9, 6, 5, 9, 8, 7, 6, 7, 7, 6, 7, 8, 8, 6, 9, 9, 7, 4, 7, 8, 4, 7, 8, 7]
    # Qwen3_30B_wo_reasoning = [7, 8, 6, 7, 8, 7, 8, 7, 7, 7, 8, 7, 7, 7, 8, 7, 8, 6, 5, 9, 8, 7, 6, 9, 7, 6, 7, 8, 8, 7, 9, 8, 8, 5, 7, 7, 5, 7, 8, 7] 

if ROUND_NUM == 2:
    # second round
    mapping_df = pd.read_csv("output/coogee-final-sanity-check/2025-12-19_10_58_00-rm_know_emb_labtest_w_n_embd_factor/final_patient_timelines_for_clinical_review/clinical_review_round2/clinical_review_round2_mapping.csv")
    ground_truth = mapping_df.type.values.tolist()
    print("Ground Truth: ", ground_truth)
    # ground_truth = ['real', 'synthetic', 'real', 'synthetic', 'synthetic', 'synthetic', 'real', 'real', 'real', 'synthetic', 'synthetic', 'synthetic', 'synthetic', 'synthetic', 'real', 'synthetic', 'synthetic', 'real', 'real', 'synthetic', 'synthetic', 'real', 'real', 'synthetic', 'synthetic', 'synthetic', 'real', 'real', 'synthetic', 'real', 'synthetic', 'synthetic', 'real', 'real', 'real', 'synthetic', 'real', 'real', 'real', 'real']
    motahare_scores = [8,3,7,2,2,2,9,2,2,1,9,3,5,2,2,6,6,7,8,8,4,6,3,2,2,7,7,6,3,6,7,2,2,4,8,5,8,5,2,8]





# ==========================================
# 2. ANALYSIS SCRIPT
# ==========================================

def calculate_significance(labels, *score_lists):
    """
    Pools scores from multiple clinicians and runs Mann-Whitney U test.
    """
    real_group = []
    syn_group = []

    # Iterate through each record
    for i, label in enumerate(labels):
        if i>=len(score_lists[0]):
            continue
        # Normalize label to lowercase
        lbl = str(label).lower().strip()
        
        # Collect scores from all clinicians for this index
        for scores in score_lists:
            if scores[i] is None:
                continue
            if lbl in ['real', '1', 'true']:
                real_group.append(scores[i])
            else:
                syn_group.append(scores[i])

    # Convert to numpy arrays for stats
    real_group = np.array(real_group)
    syn_group = np.array(syn_group)

    # Descriptive Stats
    print(f"--- Descriptive Statistics ---")
    print(f"Real Records (n={len(real_group)}):      Mean = {np.mean(real_group):.2f} (SD={np.std(real_group):.2f})")
    print(f"Synthetic Records (n={len(syn_group)}): Mean = {np.mean(syn_group):.2f} (SD={np.std(syn_group):.2f})")
    
    # Mann-Whitney U Test
    # We use this because Likert scores (1-10) are ordinal, not strictly normal.
    u_stat, p_val = stats.mannwhitneyu(real_group, syn_group, alternative='two-sided')
    
    print(f"\n--- Statistical Test (Mann-Whitney U) ---")
    print(f"P-value: {p_val:.5e}")  # Scientific notation for very small numbers

    # Check Significance
    if p_val < 0.001:
        print(f"\n[RESULT] Significant? YES (p < 0.001)")
        print("You can report: 'The difference in realism scores was statistically significant (p < 0.001).'")
    else:
        print(f"\n[RESULT] Significant? NO (p = {p_val:.4f})")
        print("You should report the exact p-value.")

def alignment_heatmaps():
    scores_data = {
        'Record_ID': [f'{i+1:03d}' for i in range(40)],
        'Type': ground_truth,
        'Reviewer 3':    armin_scores,
        'Reviewer 1':  matthew_scores,
        'Reviewer 2': motahare_scores,
        'GPT-5':               gpt_5_api,
        'Gemini-3-Pro':        gemini_3_pro_preview_api,
        'Qwen-3-Max':          qwen_3_max_api,
        # 'Qwen-30B': Qwen3_30B_A3B_Instruct_2507,
    }
    df = pd.DataFrame(scores_data)
    heatmap_cols = ['Reviewer 1', 'Reviewer 2', 'Reviewer 3',
                    'GPT-5', 'Gemini-3-Pro', 'Qwen-3-Max']

    corr_matrix = df[heatmap_cols].corr()

    # Create custom annotations with centered dot instead of decimal point
    annot_labels = corr_matrix.applymap(lambda x: f"{x:.2f}".replace('.', r'$\cdot$'))
    
    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(corr_matrix, annot=annot_labels, fmt='', cmap='Reds', vmin=0, vmax=1, square=True, linewidths=0.5)
    
    # Format colorbar tick labels with centered dot
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels([f"{x:.1f}".replace('.', r'$\cdot$') for x in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])
    
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.title('Correlation of Realism Scores Between Evaluators', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/llm-as-judge/Figure4_Correlation_Matrix.pdf')

def violin_box_plots():
    scores_data = {
        'Record_ID': [f'{i+1:03d}' for i in range(40)],
        'Type': ground_truth,
        'Reviewer 1':  matthew_scores,
        'Reviewer 2': motahare_scores,
        'Reviewer 3':    armin_scores,
        'GPT-5':               gpt_5_api,
        'Gemini-3-Pro':        gemini_3_pro_preview_api,
        'Qwen-3-Max':          qwen_3_max_api,
        # 'Qwen-30B': Qwen3_30B_A3B_Instruct_2507,
    }
    df = pd.DataFrame(scores_data)

    # Reshape for Plotting
    df_melted = df.melt(id_vars=['Record_ID', 'Type'], 
                        var_name='Evaluator', 
                        value_name='Realism Score')

    # # 2. GENERATE VIOLIN PLOT (Recommended)
    # # -----------------------
    # plt.figure(figsize=(10, 6))
    # sns.violinplot(data=df_melted, x='Evaluator', y='Realism Score', hue='Type', 
    #             split=True, inner='quartile', 
    #             palette={'real': '#7AB2D5', 'synthetic': '#E29184'},
    #             linewidth=1.2  # Elegant thin lines
    #             )

    # plt.title('Realism Score Distributions: Human vs. LLMs', fontsize=14)
    # plt.ylabel('Realism Score (1-10)', fontsize=12)
    # plt.xlabel('')
    # plt.ylim(-2, 12)
    # plt.yticks(np.arange(1, 11))
    # plt.legend(loc='upper right', fontsize=12)
    # plt.grid(False)
    # plt.tight_layout()
    # plt.savefig('results/llm-as-judge/Figure4_Violin_Plot.pdf', dpi=300)

    # 3. GENERATE BOX PLOT
    # --------------------
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df_melted, x='Evaluator', y='Realism Score', hue='Type', width=0.6, native_scale=True,
                palette={'real': '#7AB2D5', 'synthetic': '#E29184'})
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.title('Realism Score Distributions: Human vs. LLMs', fontsize=12)
    plt.ylabel('Realism Score (1-10)', fontsize=10)
    plt.xlabel('')
    plt.ylim(0, 11)
    sns.despine()
    plt.legend(loc='lower right')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('results/llm-as-judge/Figure4_Box_Plot.pdf', dpi=300)

def find_top_cheaters(ground_truth, *score_lists):
    """
    Identifies synthetic records with the highest average realism scores.
    """
    # 1. Aggregate scores into a DataFrame
    # Note: score_lists order must match the columns below
    df_scores = pd.DataFrame({
        'Record_ID': [f'{i+1:03d}' for i in range(len(ground_truth))],
        'Type': ground_truth,
        'Human_Avg': np.mean([score_lists[0], score_lists[1], score_lists[2]], axis=0),
        'LLM_Avg': np.mean([score_lists[3], score_lists[4], score_lists[5]], axis=0),
        'Total_Avg': np.mean(score_lists, axis=0)
    })

    # 2. Filter for SYNTHETIC records only
    # Adjust 'synthetic'/'0' based on your actual CSV values
    syn_df = df_scores[df_scores['Type'].astype(str).str.lower().isin(['synthetic', '0', 'false'])].copy()
    
    # 3. Sort by highest Total Average score
    top_cheaters = syn_df.sort_values(by='Total_Avg', ascending=False).head(5)
    
    print("\n=== TOP 5 SYNTHETIC RECORDS ('CHEATERS') ===")
    print(top_cheaters[['Record_ID', 'Total_Avg', 'Human_Avg', 'LLM_Avg']].to_string(index=False))
    
    return top_cheaters.iloc[0]['Record_ID']

def calculate_pairwise_icc(df, raters):
    """
    Calculates the pairwise Intraclass Correlation Coefficient (ICC).
    We use ICC(2,1) (Single random raters, Absolute Agreement) 
    because we want to know if they assign the SAME score, not just correlated scores.
    """
    matrix = pd.DataFrame(index=raters, columns=raters, dtype=float)
    
    for r1 in raters:
        for r2 in raters:
            if r1 == r2:
                matrix.loc[r1, r2] = 1.0
            else:
                # Prepare data in "Long" format for Pingouin
                subset = df[[r1, r2]].reset_index()
                subset.columns = ['Subject', 'Rater_A', 'Rater_B']
                subset_long = pd.melt(subset, id_vars='Subject', 
                                      value_vars=['Rater_A', 'Rater_B'], 
                                      var_name='Rater', value_name='Score')
                
                # Calculate ICC
                try:
                    icc_res = pg.intraclass_corr(data=subset_long, targets='Subject', 
                                                 raters='Rater', ratings='Score')
                    
                    # Extract ICC2 (Single random raters, absolute agreement)
                    # Use 'ICC2' or 'ICC3' depending on your specific definition. 
                    # ICC2 is standard for "Absolute Agreement".
                    val = icc_res.set_index('Type').loc['ICC2']['ICC']
                    
                    # Clip to 0-1 range (ICC can technically be negative if disagreement is worse than random)
                    matrix.loc[r1, r2] = max(0, val)
                except:
                    matrix.loc[r1, r2] = np.nan
    return matrix

# ================= 3. PLOT HEATMAP =================

def plot_ICC_heatmap(matrix, title, filename):
    plt.figure(figsize=(6, 4))
    
    # Custom format to replace decimal dot with center dot (e.g., 0·95)
    annot_labels = matrix.applymap(lambda x: f"{x:.2f}".replace('.', r'$\cdot$'))
    
    # Plot
    ax = sns.heatmap(matrix, annot=annot_labels, fmt='', cmap='Reds', 
                     vmin=0, vmax=1, square=True, linewidths=0.5)
    
    # Fix Colorbar ticks
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels([f"{x:.1f}".replace('.', r'$\cdot$') for x in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])
    
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300) # Uncomment to save
    # plt.show()


# EXECUTE (Pass your lists in order: Armin, Matthew, Motahare, GPT5, Gemini, Qwen)
best_record_id = find_top_cheaters(
    ground_truth, 
    armin_scores, matthew_scores, motahare_scores, 
    gpt_5_api, gemini_3_pro_preview_api, qwen_3_max_api
)

# Run the function
# calculate_significance(ground_truth, motahare_scores)
alignment_heatmaps()
violin_box_plots()
# ================= 4. RUN =================
if ROUND_NUM == 1:
    scores_data = {
        'Record_ID': [f'{i+1:03d}' for i in range(40)],
        'Reviewer 1': matthew_scores,
        'Reviewer 2': motahare_scores,
        'Reviewer 3': armin_scores,
        'GPT-5': gpt_5_api,
        'Gemini-3-Pro': gemini_3_pro_preview_api,
        'Qwen-3-Max': qwen_3_max_api,
    }
    df = pd.DataFrame(scores_data)
    evaluators = ['Reviewer 1', 'Reviewer 2', 'Reviewer 3', 'GPT-5', 'Gemini-3-Pro', 'Qwen-3-Max']
    icc_matrix = calculate_pairwise_icc(df, evaluators)
    plot_ICC_heatmap(icc_matrix, 'Inter-Rater Reliability (ICC) Between Evaluators', 'results/llm-as-judge/Figure4_ICC.pdf')
