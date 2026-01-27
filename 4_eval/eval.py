import torch
import pandas as pd
import json
import numpy as np
from recbole.evaluator import Evaluator
from recbole.quick_start import load_data_and_model

from utils import get_latest_checkpoint


def calculate_average_pcf(reranked_items_list, id_to_asin, scores_dict, dataset):
    """Calculate mean PCF across all users and their top-k items."""
    vals = [v for v in scores_dict.values() if v is not None]
    default_val = sum(vals) / len(vals) if vals else 0
    total_pcf, count = 0, 0
    for user_items in reranked_items_list:
        for item_id in user_items:
            token = dataset.id2token(dataset.iid_field, int(item_id))
            asin = id_to_asin.get(int(token))
            pcf = scores_dict.get(asin, default_val)
            total_pcf += pcf
            count += 1
    return total_pcf / count if count > 0 else 0


def get_pcf_reduction_perc(grp, col_name='PCF_Avg'):
    """Calculates the PCF reduction in terms of % for a specific column."""
    base = grp[grp['Alpha'] == 1.0][col_name].values
    new_col = col_name + '_Reduction_%'
    grp[new_col] = round(((base[0] - grp[col_name]) / base[0]) * 100, 2) if base.size > 0 else 0
    return grp


def evaluate_model_results(file_path, config, dataset, id_to_asin, co2e_scores, gt_pcf):
    """Load re-ranked results and compute RecBole, estimated and real sustainability metrics."""
    data = torch.load(file_path, weights_only=False)
    k = config['topk'][0]

    # RecBole metrics
    pos_matrix = torch.tensor(data['pos_matrix'], device=config['device'])[:, :k]
    pos_len = torch.tensor(data['pos_len'], device=config['device']).view(-1, 1)
    struct = {'rec.topk': torch.cat((pos_matrix, pos_len), dim=1).cpu()}
    rec_results = Evaluator(config).evaluate(struct)

    # Sustainability metrics (estimated by LLM)
    avg_pcf_est = calculate_average_pcf(data['reranked_items'], id_to_asin, co2e_scores, dataset)

    # Sustainability metrics (ground truth)
    avg_pcf_real = calculate_average_pcf(data['reranked_items'], id_to_asin, gt_pcf, dataset)

    # Merge results
    res = {
        'Model': data['model'],
        'Alpha': data['alpha'],
        'PCF_Est': round(avg_pcf_est, 4),
        'PCF_Real': round(avg_pcf_real, 4)
    }
    res.update({m: round(v, 4) for m, v in rec_results.items()})

    return res


# =============================================
# Setup
# =============================================
# Load configuration and dataset
config, _, dataset, *_ = load_data_and_model(
    model_file=get_latest_checkpoint("LightGCN_best"),
)

# Ground Truth CO2 values
gt_pcf = {
    "B0BLHNNSGG": 10.78, "B0148NPAT0": 5.98, "B00L1Y11D4": 8.21,
    "B0B35JDFPL": 11.14, "B01LZAK8MM": 9.48, "B0BVBDXFHM": 7.93,
    "B0BK3LYMR2": 7.80, "B01AROOL12": 8.05, "B0BVZZ36ZL": 1.73,
    "B0BMVBQZ9T": 4.07
}

# Load C02 score estimations (LLM)
co2e_scores = {}
llm_file = "../1_pcf/few_results/gpt_o3_mini_results.json"
with open(llm_file, "r", encoding="utf-8") as f:
    data_list = json.load(f)
    for data in data_list:
        if data.get("co2e_kg") is not None:
            co2e_scores[data["parent_asin"]] = data["co2e_kg"]

# Load item_index -> parent_asin mapping
item_map_df = pd.read_csv("../2_recbole/process_data/maps/item_map.tsv", sep='\t')
id_to_asin = dict(zip(item_map_df['item_index'], item_map_df['parent_asin']))

# Set result files to analyze
files = [
    '../3_reranking/few_results/reranked_results_BPR_alpha_1.0_gpt_o3_mini.pth',
    '../3_reranking/few_results/reranked_results_BPR_alpha_0.75_gpt_o3_mini.pth',
    '../3_reranking/few_results/reranked_results_BPR_alpha_0.5_gpt_o3_mini.pth',
    '../3_reranking/few_results/reranked_results_BPR_alpha_0.25_gpt_o3_mini.pth',
    '../3_reranking/few_results/reranked_results_LightGCN_alpha_1.0_gpt_o3_mini.pth',
    '../3_reranking/few_results/reranked_results_LightGCN_alpha_0.75_gpt_o3_mini.pth',
    '../3_reranking/few_results/reranked_results_LightGCN_alpha_0.5_gpt_o3_mini.pth',
    '../3_reranking/few_results/reranked_results_LightGCN_alpha_0.25_gpt_o3_mini.pth',
]

# =============================================
# Evaluation
# =============================================
# 1. Calculate LLM Estimation Errors
common_asins = [a for a in gt_pcf if a in co2e_scores]
y_true = [gt_pcf[a] for a in common_asins]
y_pred = [co2e_scores[a] for a in common_asins]

mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
rmse = np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

print("\n" + "=" * 50)
print(f" LLM Estimation Error")
print(f" MAE: {mae:.4f} kg")
print(f" RMSE: {rmse:.4f} kg")
print("=" * 50)

# 2. Evaluate model recommendations
results = []
for f in files:
    results.append(evaluate_model_results(f, config, dataset, id_to_asin, co2e_scores, gt_pcf))

# Formatting result table
df = pd.DataFrame(results)
df = df.groupby('Model', group_keys=False).apply(lambda x: get_pcf_reduction_perc(x, 'PCF_Est'))
df = df.groupby('Model', group_keys=False).apply(lambda x: get_pcf_reduction_perc(x, 'PCF_Real'))

cols = (
        ['Model', 'Alpha', 'PCF_Est', 'PCF_Est_Reduction_%', 'PCF_Real', 'PCF_Real_Reduction_%'] +
        [c for c in df.columns if
         c not in ['Model', 'Alpha', 'PCF_Est', 'PCF_Est_Reduction_%', 'PCF_Real', 'PCF_Real_Reduction_%']]
)

# Display results
print("\n" + "=" * 150 + "\n Results \n" + "=" * 150)
print(df[cols].to_string(index=False))
print("=" * 150)

# Save results
df.to_csv('few_results/gpt_o3_mini_evaluation_results.csv', index=False)