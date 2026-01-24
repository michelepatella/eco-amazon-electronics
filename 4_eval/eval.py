import torch
import pandas as pd
import json
from recbole.evaluator import Evaluator
from recbole.quick_start import load_data_and_model

from utils import get_latest_checkpoint


def calculate_average_pcf(reranked_items_list, id_to_asin, co2e_scores, dataset):
    """Calculate mean PCF across all users and their top-k items."""
    total_pcf, count = 0, 0
    for user_items in reranked_items_list:
        for item_id in user_items:
            token = dataset.id2token(dataset.iid_field, int(item_id))
            asin = id_to_asin.get(int(token))
            total_pcf += co2e_scores.get(asin)
            count += 1
    return total_pcf / count if count > 0 else 0


def get_pcf_reduction_perc(grp):
    """Calculates the PCF reduction in terms of %."""
    base = grp[grp['Alpha'] == 1.0]['PCF_Avg'].values
    grp['Reduction_%'] = round(((base[0] - grp['PCF_Avg']) / base[0]) * 100, 2) if base.size > 0 else 0
    return grp


def evaluate_model_results(file_path, config, dataset, id_to_asin, co2e_scores):
    """Load re-ranked results and compute RecBole and sustainability metrics."""
    data = torch.load(file_path, weights_only=False)
    k = config['topk'][0]

    # RecBole metrics
    pos_matrix = torch.tensor(data['pos_matrix'], device=config['device'])[:, :k]
    pos_len = torch.tensor(data['pos_len'], device=config['device']).view(-1, 1)
    struct = {'rec.topk': torch.cat((pos_matrix, pos_len), dim=1).cpu()}
    rec_results = Evaluator(config).evaluate(struct)

    # Sustainability metrics
    avg_pcf = calculate_average_pcf(data['reranked_items'], id_to_asin, co2e_scores, dataset)

    # Merge results
    res = {'Model': data['model'], 'Alpha': data['alpha'], 'PCF_Avg': round(avg_pcf, 4)}
    res.update({m: round(v, 4) for m, v in rec_results.items()})

    return res


# =============================================
# Setup
# =============================================
# Load configuration and dataset
# (using any trained model)
config, _, dataset, *_ = load_data_and_model(
    model_file=get_latest_checkpoint("LightGCN_best"),
)

# Load C02 score estimations
co2e_scores = {}
with open("../1_pcf/results/final_metadata.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        co2e_scores[data["parent_asin"]] = data["co2e_kg"]

# Load item_index -> parent_asin mapping
item_map_df = pd.read_csv("../2_recbole/process_data/maps/item_map.tsv", sep='\t')
id_to_asin = dict(zip(item_map_df['item_index'], item_map_df['parent_asin']))

# Set result files to analyze
files = [
    '../3_reranking/results/reranked_results_BPR_alpha_1.0.pth',
    '../3_reranking/results/reranked_results_BPR_alpha_0.5.pth',
    '../3_reranking/results/reranked_results_LightGCN_alpha_1.0.pth',
    '../3_reranking/results/reranked_results_LightGCN_alpha_0.5.pth'
]

# =============================================
# Evaluation
# =============================================
# Evaluate model recommendations using
# all the saved recommendations of both models
results = []
for f in files:
    results.append(evaluate_model_results(f, config, dataset, id_to_asin, co2e_scores))

# Formatting result table
df = pd.DataFrame(results)
df = df.groupby('Model', group_keys=False).apply(get_pcf_reduction_perc)
cols = (
        ['Model', 'Alpha', 'PCF_Avg', 'Reduction_%'] +
        [c for c in df.columns if c not in ['Model', 'Alpha', 'PCF_Avg', 'Reduction_%']]
)

# Display results
print("\n" + "=" * 90 + "\n Results \n" + "=" * 90)
print(df[cols].to_string(index=False))
print("=" * 90)

# Save results
df.to_csv('results/evaluation_results.csv', index=False)