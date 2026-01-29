import torch
import pandas as pd
import json
import numpy as np
from recbole.evaluator import Evaluator
from recbole.quick_start import load_data_and_model

from utils import get_latest_checkpoint


def calculate_average_pcf(reranked_items_list, id_to_asin, scores_dict, dataset):
    """Calculate mean PCF across all users and their top-k items, ignoring missing values."""
    total_pcf = 0
    count = 0
    for user_items in reranked_items_list:
        for item_id in user_items:
            token = dataset.id2token(dataset.iid_field, int(item_id))
            asin = id_to_asin.get(int(token))

            # Retrieve the score: if it doesn't exist, we skip the item
            pcf = scores_dict.get(asin)
            if pcf is not None:
                total_pcf += pcf
                count += 1

    return total_pcf / count if count > 0 else 0


def evaluate_model_results(
    file_path,
    config,
    dataset,
    id_to_asin,
    co2e_scores,
    gt_pcf,
    item_cnt,
    user_indices=None,
):
    """Load re-ranked results and compute RecBole, estimated and real sustainability
    metrics for relevant users."""
    data = torch.load(file_path, weights_only=False)
    k = config["topk"][0]
    reranked_items_list = data["reranked_items"]

    # Calculate user indices: we only consider users who have
    # at least one item with a ground truth PCF in their top-k
    if user_indices is None:
        user_indices = []
        for i, user_items in enumerate(reranked_items_list):
            for item_id in user_items[:k]:
                token = dataset.id2token(dataset.iid_field, int(item_id))
                asin = id_to_asin.get(int(token))

                # The user is valid only if we can verify the
                # real impact of the re-ranking
                if asin in gt_pcf:
                    user_indices.append(i)
                    break

    if not user_indices:
        return None, []

    # Retrieve reranked items (filtered for the provided users)
    reranked_np = np.array(reranked_items_list)[user_indices]
    reranked_items = torch.tensor(reranked_np, device=config["device"])[:, :k]

    # RecBole metrics (filtered)
    pos_matrix = torch.tensor(
        np.array(data["pos_matrix"])[user_indices], device=config["device"]
    )[:, :k]
    pos_len = torch.tensor(
        np.array(data["pos_len"])[user_indices], device=config["device"]
    ).view(-1, 1)

    # Run RecBole evaluation
    struct = {
        "rec.topk": torch.cat((pos_matrix, pos_len), dim=1).cpu(),
        "rec.items": reranked_items.cpu(),
        "data.num_items": dataset.item_num,
        "data.count_items": item_cnt,
    }
    rec_results = Evaluator(config).evaluate(struct)

    # Sustainability metrics on the same users
    # We calculate the average only on items that have a score
    top_k_items = [reranked_items_list[i][:k] for i in user_indices]
    avg_pcf_est = calculate_average_pcf(top_k_items, id_to_asin, co2e_scores, dataset)
    avg_pcf_real = calculate_average_pcf(top_k_items, id_to_asin, gt_pcf, dataset)

    res = {
        "MODEL": data["model"],
        "ALPHA": data["alpha"],
        "PCF_EST@10": round(avg_pcf_est, 4),
        "PCF_REAL@10": round(avg_pcf_real, 4),
        "#USERS": len(user_indices),
    }
    res.update({m: round(v, 4) for m, v in rec_results.items()})

    return res, user_indices


# =============================================
# Setup
# =============================================
# Fix model
# model_tag = "gemini-2_5-flash"
model_tag = "gpt-o3-mini"

# Load configuration and dataset
config, _, dataset, *_ = load_data_and_model(
    model_file=get_latest_checkpoint("LightGCN_best"),
)

# Upload configuration to define evaluation metrics
config["metrics"] = [
    "Recall",
    "NDCG",
    "GiniIndex",
    "AveragePopularity",
    "TailPercentage",
    "ItemCoverage",
    "Precision",
    "MRR",
    "Hit",
]

# Load ground truth CO2 values
gt_pcf = {}
gt_pcf_file = "../1_pcf/results/subset/ground_truth.jsonl"
with open(gt_pcf_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            item_data = json.loads(line)
            gt_pcf[item_data["parent_asin"]] = item_data["co2e_kg"]

# Load C02 score estimations (LLM)
co2e_scores = {}
llm_file = f"../1_pcf/results/subset/results/{model_tag}_results.json"
with open(llm_file, "r", encoding="utf-8") as f:
    data_list = json.load(f)
    for data in data_list:
        if data.get("co2e_kg") is not None:
            co2e_scores[data["parent_asin"]] = data["co2e_kg"]

# Load item_index -> parent_asin mapping
item_map_df = pd.read_csv("../2_recbole/process_data/maps/item_map.tsv", sep="\t")
id_to_asin = dict(zip(item_map_df["item_index"], item_map_df["parent_asin"]))

# For calculating item popularity
train_item_ids = dataset.inter_feat[dataset.iid_field].numpy()
item_cnt_array = np.zeros(dataset.item_num, dtype=np.int64)
for iid in train_item_ids:
    item_cnt_array[iid] += 1
item_cnt = [(i, count) for i, count in enumerate(item_cnt_array)]

# Set result files to analyze
files = [
    f"../3_reranking/results/subset/BPR/{model_tag}/results_alpha_0.25.pth",
    f"../3_reranking/results/subset/BPR/{model_tag}/results_alpha_0.5.pth",
    f"../3_reranking/results/subset/BPR/{model_tag}/results_alpha_0.75.pth",
    f"../3_reranking/results/subset/BPR/{model_tag}/results_alpha_1.0.pth",
    f"../3_reranking/results/subset/LightGCN/{model_tag}/results_alpha_0.25.pth",
    f"../3_reranking/results/subset/LightGCN/{model_tag}/results_alpha_0.5.pth",
    f"../3_reranking/results/subset/LightGCN/{model_tag}/results_alpha_0.75.pth",
    f"../3_reranking/results/subset/LightGCN/{model_tag}/results_alpha_1.0.pth",
]

# =============================================
# PCF estimation by LLM — Evaluation
# =============================================
# Calculate LLM's PCF estimation errors
common_asins = [a for a in gt_pcf if a in co2e_scores]
y_true = [gt_pcf[a] for a in common_asins]
y_pred = [co2e_scores[a] for a in common_asins]
mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
rmse = np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

# Display results
print("\n" + "=" * 50)
print(f" PCF ESTIMATION BY LLM — RESULTS ({model_tag})")
print("=" * 50)
print(f" MAE: {mae:.4f} kg")
print(f" RMSE: {rmse:.4f} kg")
print("=" * 50)

# Save errors
error_metrics = {
    "model": model_tag,
    "mae": round(float(mae), 4),
    "rmse": round(float(rmse), 4),
    "n_samples": len(common_asins),
}
with open(f"results/subset/{model_tag}/pcf_error.json", "w") as f:
    json.dump(error_metrics, f, indent=4)

# =============================================
# RecSys — Evaluation
# =============================================
# Evaluate model recommendations
results = []
models_to_test = ["BPR", "LightGCN"]
for model_name in models_to_test:
    model_files = [f for f in files if model_name in f]

    # Valid users are those where at least one estimated item
    # appears in the top-k (using alpha=0.25 increases the probability
    # of finding what we are looking for)
    alpha_min_file = [f for f in model_files if "0.25" in f][0]
    _, valid_indices = evaluate_model_results(
        alpha_min_file,
        config,
        dataset,
        id_to_asin,
        co2e_scores,
        gt_pcf,
        item_cnt,
    )

    # Evaluate all the model files using fixed indices
    for f in model_files:
        res, _ = evaluate_model_results(
            f,
            config,
            dataset,
            id_to_asin,
            co2e_scores,
            gt_pcf,
            item_cnt,
            user_indices=valid_indices,
        )
        if res:
            results.append(res)

# Rename columns
k = config["topk"][0]
mapping = {
    f"recall@{k}": "RECALL@10",
    f"mrr@{k}": "MRR@10",
    f"ndcg@{k}": "NDCG@10",
    f"hit@{k}": "HIT@10",
    f"precision@{k}": "PRECISION@10",
    f"averagepopularity@{k}": "AVG_POP",
    f"tailpercentage@{k}": "TAIL_PERC",
    f"giniindex@{k}": "GINI_INDEX",
    f"itemcoverage@{k}": "ITEM_COV",
}

# Create results table
df = pd.DataFrame(results)
df.rename(columns=mapping, inplace=True)

# Define all its columns
main_cols = ["MODEL", "ALPHA", "AVG_POP", "TAIL_PERC", "GINI_INDEX", "ITEM_COV"]
pcf_cols = ["PCF_EST@10", "PCF_REAL@10"]
acc_cols = ["RECALL@10", "MRR@10", "NDCG@10", "HIT@10", "PRECISION@10"]
cols = main_cols + pcf_cols + acc_cols
existing_cols = [c for c in cols if c in df.columns]
df_final = df[existing_cols].sort_values(["MODEL", "ALPHA"], ascending=[True, False])

# Display results
print("\n" + "=" * 135 + "\n RECSYS — RESULTS \n" + "=" * 135)
print(df_final.to_string(index=False))
print("=" * 135)

# Save results
df_final.to_csv(f"results/subset/{model_tag}/evaluation_results.csv", index=False)
