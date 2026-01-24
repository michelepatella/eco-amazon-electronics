import json

import pandas as pd
import torch
import numpy as np
import math
from collections import defaultdict

from recbole.evaluator import Evaluator
from tqdm import tqdm

from recbole.utils.case_study import full_sort_topk
from recbole.quick_start import load_data_and_model


def pcf_aware_reranker(
        co2e_scores,
        item_list_internal,
        score_list,
        dataset,
        id_to_asin,
        alpha
):
    """Calculates PCF-aware to re-rank recommendations."""
    # Retrieve tokens
    external_items = []
    for i in item_list_internal:
        raw_token = dataset.id2token(dataset.iid_field, int(i))
        item_idx = int(raw_token)
        asin = id_to_asin.get(item_idx)
        external_items.append(asin)

    # Retrieve PCF values
    pcf_values = [co2e_scores.get(asin) for asin in external_items]
    pcf_array = np.array(pcf_values)

    # Normalize PCF values
    max_pcf, min_pcf = pcf_array.max(), pcf_array.min()
    pcf_norm = (max_pcf - pcf_array) / (max_pcf - min_pcf) \
        if max_pcf > min_pcf \
        else np.zeros_like(pcf_array)

    # Normalize predictions
    preds = np.array(score_list)
    preds_norm = (preds - preds.min()) / (preds.max() - preds.min()) \
        if preds.max() > preds.min() \
        else np.zeros_like(preds)

    # Calculate SaS
    sas_scores = alpha * preds_norm + (1 - alpha) * pcf_norm

    # Sort items by SaS descending
    sorted_indices = np.argsort(sas_scores)[::-1]
    items_np = item_list_internal.cpu().numpy() \
        if torch.is_tensor(item_list_internal) \
        else np.array(item_list_internal)

    return items_np[sorted_indices].copy(), sas_scores[sorted_indices].copy()


def get_top_k_recommendations(model, k):
    """Retrieves top-k recommendations for all the users, given a trained model."""
    # Setup
    batch_size = 1000
    all_scores = []
    all_iids = []

    # Get user IDs internal to RecBole
    test_user_internal_ids = np.unique(test_data.dataset.inter_feat[dataset.uid_field].numpy())
    total_users = len(test_user_internal_ids)
    num_batches = math.ceil(total_users / batch_size)

    # Get top-k recommendations for all the users
    for i in tqdm(
            range(num_batches),
            desc=f"Retrieving top-{k} recommendations for {total_users} users..."
    ):
        # Calculate the current batch boundaries
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_users)

        # Retrieve user IDs
        batch_users = test_user_internal_ids[start_idx:end_idx]
        uid_tensor = torch.tensor(batch_users)

        # Get recommendations for all the items
        # and users in the current batch,
        # returning top-k predictions
        batch_scores, batch_iids = full_sort_topk(
            uid_tensor, model, test_data, k=k, device=config['device']
        )
        all_scores.append(batch_scores.cpu())
        all_iids.append(batch_iids.cpu())

    final_scores = torch.cat(all_scores, dim=0)
    final_iids = torch.cat(all_iids, dim=0)

    return final_scores, final_iids


def get_reranked_top_k_recommendations(
        final_scores,
        final_iids,
        id_to_asin,
        co2e_scores,
        alpha,
        k,
):
    # Get user and item IDs external to RecBole,
    # which match with the original dataset info
    ground_truth_map = defaultdict(set)
    uids = test_data.dataset.inter_feat[dataset.uid_field].numpy()
    iids = test_data.dataset.inter_feat[dataset.iid_field].numpy()
    for u, i in zip(uids, iids):
        ground_truth_map[u].add(i)

    # Perform the re-ranking with the PCF-aware item data
    pos_matrix_list = []
    pos_len_list = []
    reranked_items_all_users = []
    test_user_internal_ids = np.unique(test_data.dataset.inter_feat[dataset.uid_field].numpy())
    for idx, internal_uid in enumerate(
            tqdm(test_user_internal_ids, desc=f"Re-ranking top-{k} recommendations...")
    ):
        # Get both internal items and scores
        internal_items = final_iids[idx]
        item_scores = final_scores[idx].tolist()

        # Re-rank the recommendations taking care about PCF
        reranked_items, reranked_scores = pcf_aware_reranker(
            co2e_scores=co2e_scores,
            item_list_internal=internal_items,
            score_list=item_scores,
            dataset=dataset,
            id_to_asin=id_to_asin,
            alpha=alpha
        )

        # Retrieve ground truth
        user_gt = ground_truth_map[internal_uid]

        # Save re-ranked items
        reranked_items_all_users.append(reranked_items[:k])

        # For each item, check whether it appears in the ground truth:
        # 0 -> It doesn't appear
        # 1 -> It appears
        reranked_items = [int(i) for i in reranked_items]
        hits = [1 if int(item) in user_gt else 0 for item in reranked_items]
        hits = hits[:k]

        # Update matrices for evaluation
        pos_matrix_list.append(hits)
        pos_len_list.append(len(user_gt))

    return pos_matrix_list, pos_len_list, reranked_items_all_users


# =============================================
# Setup
# =============================================
# Load the best BPR and LightGCN trained models
config, bpr_model, dataset, *_, test_data = load_data_and_model(
    model_file='../2_recbole/saved/BPR_best/BPR-Jan-22-2026_21-11-15.pth',
)
_, light_gcn_model, *_ = load_data_and_model(
    model_file='../2_recbole/saved/LightGCN_best/LightGCN-Jan-22-2026_21-18-39.pth'
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

# =============================================
# Standard recommendations
# =============================================
# Retrieve the standard recommendations using both trained models
final_scores_bpr, final_iids_bpr = get_top_k_recommendations(
    model=bpr_model,
    k=100
)
final_scores_light_gcn, final_iids_light_gcn = get_top_k_recommendations(
    model=light_gcn_model,
    k=100
)

# =============================================
# PCF-aware recommendations
# =============================================
# Re-rank standard recommendations taking care about PCF
pos_matrix_list_bpr, pos_len_list_bpr, _ = (
    get_reranked_top_k_recommendations(
        final_scores=final_scores_bpr,
        final_iids=final_iids_bpr,
        id_to_asin=id_to_asin,
        co2e_scores=co2e_scores,
        alpha=0.5,
        k=10
    )
)
pos_matrix_list_light_gcn, pos_len_list_light_gcn, _ = (
    get_reranked_top_k_recommendations(
        final_scores=final_scores_light_gcn,
        final_iids=final_iids_light_gcn,
        id_to_asin=id_to_asin,
        co2e_scores=co2e_scores,
        alpha=0.5,
        k=10
    )
)