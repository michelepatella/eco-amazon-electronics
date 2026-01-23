import torch
import numpy as np
import math
from collections import defaultdict
from tqdm import tqdm

from recbole.utils.case_study import full_sort_topk
from recbole.quick_start import load_data_and_model
from recbole.evaluator import Evaluator


# dummy function to rerank the reclist
def pcf_aware_reranker(user_id_external, item_list_external, score_list):
    pairs = list(zip(item_list_external, score_list))
    pairs.reverse()
    reranked_items, reranked_scores = zip(*pairs)
    return list(reranked_items), list(reranked_scores)


def get_top_k_recommendations(model, k=100):
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


def get_reranked_top_k_recommendations(final_scores, final_iids, k=100):
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
    test_user_internal_ids = np.unique(test_data.dataset.inter_feat[dataset.uid_field].numpy())
    for idx, internal_uid in enumerate(
            tqdm(test_user_internal_ids, desc=f"Re-ranking top-{k} recommendations...")
    ):
        # Get both internal items and scores
        internal_items = final_iids[idx]
        item_scores = final_scores[idx].tolist()

        # Get both external user and item IDs
        external_uid = dataset.id2token(dataset.uid_field, internal_uid)
        external_items = dataset.id2token(dataset.iid_field, internal_items)

        # Re-rank the recommendations taking care about PCF
        reranked_items, reranked_scores = pcf_aware_reranker(
            external_uid, external_items, item_scores
        )

        # Convert external items to internal ones
        reranked_internal = dataset.token2id(dataset.iid_field, reranked_items)

        # Retrieve ground truth
        user_gt = ground_truth_map[internal_uid]

        # For each item, check whether it appears in the ground truth:
        # 0 -> It doesn't appear
        # 1 -> It appears
        hits = [1 if item in user_gt else 0 for item in reranked_internal]
        hits = hits[:k]

        # Update matrices for evaluation
        pos_matrix_list.append(hits)
        pos_len_list.append(len(user_gt))

    return pos_matrix_list, pos_len_list


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

# =============================================
# Standard recommendations
# =============================================
# Retrieve the standard recommendations using both trained models
final_scores_bpr, final_iids_bpr = get_top_k_recommendations(bpr_model)
final_scores_light_gcn, final_iids_light_gcn = get_top_k_recommendations(light_gcn_model)

# =============================================
# PCF-aware recommendations
# =============================================
# Re-rank standard recommendations taking care about PCF
pos_matrix_list_bpr, pos_len_list_bpr = (
    get_reranked_top_k_recommendations(final_scores_bpr, final_iids_bpr)
)
pos_matrix_list_light_gcn, pos_len_list_light_gcn = (
    get_reranked_top_k_recommendations(final_scores_light_gcn, final_iids_light_gcn)
)

"""
# step 4. evaluate the reranked recommendation list
print("\n>>> [Phase 4] Running RecBole Evaluator...")

max_k = max(config['topk'])
pos_matrix = torch.tensor(pos_matrix_list, device=config['device'], dtype=torch.int)[:, :max_k]
pos_len_tensor = torch.tensor(pos_len_list, device=config['device'], dtype=torch.int).view(-1, 1)
combined_matrix = torch.cat((pos_matrix, pos_len_tensor), dim=1)
struct = {
    'rec.topk': combined_matrix.cpu()
}
evaluator = Evaluator(config)
final_results = evaluator.evaluate(struct)

# print results
print("\n" + "="*40)
print(" FINAL RESULTS (Via RecBole Evaluator)")
print("="*40)
for metric, value in final_results.items():
    print(f" {metric}: {value:.4f}")
print("="*40)
"""