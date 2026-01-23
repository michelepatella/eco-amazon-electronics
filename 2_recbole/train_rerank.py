import torch
import numpy as np
import math
from collections import defaultdict
from tqdm import tqdm

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR
from recbole.trainer import Trainer
from recbole.utils.case_study import full_sort_topk
from recbole.quick_start import load_data_and_model
from recbole.evaluator import Evaluator, Collector

# dummy function to rerank the reclist
def pcf_aware_reranker(user_id_external, item_list_external, score_list):
    pairs = list(zip(item_list_external, score_list))
    pairs.reverse()
    reranked_items, reranked_scores = zip(*pairs)
    return list(reranked_items), list(reranked_scores)


if __name__ == "__main__":
   

    # param list
    epochs = 1

    # step 1: train the vanilla model
    print(f">>> [Phase 1] Configuring and Training BPR...")

    config_dict = {
        'model': 'BPR',
        'dataset': 'ml-100k',
        'epochs': epochs,
        'eval_step': 1,
        'topk': [5, 10, 20, 50, 100],
        'metrics': ['Recall', 'NDCG', 'MRR'],
        'train_batch_size': 2048,
        'eval_batch_size': 4096,
        'save_dataset': True,
        'save_dataloaders': True,
        'show_progress': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'valid_metric': 'recall@20'
    }

    # configure the model and the trainer
    config = Config(model='BPR', dataset='ml-100k', config_dict=config_dict)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = BPR(config, dataset).to(config['device'])
    trainer = Trainer(config, model)

    # train the model and save the .pth checkpoint file name
    trainer.fit(train_data, valid_data, show_progress=True, saved=True)
    saved_model_path = trainer.saved_model_file
    print(f">>> Training finished. Model saved at: {saved_model_path}")

    # step 2: load the model and get top-100
    print(f"\n>>> [Phase 2] Loading Model and Retrieving Top-100...")

    # load the model
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=saved_model_path
    )
    
    # get user IDs internal to recbole
    test_user_internal_ids = np.unique(test_data.dataset.inter_feat[dataset.uid_field].numpy())
    batch_size = 1000 
    k = 100
    total_users = len(test_user_internal_ids)
    
    all_scores = []
    all_iids = []

    print(f"Retrieving recommendations for {total_users} users...")
    
    # get top-100 per user + recommendation scores
    num_batches = math.ceil(total_users / batch_size)
    for i in tqdm(range(num_batches), desc="Batch Inference"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_users)
        
        batch_users = test_user_internal_ids[start_idx:end_idx]
        uid_tensor = torch.tensor(batch_users)

        batch_scores, batch_iids = full_sort_topk(
            uid_tensor, model, test_data, k=k, device=config['device']
        )
        
        all_scores.append(batch_scores.cpu())
        all_iids.append(batch_iids.cpu())

    final_scores = torch.cat(all_scores, dim=0)
    final_iids = torch.cat(all_iids, dim=0)
    print(">>> Retrieval Complete.")

    # step 3: perform the reranking with 
    print("\n>>> [Phase 3] Reranking and Preparing RecBole Evaluation...")

    # get user and item IDs external to RecBole, which match with the original dataset info
    ground_truth_map = defaultdict(set)
    uids = test_data.dataset.inter_feat[dataset.uid_field].numpy()
    iids = test_data.dataset.inter_feat[dataset.iid_field].numpy()
    for u, i in zip(uids, iids):
        ground_truth_map[u].add(i)

    pos_matrix_list = []
    pos_len_list = []
    
    # perform the reranking with the PCF aware item data
    for idx, internal_uid in enumerate(tqdm(test_user_internal_ids, desc="Reranking")):
        
        internal_items = final_iids[idx]
        item_scores = final_scores[idx].tolist()
        
        external_uid = dataset.id2token(dataset.uid_field, internal_uid)
        external_items = dataset.id2token(dataset.iid_field, internal_items)
        
        reranked_items, reranked_scores = pcf_aware_reranker(
            external_uid, external_items, item_scores
        )
        
        reranked_internal = dataset.token2id(dataset.iid_field, reranked_items)
        user_gt = ground_truth_map[internal_uid]
        hits = [1 if item in user_gt else 0 for item in reranked_internal]
        hits = hits[:k]
        pos_matrix_list.append(hits)
        pos_len_list.append(len(user_gt))

    # step 4. evaluate the reranked recommendation list
    print("\n>>> [Phase 4] Running RecBole Evaluator...")

    pos_matrix = torch.tensor(pos_matrix_list, device=config['device'], dtype=torch.int)    
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