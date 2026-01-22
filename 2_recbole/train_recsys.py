from ray import tune
from recbole.quick_start import run_recbole, load_data_and_model
import os
import torch
import glob
from tqdm import tqdm
import pandas as pd


def get_latest_checkpoint(model_name):
    """Get the latest model checkpoint from its folder."""
    saved_model_dir = f"./saved/{model_name}"
    checkpoint_files = glob.glob(os.path.join(saved_model_dir, "*.pth"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No mdoel checkpoint found at '{saved_model_dir}'.")
    latest_file = sorted(checkpoint_files)[-1]
    return latest_file


def get_topk_from_scores(score_matrix, dataset, k, filename, batch_size=1024):
    """Compute top-k items per user and save to a TSV file in batches."""

    all_item_tokens = dataset.id2token(dataset.iid_field, range(dataset.item_num))
    all_user_tokens = dataset.id2token(dataset.uid_field, range(dataset.user_num))

    # Write header
    with open(filename, 'w') as f:
        f.write("users\titems\tscores\n")

    # Process users in batches to save memory
    for start in tqdm(range(0, score_matrix.shape[0], batch_size), desc="Saving to disk..."):
        end = min(start + batch_size, score_matrix.shape[0])
        batch_scores = score_matrix[start:end]

        # Top-k items per user
        topk_scores, topk_items = torch.topk(batch_scores, k, dim=1)
        batch_user_tokens = [all_user_tokens[u] for u in range(start, end)]

        # Flatten batch to lines
        chunk_rows = [
            f"{u_token}\t{all_item_tokens[i]}\t{s}"
            for u_token, items, scores in zip(batch_user_tokens, topk_items.tolist(), topk_scores.tolist())
            for i, s in zip(items, scores)
        ]

        # Append batch to file
        with open(filename, 'a') as f:
            f.write("\n".join(chunk_rows) + "\n")


def get_topk(model_name_file):
    """Save top-k and full predictions."""
    # Load model e dataset
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_name_file)

    # Full sort scores
    model.eval()
    with torch.no_grad():
        user_e = model.user_embedding(torch.arange(dataset.user_num).to(config['device']))
        item_e = model.item_embedding(torch.arange(dataset.item_num).to(config['device']))
        score_matrix = torch.matmul(user_e, item_e.transpose(0, 1)).cpu()

    # Save full predictions
    full_path = f'preds/full/{model_name_file.split("/")[1]}.tsv'
    get_topk_from_scores(score_matrix, dataset, score_matrix.shape[1], full_path)

    # Save top-k predictions
    reader = pd.read_csv(full_path, sep='\t', chunksize=100000)
    top_1_list = []
    top_2_list = []
    top_3_list = []
    top_5_list = []
    top_10_list = []
    for chunk in reader:
        top_1_list.append(chunk.groupby('users').head(1))
        top_2_list.append(chunk.groupby('users').head(2))
        top_3_list.append(chunk.groupby('users').head(3))
        top_5_list.append(chunk.groupby('users').head(5))
        top_10_list.append(chunk.groupby('users').head(10))
    pd.concat(top_1_list).to_csv(f'preds/top1/{model_name_file.split("/")[1]}.tsv', sep='\t', header=True, index=False)
    pd.concat(top_2_list).to_csv(f'preds/top2/{model_name_file.split("/")[1]}.tsv', sep='\t', header=True, index=False)
    pd.concat(top_3_list).to_csv(f'preds/top3/{model_name_file.split("/")[1]}.tsv', sep='\t', header=True, index=False)
    pd.concat(top_5_list).to_csv(f'preds/top5/{model_name_file.split("/")[1]}.tsv', sep='\t', header=True, index=False)
    pd.concat(top_10_list).to_csv(f'preds/top10/{model_name_file.split("/")[1]}.tsv', sep='\t', header=True, index=False)


def train_recbole_hpo(config, model_name, dataset_name, exp_name):
    """Trainable Ray Tune function."""
    result = run_recbole(
        model=model_name,
        dataset=dataset_name,
        config_dict={
            'tensorboard': False,
            'device': device,
            'epochs': config['epochs'],
            'eval_step': 10,
            'learning_rate': config['lr'],
            'embedding_size': config['embedding_size'],
            'reg_weight': config['reg_weight'],
            **({'n_layers': config['n_layers']} if model_name == 'LightGCN' else {}),
            'benchmark_filename': ['train', 'valid', 'test'],
            'checkpoint_dir': f'./saved/{exp_name}',
        }
    )

    # RecBole returns best valid score dict
    valid_recall = result['best_valid_score']['recall@10']
    tune.report(recall_10=valid_recall)


def run_hpo_bpr():
    """Run HPO for BPR."""
    # Define search space for BPR
    search_space = {
        'lr': tune.loguniform(1e-4, 5e-3),
        'embedding_size': tune.choice([32, 64]),
        'reg_weight': tune.loguniform(1e-5, 1e-4),
        'epochs': 100
    }

    # Run HPO for BPR
    analysis = tune.run(
        tune.with_parameters(
            train_recbole_hpo,
            model_name='BPR',
            dataset_name='amazon_elec',
            exp_name='BPR_HPO'
        ),
        metric='recall_10',
        mode='max',
        config=search_space,
        num_samples=10,
        resources_per_trial={'cpu': 4, 'gpu': 0}
    )

    # Get best config for BPR
    best_config = analysis.get_best_config(metric='recall_10', mode='max')
    return best_config


def run_hpo_lightgcn():
    """Run HPO for BPR."""
    # Define search space for Light GCN
    search_space = {
        'lr': tune.loguniform(1e-4, 5e-3),
        'embedding_size': tune.choice([32, 64]),
        'reg_weight': tune.loguniform(1e-5, 1e-4),
        'n_layers': tune.choice([1, 2, 3]),
        'epochs': 100
    }

    # Run HPO for Light GCN
    analysis = tune.run(
        tune.with_parameters(
            train_recbole_hpo,
            model_name='LightGCN',
            dataset_name='amazon_elec',
            exp_name='LightGCN_HPO'
        ),
        metric='recall_10',
        mode='max',
        config=search_space,
        num_samples=15,
        resources_per_trial={'cpu': 4, 'gpu': 0}
    )

    # Get best config for Light GCN
    best_config = analysis.get_best_config(metric='recall_10', mode='max')
    return best_config


# =============================================
# Setup
# =============================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# =============================================
# HPO both for BPR and Light GCN
# =============================================
best_bpr = run_hpo_bpr()
best_lgcn = run_hpo_lightgcn()

# =============================================
# Final training both for BPR and Light GCN
# =============================================
run_recbole(
    model='BPR',
    dataset='amazon_elec',
    config_dict={
        **best_bpr,
        'device': device,
        'benchmark_filename': ['train', 'valid', 'test'],
        'checkpoint_dir': './saved/BPR_best'
    }
)
model_file = get_latest_checkpoint("BPR_best")
get_topk(model_file)

run_recbole(
    model='LightGCN',
    dataset='amazon_elec',
    config_dict={
        **best_lgcn,
        'device': device,
        'benchmark_filename': ['train', 'valid', 'test'],
        'checkpoint_dir': './saved/LightGCN_best'
    }
)
model_file = get_latest_checkpoint("LightGCN_best")
get_topk(model_file)