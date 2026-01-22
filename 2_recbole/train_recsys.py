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
        raise FileNotFoundError(f"No checkpoint found at {saved_model_dir}")
    latest_file = sorted(checkpoint_files)[-1]
    return latest_file


def get_topk_from_scores(score_matrix, dataset, k, filename, batch_size=1024):
    """Vectorized top-k from full sort scores and save results."""
    all_item_tokens = dataset.id2token(dataset.iid_field, range(dataset.item_num))
    all_user_tokens = dataset.id2token(dataset.uid_field, range(dataset.user_num))

    with open(filename, 'w') as f:
        f.write("users\titems\tscores\n")

    for start in tqdm(range(0, score_matrix.shape[0], batch_size), desc="Saving to disk"):
        end = min(start + batch_size, score_matrix.shape[0])
        batch_scores = score_matrix[start:end]
        topk_scores, topk_items = torch.topk(batch_scores, k, dim=1)

        batch_user_tokens = [all_user_tokens[u] for u in range(start, end)]

        chunk_rows = []
        for i, u_token in enumerate(batch_user_tokens):
            u_items = [all_item_tokens[j] for j in topk_items[i].tolist()]
            u_scores = topk_scores[i].tolist()

            for item, score in zip(u_items, u_scores):
                chunk_rows.append(f"{u_token}\t{item}\t{score}")

        with open(filename, 'a') as f:
            f.write("\n".join(chunk_rows) + "\n")


def get_topk(model_name_file):
    """Save top-5, top-10, and top-10 full predictions."""
    # load model e dataset
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


# dataset
amazon_elec = 'amazon_elec'

# Create folders for predictions
os.makedirs('preds/full', exist_ok=True)
os.makedirs('preds/top1', exist_ok=True)
os.makedirs('preds/top2', exist_ok=True)
os.makedirs('preds/top3', exist_ok=True)
os.makedirs('preds/top5', exist_ok=True)
os.makedirs('preds/top10', exist_ok=True)

# BPR model
run_recbole(
    model='BPR',
    dataset=amazon_elec,
    config_dict={
        'tensorboard': False,
        'epochs': 200,
        'eval_step': 10,
        'benchmark_filename': ['train', 'valid', 'test'],
        'checkpoint_dir': './saved/BPR',
    }
)
model_file = get_latest_checkpoint("BPR")
get_topk(model_file)
with open('logs.txt', 'a') as f:
    f.write(f"{model_file}\tk=64\n")

# LightGCN model (layer=1)
run_recbole(
    model='LightGCN',
    dataset=amazon_elec,
    config_dict={
        'tensorboard': False,
        'epochs': 200,
        'eval_step': 10,
        'n_layers': 1,
        'benchmark_filename': ['train', 'valid', 'test'],
        'checkpoint_dir': './saved/LightGCN_layer1',
    }
)
model_file = get_latest_checkpoint("LightGCN_layer1")
get_topk(model_file)
with open('logs.txt', 'a') as f:
    f.write(f"{model_file}\tn_layers=1\n")

# LightGCN model (layer=2)
run_recbole(
    model='LightGCN',
    dataset=amazon_elec,
    config_dict={
        'tensorboard': False,
        'epochs': 200,
        'eval_step': 10,
        'n_layers': 2,
        'benchmark_filename': ['train', 'valid', 'test'],
        'checkpoint_dir': './saved/LightGCN_layer2',
    }
)
model_file = get_latest_checkpoint("LightGCN_layer2")
get_topk(model_file)
with open('logs.txt', 'a') as f:
    f.write(f"{model_file}\tn_layers=2\n")

# LightGCN model (layer=3)
run_recbole(
    model='LightGCN',
    dataset=amazon_elec,
    config_dict={
        'tensorboard': False,
        'epochs': 200,
        'eval_step': 10,
        'n_layers': 3,
        'benchmark_filename': ['train', 'valid', 'test'],
        'checkpoint_dir': './saved/LightGCN_layer3'
    }
)
model_file = get_latest_checkpoint("LightGCN_layer3")
get_topk(model_file)
with open('logs.txt', 'a') as f:
    f.write(f"{model_file}\tn_layers=3\n")