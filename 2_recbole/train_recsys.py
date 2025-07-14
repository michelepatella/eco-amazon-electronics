from recbole.quick_start import run_recbole, load_data_and_model
from recbole.data.interaction import Interaction
import pandas as pd
import os

def top_k_scores(predictions, k):

  top_k = pd.DataFrame()
  for u in list(set(predictions['users'])):
    p = predictions.loc[predictions['users'] == u ].head(k)
    top_k = pd.concat([top_k, p])
  return top_k

def get_topk(model_name_file):

    # load model and data
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_name_file)

    users = test_data.dataset.inter_feat['user_id']
    items = test_data.dataset.inter_feat['item_id']
    scores = model.predict(Interaction({'user_id': users,
                                        'item_id': items}).to(device=model.device))

    # save top5 and top10 predictions
    predictions = pd.DataFrame()
    predictions['users'] = list(map(int, dataset.id2token(dataset.uid_field, users)))
    predictions['items'] = list(map(int, dataset.id2token(dataset.iid_field, items)))
    predictions['scores'] = scores.tolist()
    predictions = predictions[predictions['scores'] != 0 ]
    predictions = predictions.sort_values(by=['users', 'scores'],ascending=[True, False])

    predictions.to_csv(f'preds/full/{model_name_file.split("/")[1]}.tsv', sep='\t', header=True, index=False)
    top_5 = top_k_scores(predictions, 5)
    top_5.to_csv(f'preds/top5/{model_name_file.split("/")[1]}.tsv', sep='\t', header=True, index=False)
    top_10 = top_k_scores(predictions, 10)
    top_10.to_csv(f'preds/top10/{model_name_file.split("/")[1]}.tsv', sep='\t', header=True, index=False)


# dataset
amazon_elec = 'amazon_elec'

# create folders for predictions
if not os.path.exists('preds'):
    os.makedirs('preds')
if not os.path.exists('preds/top5'):
    os.makedirs('preds/top5')
if not os.path.exists('preds/top10'):
    os.makedirs('preds/top10')
if not os.path.exists('preds/full'):
    os.makedirs('preds/full')

# for each model and settings, we train it, get the top5 and top10, and save name -> setting

# # itemknn k=10
# model_name_file, results = run_recbole(model='ItemKNN', dataset=amazon_elec, config_dict={'tensorboard': False,
#                                                     'k': 10,
#                                                     'benchmark_filename': ['train', 'valid', 'test']})
# get_topk(model_name_file)
# with open('logs.txt', 'a') as f:
#     f.write(f"{model_name_file}\tk=10\n")

# # itemknn k=50
# model_name_file, results = run_recbole(model='ItemKNN', dataset=amazon_elec, config_dict={'tensorboard': False,
#                                                     'k': 50,
#                                                     'benchmark_filename': ['train', 'valid', 'test']})
# get_topk(model_name_file)
# with open('logs.txt', 'a') as f:
#     f.write(f"{model_name_file}\tk=50\n")

# # itemknn k=100
# model_name_file, results = run_recbole(model='ItemKNN', dataset=amazon_elec, config_dict={'tensorboard': False,
#                                                     'k': 100,
#                                                     'benchmark_filename': ['train', 'valid', 'test']})
# get_topk(model_name_file)
# with open('logs.txt', 'a') as f:
#     f.write(f"{model_name_file}\tk=100\n")

# bpr
model_name_file, results = run_recbole(model='BPR', dataset=amazon_elec, config_dict={'tensorboard': False, 
                                                    'epochs': 200, 'eval_step': 10,
                                                    'benchmark_filename': ['train', 'valid', 'test']})
get_topk(model_name_file)
with open('logs.txt', 'a') as f:
    f.write(f"{model_name_file}\tk=64\n")

# lightgcn layer=1
model_name_file, results = run_recbole(model='LightGCN', dataset=amazon_elec, config_dict={'tensorboard': False, 'use_gpu': True,
                                                    'epochs': 200, 'eval_step': 10,
                                                    'n_layers': 1,
                                                    'benchmark_filename': ['train', 'valid', 'test']})
get_topk(model_name_file)
with open('logs.txt', 'a') as f:
    f.write(f"{model_name_file}\tn_layers=1\n")

# lightgcn layer=2
model_name_file, results = run_recbole(model='LightGCN', dataset=amazon_elec, config_dict={'tensorboard': False, 
                                                    'epochs': 200, 'eval_step': 10,
                                                    'n_layers': 2,
                                                    'benchmark_filename': ['train', 'valid', 'test']})
get_topk(model_name_file)
with open('logs.txt', 'a') as f:
    f.write(f"{model_name_file}\tn_layers=2\n")

# lightgcn layer=3
model_name_file, results = run_recbole(model='LightGCN', dataset=amazon_elec, config_dict={'tensorboard': False, 
                                                    'epochs': 200, 'eval_step': 10,
                                                    'n_layers': 3,
                                                    'benchmark_filename': ['train', 'valid', 'test']})
get_topk(model_name_file)
with open('logs.txt', 'a') as f:
    f.write(f"{model_name_file}\tn_layers=3\n")