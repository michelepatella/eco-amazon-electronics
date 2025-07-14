# Repository for the paper Estimating Product Carbon Footprint via Large Language Models for Sustainable Recommender Systems

In this repository, we provide the scripts to reproduce our workflow, composed in mainly three steps.

1. The first step of our methodology consists in reading the Electronics datasets and process it with an LLM to obtain the quantity of CO2-eq emitted for the production of the item. The scripts for this step are in the folder `1_pcf/pcf.py`. This script read the metadata files in the `metadata_split` folder, and, for each item, provides the LLM estimation of the CO2-eq. Any LLM can be used at this point. To execute this, you just need to run:
```
python pcf.py
```
The output of this process is a file named `metadata.json`, which consists in the original metadata file, in addition to the predicted CO2-eq.

2. The second step consists in training the recommendation model with [RecBole](https://recbole.io/docs/). The scripts for this step are in the folder `2_recbole`. To this purpose, we remapped the items from 0 to n-1, as it is required by RecBole data format (script `process.py`). 
Then, we trained the `BPR` and `LightGCN` models and saved the prediction lists in the folder `preds`. Note that at this point, any recommendation model can be trained.
Please refer to the RecBole documentation for the environmental setting.
To train these models with our data format, just run
```
python train_recsys.py
```

Normalizzazione?

3. The third step consists in the re-ranking of the prediction lists generated at the previous step. To this aim, the folder `3_re-ranking` provides the script `rerank_linear_comb.py`. This script reads the prediction lists and the augmented dataset, then applies the reranking strategy with different importance weights, and returns the reranked list by saving them in the `reranked_linear_combination_full` folder. To run the script, run the command:
```
python rerank_linear_comb.py
```

4. The last step consists in the evaluation process, which can be performed with any evaluation framework. In our work we used [ClayRS](), for which we suggest to refer to set up the environment. The script to run the evaluation can be found in the `4_eval` folder. The scripts read a set of recommendation lists (both pre-reraking and post-reranking) and evaluate them in terms of classic metrics like Precision, recall, ndcg, f1, gini index. To run the evaluation, after setting up the environment, run the command 
```
python eval.py
```.