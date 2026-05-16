conda_env = eco-amazon-electronics

# ===================================
# DATA PREPROCESSING
# ===================================
preprocess_data:
	conda run --no-capture-output -n $(conda_env) python -u -m src.data.preprocess

# ===================================
# EMISSION DATA ENRICHMENT
# ===================================
enrich_data_with_emissions:
	conda run --no-capture-output -n $(conda_env) python -u -m src.data.enrich_with_emissions

# ===================================
# MODEL TRAINING
# ===================================
train_recsys:
	conda run --no-capture-output -n $(conda_env) python -u -m src.modeling.train

# ===================================
# MODEL INFERENCE
# +
# SUSTAINABILITY-AWARE RECOMMENDATION
# RE-RANKING
# ===================================
predict_recommendations:
	conda run --no-capture-output -n $(conda_env) python -u -m src.modeling.predict

# ===================================
# MODEL EVALUATION
# ===================================
evaluate_recsys:
	conda run --no-capture-output -n $(conda_env) python -u -m src.modeling.evaluate
