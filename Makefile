# ===================================
# DATA PREPROCESSING
# ===================================
preprocess_data:
	python -m src.data.preprocess

# ===================================
# EMISSION DATA ENRICHMENT
# ===================================
enrich_data_with_emissions:
	python -m src.data.enrich_with_emissions

# ===================================
# MODEL TRAINING
# ===================================
train_recsys:
	python -m src.modeling.train

# ===================================
# MODEL INFERENCE
# +
# SUSTAINABILITY-AWARE RECOMMENDATION
# RE-RANKING
# ===================================
predict_recommendations:
	python -m src.modeling.predict

# ===================================
# MODEL EVALUATION
# ===================================
evaluate_recsys:
	python -m src.modeling.evaluate
