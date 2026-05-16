conda_env = eco-amazon-electronics

# ===================================
# DATA PREPROCESSING
# ===================================
preprocess_data:
	mkdir -p pipeline_logs
	conda run --no-capture-output -n $(conda_env) python -u -m src.data.preprocess 2>&1 | tee -a pipeline_logs/preprocess_data.log

# ===================================
# EMISSION DATA ENRICHMENT
# ===================================
enrich_data_with_emissions:
	mkdir -p pipeline_logs
	conda run --no-capture-output -n $(conda_env) python -u -m src.data.enrich_with_emissions 2>&1 | tee -a pipeline_logs/enrich_data_with_emissions.log

# ===================================
# MODEL TRAINING
# ===================================
train_recsys:
	mkdir -p pipeline_logs
	conda run --no-capture-output -n $(conda_env) python -u -m src.modeling.train 2>&1 | tee -a pipeline_logs/train_recsys.log

# ===================================
# MODEL INFERENCE
# +
# SUSTAINABILITY-AWARE RECOMMENDATION
# RE-RANKING
# ===================================
predict_recommendations:
	mkdir -p pipeline_logs
	conda run --no-capture-output -n $(conda_env) python -u -m src.modeling.predict 2>&1 | tee -a pipeline_logs/predict_recommendations.log

# ===================================
# MODEL EVALUATION
# ===================================
evaluate_recsys:
	mkdir -p pipeline_logs
	conda run --no-capture-output -n $(conda_env) python -u -m src.modeling.evaluate 2>&1 | tee -a pipeline_logs/evaluate_recsys.log
