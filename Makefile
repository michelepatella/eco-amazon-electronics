conda_env = eco-amazon-electronics

# ===================================
# DATA PREPROCESSING
# ===================================
preprocess_data:
	mkdir -p logs/pipeline
	@echo CONFIGURATIONS | tee -a logs/pipeline/preprocess_data.log
	@python -c "print('*' * 100)" | tee -a logs/pipeline/preprocess_data.log
	@tail -n +1 src/config/* 2>/dev/null | tee -a logs/pipeline/preprocess_data.log || true
	@python -c "print('*' * 100)" | tee -a logs/pipeline/preprocess_data.log
	conda run --no-capture-output -n $(conda_env) python -u -m src.data.preprocess 2>&1 | tee -a logs/pipeline/preprocess_data.log

# ===================================
# EMISSION DATA ENRICHMENT
# ===================================
enrich_data_with_emissions:
	mkdir -p logs/pipeline
	@echo CONFIGURATIONS | tee -a logs/pipeline/enrich_data_with_emissions.log
	@python -c "print('*' * 100)" | tee -a logs/pipeline/enrich_data_with_emissions.log
	@tail -n +1 src/config/* 2>/dev/null | tee -a logs/pipeline/enrich_data_with_emissions.log || true
	@python -c "print('*' * 100)" | tee -a logs/pipeline/enrich_data_with_emissions.log
	conda run --no-capture-output -n $(conda_env) python -u -m src.data.enrich_with_emissions 2>&1 | tee -a logs/pipeline/enrich_data_with_emissions.log

# ===================================
# MODEL TRAINING
# ===================================
train_recsys:
	mkdir -p logs/pipeline
	@echo CONFIGURATIONS | tee -a logs/pipeline/train_recsys.log
	@python -c "print('*' * 100)" | tee -a logs/pipeline/train_recsys.log
	@tail -n +1 src/config/* 2>/dev/null | tee -a logs/pipeline/train_recsys.log || true
	@python -c "print('*' * 100)" | tee -a logs/pipeline/train_recsys.log
	conda run --no-capture-output -n $(conda_env) python -u -m src.modeling.train 2>&1 | tee -a logs/pipeline/train_recsys.log

# ===================================
# MODEL INFERENCE
# +
# SUSTAINABILITY-AWARE RECOMMENDATION
# RE-RANKING
# ===================================
predict_recommendations:
	mkdir -p logs/pipeline
	@echo CONFIGURATIONS | tee -a logs/pipeline/predict_recommendations.log
	@python -c "print('*' * 100)" | tee -a logs/pipeline/predict_recommendations.log
	@tail -n +1 src/config/* 2>/dev/null | tee -a logs/pipeline/predict_recommendations.log || true
	@python -c "print('*' * 100)" | tee -a logs/pipeline/predict_recommendations.log
	conda run --no-capture-output -n $(conda_env) python -u -m src.modeling.predict 2>&1 | tee -a logs/pipeline/predict_recommendations.log

# ===================================
# MODEL EVALUATION
# ===================================
evaluate_recsys:
	mkdir -p logs/pipeline
	@echo CONFIGURATIONS | tee -a logs/pipeline/evaluate_recsys.log
	@python -c "print('*' * 100)" | tee -a logs/pipeline/evaluate_recsys.log
	@tail -n +1 src/config/* 2>/dev/null | tee -a logs/pipeline/evaluate_recsys.log || true
	@python -c "print('*' * 100)" | tee -a logs/pipeline/evaluate_recsys.log
	conda run --no-capture-output -n $(conda_env) python -u -m src.modeling.evaluate 2>&1 | tee -a logs/pipeline/evaluate_recsys.log
