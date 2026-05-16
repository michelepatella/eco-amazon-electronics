# `src/`

```text
.
├── config
│   ├── dataset.yaml                    <- Dataset configuration
│   ├── emissions_enrichment.yaml       <- Emissions enrichment settings
│   ├── evaluation.yaml                 <- Model evaluation configuration
│   ├── inference.yaml                  <- Inference pipeline settings
│   ├── param_space.yaml                <- Hyperparameter tuning space
│   ├── preprocessing.yaml              <- Data preprocessing settings
│   ├── ray_tune.yaml                   <- Ray Tune configuration
│   └── recbole.yaml                    <- RecBole configuration
├── data
│   ├── enrich_with_emissions.py        <- Emission estimates product data enrichment
│   └── preprocess.py                   <- Dataset preprocessing
├── modeling
│   ├── evaluate.py                     <- Model performance evaluation
│   ├── predict.py                      <- Recommendation generation and re-ranking
│   └── train.py                        <- Model training pipeline
├── README.md                           <- Structure documentation (this file)
├── const.py                            <- Global constants
└── utils.py                            <- Utility functions
```
