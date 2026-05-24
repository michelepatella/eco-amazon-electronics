# `models/`

Directory for all trained recommendation models, their best hyperparameters and re-ranked predictions.

**Note**: This project uses DVC to manage model files. Make sure `dvc` is installed, then pull all model artifacts from the remote bucket:
```bash
dvc pull
```

## `electronics/BPR/`

| Name | Type | Format | Dataset | Parameters | Size | Description |
|------|------|--------|---------|-----------|------|-------------|
| `BPR.pth` | Model | PyTorch | Amazon Reviews'23 (Electronics) | `embedding_size=64` | 25.5 MB | Trained BPR model on 15-core filtered, binarized user-item interactions |
| `best_params.json` | Best Hyperparameters | JSON | Amazon Reviews'23 (Electronics) | `train_batch_size=512`, `learning_rate=0.0003`, `weight_decay=1e-6`, `embedding_size=64` | 107 byte | Best hyperparameters found for BPR model after HPO. |
| `predictions/google_genai:gemini-2.5-flash/alpha_0_25.pth` | Predictions | PyTorch | Amazon Reviews'23 (Electronics) | `alpha=0.25` | - | Re-ranked BPR predictions on Gemini 2.5 Flash PCF estimates (sustainability-first) |
| `predictions/google_genai:gemini-2.5-flash/alpha_0_5.pth` | Predictions | PyTorch | Amazon Reviews'23 (Electronics) | `alpha=0.50`| - | Re-ranked BPR predictions on Gemini 2.5 Flash PCF estimates (balanced relevance-sustainability trade-off) |
| `predictions/google_genai:gemini-2.5-flash/alpha_0_75.pth` | Predictions | PyTorch | Amazon Reviews'23 (Electronics) | `alpha=0.75` | - | Re-ranked BPR predictions on Gemini 2.5 Flash PCF estimates (relevance-first) |
| `predictions/google_genai:gemini-2.5-flash/alpha_1_0.pth` | Predictions | PyTorch | Amazon Reviews'23 (Electronics) | `alpha=1.00` | - | Re-ranked BPR predictions on Gemini 2.5 Flash PCF estimates (pure relevance, no sustainability) |

## `electronics/LightGCN/`

| Name | Type | Format | Dataset | Parameters | Size | Description |
|------|------|--------|---------|-----------|------|-------------|
| `LightGCN.pth` | Model | PyTorch | Amazon Reviews'23 (Electronics) | `embedding_size=128`, `n_layers=3`, `reg_weight=1e-5`, `require_pow=False` | 68 MB | Trained LightGCN model on 15-core filtered, binarized user-item interactions |
| `best_params.json` | Best Hyperparameters | JSON | Amazon Reviews'23 (Electronics) | `train_batch_size=1024`, `learning_rate=0.003`, `weight_decay=1e-6`, `embedding_size=128`, `n_layers=3`, `reg_weight=1e-5` | 148 byte | Best hyperparameters found for LightGCN model after HPO. |
| `predictions/google_genai:gemini-2.5-flash/alpha_0_25.pth` | Predictions | PyTorch | Amazon Reviews'23 (Electronics) | `alpha=0.25` | - | Re-ranked LightGCN predictions on Gemini 2.5 Flash PCF estimates (sustainability-first) |
| `predictions/google_genai:gemini-2.5-flash/alpha_0_5.pth` | Predictions | PyTorch | Amazon Reviews'23 (Electronics) | `alpha=0.50` | - | Re-ranked LightGCN predictions on Gemini 2.5 Flash PCF estimates (balanced relevance-sustainability trade-off) |
| `predictions/google_genai:gemini-2.5-flash/alpha_0_75.pth` | Predictions | PyTorch | Amazon Reviews'23 (Electronics) | `alpha=0.75` | - | Re-ranked LightGCN predictions on Gemini 2.5 Flash PCF estimates (relevance-first) |
| `predictions/google_genai:gemini-2.5-flash/alpha_1_0.pth` | Predictions | PyTorch | Amazon Reviews'23 (Electronics) | `alpha=1.00` | - | Re-ranked LightGCN predictions on Gemini 2.5 Flash PCF estimates (pure relevance, no sustainability) |
