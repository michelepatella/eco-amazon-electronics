# `models/`

Directory for all trained recommendation models and their re-ranked predictions.

**Note**: This project uses DVC to manage model files. Make sure `dvc` is installed, then pull all model artifacts from the remote bucket:
```bash
dvc pull
```

## `BPR/`

| Name | Type | Format | Dataset | Parameters | Size | Description |
|------|------|--------|---------|-----------|------|-------------|
| `BPR.pth` | Model | PyTorch | Amazon Reviews'23 (Electronics) | - | - | Trained BPR model on 15-core filtered, binarized user-item interactions |
| `predictions/gemini_2_5_flash/alpha_0_25.pth` | Predictions | PyTorch | Amazon Reviews'23 (Electronics) | `alpha=0.25` | - | Re-ranked BPR predictions on Gemini 2.5 Flash PCF estimates (sustainability-first) |
| `predictions/gemini_2_5_flash/alpha_0_5.pth` | Predictions | PyTorch | Amazon Reviews'23 (Electronics) | `alpha=0.50`| - | Re-ranked BPR predictions on Gemini 2.5 Flash PCF estimates (balanced relevance-sustainability trade-off) |
| `predictions/gemini_2_5_flash/alpha_0_75.pth` | Predictions | PyTorch | Amazon Reviews'23 (Electronics) | `alpha=0.75` | - | Re-ranked BPR predictions on Gemini 2.5 Flash PCF estimates (relevance-first) |
| `predictions/gemini_2_5_flash/alpha_1_0.pth` | Predictions | PyTorch | Amazon Reviews'23 (Electronics) | `alpha=1.00` | - | Re-ranked BPR predictions on Gemini 2.5 Flash PCF estimates (pure relevance, no sustainability) |
| `predictions/openai_o3_mini/alpha_0_25.pth` | Predictions | PyTorch | Amazon Reviews'23 (Electronics) | `alpha=0.25` | - | Re-ranked BPR predictions on Openai o3-mini PCF estimates (sustainability-first) |
| `predictions/openai_o3_mini/alpha_0_5.pth` | Predictions | PyTorch | Amazon Reviews'23 (Electronics) | `alpha=0.50` | - | Re-ranked BPR predictions on Openai o3-mini PCF estimates (balanced relevance-sustainability trade-off) |
| `predictions/openai_o3_mini/alpha_0_75.pth` | Predictions | PyTorch | Amazon Reviews'23 (Electronics) | `alpha=0.75` | - | Re-ranked BPR predictions on Openai o3-mini PCF estimates (relevance-first) |
| `predictions/openai_o3_mini/alpha_1_0.pth` | Predictions | PyTorch | Amazon Reviews'23 (Electronics) | `alpha=1.00` | - | Re-ranked BPR predictions on Openai o3-mini PCF estimates (pure relevance, no sustainability) |

## `LightGCN/`

| Name | Type | Format | Dataset | Parameters | Size | Description |
|------|------|--------|---------|-----------|------|-------------|
| `LightGCN.pth` | Model | PyTorch | Amazon Reviews'23 (Electronics) | - | - | Trained LightGCN model on 15-core filtered, binarized user-item interactions |
| `predictions/gemini_2_5_flash/alpha_0_25.pth` | Predictions | PyTorch | Amazon Reviews'23 (Electronics) | `alpha=0.25` | - | Re-ranked LightGCN predictions on Gemini 2.5 Flash PCF estimates (sustainability-first) |
| `predictions/gemini_2_5_flash/alpha_0_5.pth` | Predictions | PyTorch | Amazon Reviews'23 (Electronics) | `alpha=0.50` | - | Re-ranked LightGCN predictions on Gemini 2.5 Flash PCF estimates (balanced relevance-sustainability trade-off) |
| `predictions/gemini_2_5_flash/alpha_0_75.pth` | Predictions | PyTorch | Amazon Reviews'23 (Electronics) | `alpha=0.75` | - | Re-ranked LightGCN predictions on Gemini 2.5 Flash PCF estimates (relevance-first) |
| `predictions/gemini_2_5_flash/alpha_1_0.pth` | Predictions | PyTorch | Amazon Reviews'23 (Electronics) | `alpha=1.00` | - | Re-ranked LightGCN predictions on Gemini 2.5 Flash PCF estimates (pure relevance, no sustainability) |
| `predictions/openai_o3_mini/alpha_0_25.pth` | Predictions | PyTorch | Amazon Reviews'23 (Electronics) | `alpha=0.25` | - | Re-ranked LightGCN predictions on OpenAI o3-mini PCF estimates (sustainability-first) |
| `predictions/openai_o3_mini/alpha_0_5.pth` | Predictions | PyTorch | Amazon Reviews'23 (Electronics) | `alpha=0.50` | - | Re-ranked LightGCN predictions on OpenAI o3-mini PCF estimates (balanced relevance-sustainability trade-off) |
| `predictions/openai_o3_mini/alpha_0_75.pth` | Predictions | PyTorch | Amazon Reviews'23 (Electronics) | `alpha=0.75` | - | Re-ranked LightGCN predictions on OpenAI o3-mini PCF estimates (relevance-first) |
| `predictions/openai_o3_mini/alpha_1_0.pth` | Predictions | PyTorch | Amazon Reviews'23 (Electronics) | `alpha=1.00` | - | Re-ranked LightGCN predictions on OpenAI o3-mini PCF estimates (pure relevance, no sustainability) |
