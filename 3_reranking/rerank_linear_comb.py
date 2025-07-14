import pandas as pd
import json
import os
from typing import Dict, List


def top_k_scores(predictions: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Extracts the top-k items for each user based on scores.

    Args:
        predictions: DataFrame with columns ['users', 'items', 'scores']
        k: Number of items to keep per user

    Returns:
        DataFrame with top-k items per user
    """
    top_k = pd.DataFrame()
    for user in predictions['users'].unique():
        user_predictions = predictions[predictions['users'] == user].head(k)
        top_k = pd.concat([top_k, user_predictions])
    return top_k


class EnvironmentalReranker:
    """
    Applies environmentally-aware re-ranking to recommendation lists
    by combining original recommendation scores with greenness scores.
    """

    def __init__(self, item_map_path: str, score_norm_path: str):
        """
        Initializes the reranker with item mapping and greenness scores.

        Args:
            item_map_path: Path to 'item_map.tsv'
            score_norm_path: Path to 'score_norm.json'
        """
        self.item_map = self._load_item_map(item_map_path)
        self.score_norm = self._load_score_norm(score_norm_path)

        # Weighting configurations
        self.weight_configs = {
            'balanced': {'rec_weight': 0.5, 'green_weight': 0.5},
            'rec_focused': {'rec_weight': 0.75, 'green_weight': 0.25},
            'green_focused': {'rec_weight': 0.25, 'green_weight': 0.75},
        }

    def _load_item_map(self, path: str) -> Dict[int, str]:
        """Load item_index → parent_asin mapping."""
        df = pd.read_csv(path, sep='\t')
        return dict(zip(df['item_index'], df['parent_asin']))

    def _load_score_norm(self, path: str) -> Dict[str, float]:
        """Load normalized greenness scores."""
        with open(path, 'r') as f:
            data = json.load(f)

        score_dict = {}
        for item in data:
            if isinstance(item, dict) and 'parent_asin' in item and 'Score_norm' in item:
                score_dict[item['parent_asin']] = item['Score_norm']
        return score_dict

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores between 0 and 1 using min-max scaling."""
        if not scores:
            return scores

        min_score = min(scores)
        max_score = max(scores)

        if min_score == max_score:
            return [0.5] * len(scores)

        return [(s - min_score) / (max_score - min_score) for s in scores]

    def _apply_reranking(self, df: pd.DataFrame, rec_weight: float, green_weight: float) -> pd.DataFrame:
        """
        Apply re-ranking combining rec scores and greenness.

        Args:
            df: Original recommendations
            rec_weight: Weight of recommendation score
            green_weight: Weight of greenness score

        Returns:
            Reranked DataFrame
        """
        df_reranked = df.copy()
        df_reranked['parent_asin'] = df_reranked['items'].map(self.item_map)
        df_reranked['greenness_score'] = df_reranked['parent_asin'].map(lambda x: self.score_norm.get(x, 0.0) if x else 0.0)

        reranked_rows = []
        for user_id in df_reranked['users'].unique():
            user_data = df_reranked[df_reranked['users'] == user_id].copy()

            # Normalize recommendation scores
            user_data['normalized_rec_score'] = self._normalize_scores(user_data['scores'].tolist())

            # Combine scores
            user_data['combined_score'] = (
                rec_weight * user_data['normalized_rec_score'] +
                green_weight * user_data['greenness_score']
            )

            user_data = user_data.sort_values('combined_score', ascending=False)
            user_data['original_rank'] = range(1, len(user_data) + 1)
            user_data['new_rank'] = range(1, len(user_data) + 1)

            reranked_rows.append(user_data)

        return pd.concat(reranked_rows, ignore_index=True)

    def _save_topk_versions(self, reranked_df: pd.DataFrame, output_dir: str, model_name: str, config_name: str):
        """
        Save top-5 and top-10 reranked lists.

        Args:
            reranked_df: Reranked DataFrame
            output_dir: Output directory
            model_name: Model name
            config_name: Configuration name
        """
        topk_df = reranked_df[['users', 'items', 'combined_score']].copy()
        topk_df.columns = ['users', 'items', 'scores']

        top5_dir = os.path.join(output_dir, 'top5')
        top10_dir = os.path.join(output_dir, 'top10')
        os.makedirs(top5_dir, exist_ok=True)
        os.makedirs(top10_dir, exist_ok=True)

        top5 = top_k_scores(topk_df, 5)
        top10 = top_k_scores(topk_df, 10)

        top5_path = os.path.join(top5_dir, f"{model_name}_{config_name}_reranked.tsv")
        top10_path = os.path.join(top10_dir, f"{model_name}_{config_name}_reranked.tsv")

        top5.to_csv(top5_path, sep='\t', index=False)
        top10.to_csv(top10_path, sep='\t', index=False)

        return len(top5), len(top10)

    def rerank_recommendations(self, input_path: str, output_dir: str, model_name: str):
        """
        Apply re-ranking to a single TSV file and save results.

        Args:
            input_path: Path to input TSV
            output_dir: Where to save output
            model_name: Used in output file naming
        """
        df = pd.read_csv(input_path, sep='\t')
        print(f"Processing {model_name} ({len(df)} items for {df['users'].nunique()} users)")

        full_dir = os.path.join(output_dir, 'full')
        os.makedirs(full_dir, exist_ok=True)

        for config_name, weights in self.weight_configs.items():
            print(f"  Applying {config_name} config (rec: {weights['rec_weight']}, green: {weights['green_weight']})")

            reranked_df = self._apply_reranking(df, weights['rec_weight'], weights['green_weight'])

            full_output_file = os.path.join(full_dir, f"{model_name}_{config_name}_reranked.tsv")
            reranked_df.to_csv(full_output_file, sep='\t', index=False)

            # Simplified version (just user, item, combined score)
            simple_output = reranked_df[['users', 'items', 'combined_score']].copy()
            simple_output.columns = ['users', 'items', 'scores']
            simple_file = os.path.join(full_dir, f"{model_name}_{config_name}_simple.tsv")
            simple_output.to_csv(simple_file, sep='\t', index=False)

            # Save top-5 and top-10
            top5_count, top10_count = self._save_topk_versions(reranked_df, output_dir, model_name, config_name)

            print(f"    Items with greenness score: {(reranked_df['greenness_score'] > 0).sum()}/{len(reranked_df)}")
            print(f"    Avg greenness score: {reranked_df['greenness_score'].mean():.4f}")
            print(f"    Full list: {full_output_file}")
            print(f"    Top-5: {top5_count} items | Top-10: {top10_count} items")

    def process_all_recommendations(self, input_dirs: List[str], output_base_dir: str):
        """
        Process recommendation files in multiple directories.

        Args:
            input_dirs: List of directories with .tsv files
            output_base_dir: Base directory to save results
        """
        for input_dir in input_dirs:
            if not os.path.exists(input_dir):
                print(f"Directory {input_dir} not found, skipping...")
                continue

            dir_name = os.path.basename(input_dir)
            output_dir = os.path.join(output_base_dir, f"{dir_name}_reranked")
            os.makedirs(output_dir, exist_ok=True)

            print(f"\n=== Processing directory: {input_dir} ===")

            for filename in os.listdir(input_dir):
                if filename.endswith('.tsv'):
                    model_name = filename.replace('.tsv', '')
                    input_path = os.path.join(input_dir, filename)
                    try:
                        self.rerank_recommendations(input_path, output_dir, model_name)
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")

            print(f"Results saved in: {output_dir}")


def main():
    """Main function to run environmentally-aware re-ranking."""

    # Configuration
    ITEM_MAP_PATH = "../2_recbole/process_data/item_map_2.tsv"
    SCORE_NORM_PATH = "score_norm.json"
    INPUT_DIRS = ["../2_recbole/preds/full"]
    OUTPUT_BASE_DIR = "reranked_linear_combination_full"

    if not os.path.exists(ITEM_MAP_PATH):
        print(f"Error: {ITEM_MAP_PATH} not found!")
        return

    if not os.path.exists(SCORE_NORM_PATH):
        print(f"Error: {SCORE_NORM_PATH} not found!")
        return

    reranker = EnvironmentalReranker(ITEM_MAP_PATH, SCORE_NORM_PATH)
    reranker.process_all_recommendations(INPUT_DIRS, OUTPUT_BASE_DIR)

    print("\n=== Re-ranking complete! ===")
    print(f"Results saved to: {OUTPUT_BASE_DIR}")
    print("Output directory structure:")
    print("├── full/")
    print("├── top5/")
    print("└── top10/")
    print("\nEach model/config includes:")
    print("- Full reranked list")
    print("- Simplified reranked list")
    print("- Top-5 and Top-10 reranked lists")
    print("\nWeight configurations used:")
    print("- balanced: 50% recommendation, 50% greenness")
    print("- rec_focused: 75% recommendation, 25% greenness")
    print("- green_focused: 25% recommendation, 75% greenness")


if __name__ == "__main__":
    main()
