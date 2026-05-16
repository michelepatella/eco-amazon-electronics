"""src/data/enrich_with_emissions.py

Enrich product with emissions data.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
from reco2gnizer.context import Context
from reco2gnizer.graph import graph
from reco2gnizer.state import InputState
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm.asyncio import tqdm

from src.const import (
    DATA_PROCESSED_AR23_IM_G25F_ELEC_PATH,
    DATA_PROCESSED_AR23_IM_O3M_ELEC_PATH,
    DATA_PROCESSED_GT_G25F_ELEC_PATH,
    DATA_PROCESSED_GT_O3M_ELEC_PATH,
    DATA_RAW_AR23_IM_ELEC_PATH,
    DATA_RAW_GT_ELEC_PATH,
    DATASET_NAME_ELEC,
    LLM_NAME_G25F,
    LLM_NAME_O3M,
    SUPPORTED_DATASETS,
)
from src.utils import load_config, load_jsonl

# Load configuration
config = load_config()

# Determine paths based on dataset name
assert config["dataset"]["name"] in SUPPORTED_DATASETS, (
    f"Supported dataset: {SUPPORTED_DATASETS}, got: {config['dataset']['name']}"
)
if config["dataset"]["name"] == DATASET_NAME_ELEC:
    # Map data types and models to their raw and processed paths
    path_registry = {
        "ground_truth": {
            LLM_NAME_G25F: {
                "raw_path": DATA_RAW_GT_ELEC_PATH,
                "processed_path": DATA_PROCESSED_GT_G25F_ELEC_PATH,
            },
            LLM_NAME_O3M: {
                "raw_path": DATA_RAW_GT_ELEC_PATH,
                "processed_path": DATA_PROCESSED_GT_O3M_ELEC_PATH,
            },
        },
        "item_metadata": {
            LLM_NAME_G25F: {
                "raw_path": DATA_RAW_AR23_IM_ELEC_PATH,
                "processed_path": DATA_PROCESSED_AR23_IM_G25F_ELEC_PATH,
            },
            LLM_NAME_O3M: {
                "raw_path": DATA_RAW_AR23_IM_ELEC_PATH,
                "processed_path": DATA_PROCESSED_AR23_IM_O3M_ELEC_PATH,
            },
        },
    }

# Determine data type
data_type = (
    "ground_truth"
    if config["emissions_enrichment"]["is_ground_truth"]
    else "item_metadata"
)

# Determine model
model = config["emissions_enrichment"]["model"]

# Get paths from registry
paths = path_registry[data_type][model]
raw_jsonl_file = paths["raw_path"]
processed_jsonl_file = paths["processed_path"]

baseline_key_map = {
    LLM_NAME_G25F: "co2e_kg_baseline_gemini_2_5_flash_estimates",
    LLM_NAME_O3M: "co2e_kg_baseline_openai_o3_mini_estimates",
}
baseline_estimates_key = baseline_key_map[model]


def _calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error (RMSE).

    This function computes RMSE, representing the square root of the average
    squared differences between true and predicted values. It is sensitive to
    large errors and provides a measure of the magnitude of prediction errors.

    Args:
        y_true (np.ndarray):
            True values.

        y_pred (np.ndarray):
            Predicted values.

    Returns:
        float:
            RMSE value in the same unit as the input data.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _calculate_me(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Error (ME).

    This function computes ME, representing the average difference between
    true and predicted values. It preserves the sign of the error to
    indicate systematic overestimation or underestimation.

    Args:
        y_true (np.ndarray):
            True values.

        y_pred (np.ndarray):
            Predicted values.

    Returns:
        float:
            ME value in the same unit as the input data.
    """
    return float(np.mean(y_true - y_pred))


def _calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error (MAPE).

    This function computes MAPE, representing the average absolute percentage
    difference between true and predicted values.

    Args:
        y_true (np.ndarray):
            True values.

        y_pred (np.ndarray):
            Predicted values.

    Returns:
        float:
            MAPE value (0-100).
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def _calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Weighted Absolute Percentage Error (WAPE).

    This function computes WAPE, representing the total absolute error
    weighted by the sum of the true values. It is more robust than MAPE
    when dealing with true values close to or equal to zero.

    Args:
        y_true (np.ndarray):
            True values.

        y_pred (np.ndarray):
            Predicted values.

    Returns:
        float:
            WAPE value (0-100 under normal conditions, can exceed 100).
    """
    sum_absolute_error = np.sum(np.abs(y_true - y_pred))
    sum_true_values = np.sum(np.abs(y_true))

    if sum_true_values == 0:
        return 0.0 if sum_absolute_error == 0 else float("inf")

    return (sum_absolute_error / sum_true_values) * 100


def _calculate_ndcg(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG).

    This function computes NDCG, a ranking metric that evaluates the quality of
    predicted rankings based on the true relevance of items.

    Args:
        y_true (np.ndarray):
            True values (relevance scores).

        y_pred (np.ndarray):
            Predicted values (ranking scores).

    Returns:
        float:
            NDCG value (0-1).
    """
    # Get indices that would sort predictions in descending order
    pred_indices = np.argsort(-y_pred)

    # Get the true values (relevance) in the order of predictions
    relevance = y_true[pred_indices]

    # Calculate DCG: sum of (relevance / log2(position + 1))
    positions = np.arange(1, len(relevance) + 1)
    dcg = np.sum(relevance / np.log2(positions + 1))

    # Calculate IDCG: DCG of ideal ranking (when items are sorted
    # by true values in descending order)
    ideal_relevance = np.sort(y_true)[::-1]
    ideal_positions = np.arange(1, len(ideal_relevance) + 1)
    idcg = np.sum(ideal_relevance / np.log2(ideal_positions + 1))

    # NDCG = DCG / IDCG
    return dcg / idcg if idcg > 0 else 0.0


def _calculate_rmse_to_mae_ratio(rmse: float, mae: float) -> float:
    """Calculate the ratio of RMSE to MAE.

    This function computes the ratio of RMSE to MAE, providing insight into
    the distribution of errors. A ratio close to 1 suggests that errors are
    uniformly distributed, while a higher ratio indicates the presence of
    larger outliers.

    Args:
        rmse (float):
            Root Mean Squared Error value.

        mae (float):
            Mean Absolute Error value.

    Returns:
        float:
            Ratio of RMSE to MAE (unitless).
    """
    if mae == 0.0:
        return float("inf") if rmse > 0 else 1.0

    return rmse / mae


def _calculate_metric_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute a comprehensive set of metrics.

    This function computes the following metrics:
    - Mean Absolute Error (MAE)
    - Mean Error (ME)
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Percentage Error (MAPE)
    - Weighted Absolute Percentage Error (WAPE)
    - Normalized Discounted Cumulative Gain (NDCG)
    - Spearman's Rank Correlation Coefficient
    - RMSE to MAE Ratio

    Args:
        y_true (np.ndarray):
            Array of true values.

        y_pred (np.ndarray):
            Array of predicted values.

    Returns:
        dict[str, float]:
            Dictionary containing all computed metric values.
    """
    mae = mean_absolute_error(y_true, y_pred)
    me = _calculate_me(y_true, y_pred)
    rmse = _calculate_rmse(y_true, y_pred)
    mape = _calculate_mape(y_true, y_pred)
    wape = _calculate_wape(y_true, y_pred)
    ndcg = _calculate_ndcg(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)
    rmse_to_mae_ratio = _calculate_rmse_to_mae_ratio(rmse, mae)

    return {
        "MAE": mae,
        "ME": me,
        "RMSE": rmse,
        "MAPE": mape,
        "WAPE": wape,
        "NDCG": ndcg,
        "Spearman": spearman_corr,
        "RMSE/MAE": rmse_to_mae_ratio,
    }


async def _process_single_product(
    product_data: dict[str, Any],
    context: Context,
    semaphore: asyncio.Semaphore,
    is_ground_truth: bool,
) -> dict[str, Any]:
    """Process a single product asynchronously with concurrency limit.

    This function processes a single product by invoking the agent to compute the
    carbon footprint estimate. It uses an asyncio semaphore to limit the number
    of concurrent invocations. The function extracts the carbon footprint value
    from the agent's result and updates the product data.

    Args:
        product_data (dict[str, Any]):
            Product data dictionary.

        context (Context):
            Context configuration for the agent.

        semaphore (asyncio.Semaphore):
            Asyncio semaphore to limit concurrency.

        is_ground_truth (bool):
            Flag indicating whether the product data comes from the ground
            truth or not.

    Returns:
        dict[str, Any]:
            Updated product data.
    """
    async with semaphore:
        # Create input state for the agent with product data
        title = product_data["title"]
        input_state = InputState(product_data={"title": title})

        # Initialize the appropriate field in product data to store
        # the carbon footprint
        if is_ground_truth:
            if "co2e_kg_estimates" not in product_data:
                product_data["co2e_kg_estimates"] = []
        elif "co2e_kg" not in product_data:
            product_data["co2e_kg"] = None

        try:
            # Invoke the agent
            result = await graph.ainvoke(input=input_state, context=context)

            # Extract carbon footprint value from result
            # and save it in the product data
            co2e_kg = result["co2e_kg"]
            if is_ground_truth:
                product_data["co2e_kg_estimates"].append(co2e_kg)
            else:
                product_data["co2e_kg"] = co2e_kg
        except Exception as e:
            print(f"Error processing '{title}': {e!s}")

            # If there's an error, set the carbon footprint value to
            # None to indicate failure
            if is_ground_truth:
                product_data["co2e_kg_estimates"].append(None)
            else:
                product_data["co2e_kg"] = None

        return product_data


async def _calculate_metrics(
    products: list,
    compare_with_baseline: bool,
) -> None:
    """Calculate metrics from products list.

    This function computes metrics from a matrix where rows represent products
    and columns represent independent estimate calls. Metrics are calculated
    column-wise (across products for each call) while statistics are calculated
    row-wise (consistency within each product across calls). Optionally compares
    current metrics with baseline.

    Args:
        products (list):
            List of product data dictionaries.

        compare_with_baseline (bool):
            If True, compares current metrics with baseline metrics.

    Returns:
        None
    """
    # Extract true values
    true_values = np.array([float(p["co2e_kg"]) for p in products])
    num_products = len(products)
    num_calls = config["emissions_enrichment"]["num_estimates_per_product"]

    # Populate current prediction matrix (rows=products, columns=calls) with
    # estimates from products data
    current_preds_matrix = np.full((num_products, num_calls), np.nan)
    for row_idx, product_data in enumerate(products):
        current_estimates = product_data["co2e_kg_estimates"]
        for col_idx in range(min(len(current_estimates), num_calls)):
            if current_estimates[col_idx] is not None:
                current_preds_matrix[row_idx, col_idx] = float(
                    current_estimates[col_idx],
                )

    # Calculate current column-wise metrics (across products for each call)
    current_call_metrics = [
        _calculate_metric_values(true_values, current_preds_matrix[:, i])
        for i in range(num_calls)
    ]
    current_metrics = (
        {
            k: np.mean([m[k] for m in current_call_metrics])
            for k in current_call_metrics[0]
        }
        if current_call_metrics
        else {}
    )

    # Calculate current row-wise statistics (within each product across calls)
    current_means = np.nanmean(current_preds_matrix, axis=1)
    current_stds = np.nanstd(current_preds_matrix, axis=1)
    current_metrics["StdDev"] = (
        float(np.nanmean(current_stds)) if len(current_stds) > 0 else 0.0
    )
    current_metrics["CV"] = (
        float(
            np.nanmean(
                np.where(
                    current_means != 0,
                    current_stds / current_means,
                    0.0,
                ),
            ),
        )
        if len(current_means) > 0
        else 0.0
    )

    baseline_metrics = {}
    if compare_with_baseline:
        # Populate baseline prediction matrix (rows=products, columns=calls) with
        # baseline estimates from products data
        baseline_preds_matrix = np.full((num_products, num_calls), np.nan)
        for row_idx, product_data in enumerate(products):
            baseline_estimates = product_data[baseline_estimates_key]
            baseline_preds_matrix[row_idx, : len(baseline_estimates)] = [
                float(x) if x is not None else np.nan
                for x in baseline_estimates[:num_calls]
            ]

        # Calculate baseline column-wise metrics (across products for each call)
        baseline_call_metrics = [
            _calculate_metric_values(true_values, baseline_preds_matrix[:, i])
            for i in range(num_calls)
        ]
        baseline_metrics = (
            {
                k: np.mean([m[k] for m in baseline_call_metrics])
                for k in baseline_call_metrics[0]
            }
            if baseline_call_metrics
            else {}
        )

        # Calculate baseline row-wise statistics (within each product across calls)
        baseline_means = np.nanmean(baseline_preds_matrix, axis=1)
        baseline_stds = np.nanstd(baseline_preds_matrix, axis=1)
        baseline_metrics["StdDev"] = (
            float(np.nanmean(baseline_stds)) if len(baseline_stds) > 0 else 0.0
        )
        baseline_metrics["CV"] = (
            float(
                np.nanmean(
                    np.where(
                        baseline_means != 0,
                        baseline_stds / baseline_means,
                        0.0,
                    ),
                ),
            )
            if len(baseline_means) > 0
            else 0.0
        )

    # Display results
    print("=" * 80)
    print("Results")
    print("=" * 80)

    # Baseline metrics (if baseline comparison is enabled)
    if compare_with_baseline:
        print("\nBaseline System:")
        print("-" * 80)
        for metric_name, metric_value in baseline_metrics.items():
            if isinstance(metric_value, (float, np.floating)):
                print(f"  {metric_name:12s}: {metric_value:12.4f}")
            else:
                print(f"  {metric_name:12s}: {metric_value}")

    # Current system metrics
    print("\nCurrent System:")
    print("-" * 80)
    for metric_name, metric_value in current_metrics.items():
        if isinstance(metric_value, (float, np.floating)):
            print(f"  {metric_name:12s}: {metric_value:12.4f}")
        else:
            print(f"  {metric_name:12s}: {metric_value}")

    # Baseline-current system comparison (if baseline comparison is enabled)
    if compare_with_baseline:
        print("\nBaseline vs. Current System:")
        print("-" * 80)
        for metric_name in baseline_metrics:
            baseline_val = baseline_metrics[metric_name]
            current_val = current_metrics[metric_name]

            # For ranking metrics, higher is better
            if metric_name in ["NDCG", "Spearman"]:
                improvement = current_val - baseline_val
                improvement_pct = (
                    (improvement / abs(baseline_val) * 100)
                    if baseline_val != 0
                    else 0
                )
                direction = "↑" if improvement > 0 else "↓"

            # For error metrics and statistics, lower is better
            else:
                improvement = baseline_val - current_val
                improvement_pct = (
                    (improvement / abs(baseline_val) * 100)
                    if baseline_val != 0
                    else 0
                )
                direction = "↓" if improvement > 0 else "↑"

            print(
                f"  {metric_name:12s}: {improvement:+.4f} ({improvement_pct:+.2f}%) {direction}",
            )

    print("\n" + "=" * 80)


async def enrich_data_with_emissions() -> None:
    """Enrich product with emissions data.

    This function enriches product data with emissions estimates by processing
    products from a raw JSONL file. It checks if emissions data is already
    computed for each product and only processes those that require it. The
    function uses asynchronous processing with concurrency limits to efficiently
    compute emissions estimates for products. After processing, it updates the
    product data and writes it back to a processed JSONL file. If the data being
    processed is ground truth data, it also calculates and displays metrics
    comparing the estimates with the ground truth values.

    Returns:
        None
    """
    # Load products from JSONL file and check if all the
    # emissions are already computed
    products = []
    products_to_process = []
    for idx, product_data in enumerate(load_jsonl(str(Path(raw_jsonl_file)))):
        # Save product data
        products.append(product_data)

        # Check if emissions are already computed for this product
        if config["emissions_enrichment"]["is_ground_truth"]:
            if "co2e_kg_estimates" in product_data:
                # Filter out None values from estimates
                valid_estimates = [
                    x
                    for x in product_data["co2e_kg_estimates"]
                    if x is not None
                ]
                if (
                    len(valid_estimates)
                    < config["emissions_enrichment"][
                        "num_estimates_per_product"
                    ]
                ):
                    # This product does not have the required number of
                    # emission estimates
                    products_to_process.append((idx, product_data))
            else:
                # This product does not have any emission estimates
                products_to_process.append((idx, product_data))
        elif "co2e_kg" in product_data:
            if product_data["co2e_kg"] is None:
                # This product does not have a valid emission value
                products_to_process.append((idx, product_data))
        else:
            # This product does not have any emission data
            products_to_process.append((idx, product_data))

    num_initial_products = len(products)

    # Sort products by 'co2e_kg' in descending order
    products.sort(
        key=lambda x: (
            x["co2e_kg"] if x["co2e_kg"] is not None else float("-inf")
        ),
        reverse=True,
    )

    # If all products already have the required emission data and
    # we are processing ground truth data, skip processing and directly
    # calculate metrics comparing emissions estimates with the ground truth value
    if (
        len(products_to_process) == 0
        and config["emissions_enrichment"]["is_ground_truth"]
    ):
        print(
            f"All estimates already computed in {raw_jsonl_file}, calculating metrics...",
        )
        await _calculate_metrics(
            products,
            compare_with_baseline=config["emissions_enrichment"][
                "compare_with_baseline"
            ],
        )
        return

    # If all products already have the required emission data and we are
    # processing non-ground truth data, skip processing
    if (
        len(products_to_process) == 0
        and not config["emissions_enrichment"]["is_ground_truth"]
    ):
        print(f"All estimates already computed in {raw_jsonl_file}!")

    # Regardless of whether we are processing ground truth data or not,
    # if there are products to process, we need to compute the estimates
    # for those products
    else:
        print(
            f"Identified {len(products_to_process)} products to process from {raw_jsonl_file}.",
        )

    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(
        config["emissions_enrichment"]["max_concurrent_requests"],
    )

    # Process products
    tasks = []
    task_to_product_idx = {}

    for idx, product_data in products_to_process:
        # Calculate how many runs are needed for this product
        # based on whether we are processing ground truth data
        # or not and how many valid estimates it already has
        # (in case of ground truth data)
        if config["emissions_enrichment"]["is_ground_truth"]:
            valid_estimates = [
                x
                for x in product_data.get("co2e_kg_estimates", [])
                if x is not None
            ]
            runs_needed = config["emissions_enrichment"][
                "num_estimates_per_product"
            ] - len(valid_estimates)
        else:
            runs_needed = 1

        # Generate the required number of tasks for this product based on
        # the runs needed
        for _ in range(runs_needed):
            task = asyncio.create_task(
                _process_single_product(
                    product_data=product_data,
                    context=Context(
                        model=config["emissions_enrichment"]["model"],
                        model_temperature=config["emissions_enrichment"][
                            "model_temperature"
                        ],
                        search_web_max_results=config["emissions_enrichment"][
                            "search_web_max_results"
                        ],
                        search_web_type=config["emissions_enrichment"][
                            "search_web_type"
                        ],
                        search_web_max_calls=config["emissions_enrichment"][
                            "search_web_max_calls"
                        ],
                        enable_logging=config["emissions_enrichment"][
                            "enable_logging"
                        ],
                        logs_dir=config["emissions_enrichment"]["logs_dir"],
                        logs_path=f"{product_data['title'].replace(' ', '_')}.log",
                    ),
                    semaphore=semaphore,
                    is_ground_truth=config["emissions_enrichment"][
                        "is_ground_truth"
                    ],
                ),
            )
            tasks.append(task)
            task_to_product_idx[task] = idx

    if not tasks:
        return

    # Use tqdm to display progress bar while processing products
    with tqdm(
        total=len(tasks),
        desc="Enriching products with emission data...",
    ) as pbar:
        # Iterate over tasks as they complete to monitor progress and catch errors
        for next_to_complete in asyncio.as_completed(tasks):
            try:
                # Await the task execution directly (fixes the unpacking bug)
                await next_to_complete
            except Exception as e:
                print(f"\nError processing a single product run: {e}")
            finally:
                pbar.update(1)

    # Sort final products by 'co2e_kg' in descending order before saving
    products.sort(
        key=lambda x: (
            x["co2e_kg"] if x["co2e_kg"] is not None else float("-inf")
        ),
        reverse=True,
    )

    # Write fully updated and sorted products data back to processed JSONL file
    with open(processed_jsonl_file, "w") as f:
        f.writelines(json.dumps(product) + "\n" for product in products)

    if config["emissions_enrichment"]["is_ground_truth"]:
        # Calculate and display metrics
        print("Calculating metrics...")
        await _calculate_metrics(
            products,
            compare_with_baseline=config["emissions_enrichment"][
                "compare_with_baseline"
            ],
        )

    # Sanity checks
    vals = [p["co2e_kg"] for p in products if p["co2e_kg"] is not None]
    assert len(products) == num_initial_products, (
        f"Product count mismatch: expected {num_initial_products}, got {len(products)}"
    )
    assert len(products) == len({p["title"] for p in products}), (
        "Duplicate titles found"
    )
    assert all(v >= 0 for v in vals), (
        f"Negative co2e_kg found: {[v for v in vals if v < 0]}"
    )
    assert all(i >= j for i, j in zip(vals, vals[1:])), (
        "Products are not sorted in descending order"
    )


if __name__ == "__main__":
    asyncio.run(enrich_data_with_emissions())
