import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import getenv

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def estimate_co2_for_product(product_data, llm_model, num_calls=4):
    """
    Example of LLM prompting to predict the CO2eq of a product
    """

    prompt = f"""
    You are an expert in life cycle analysis (LCA) and CO2e emission calculation for electronic products.
    You must estimate the CO2e emissions, based on the entire life cycle (cradle to grave), for the following electronic product.

    Product data: {json.dumps(product_data, ensure_ascii=False)}

    INSTRUCTIONS:
    1. FIRST, check if there are any official carbon footprint reports or environmental product declarations (EPD) 
       from the manufacturer for this specific product.
       If found, use these official values as your primary source.

    2. If NO official manufacturer reports are available, then estimate emissions following these protocols:
       - GHG Protocol Product Standard for system boundaries and calculation methodology
       - ISO 14040/14044 for Life Cycle Assessment principles
       - PAS 2050 and ISO/TS 14067 for carbon footprint calculation guidelines

    3. For estimation, consider:
       - Main materials composition
       - Manufacturing processes
       - Transportation
       - Use phase energy consumption
       - End-of-life disposal

    4. Use the most recent emission factors and scientific data available
    5. Document your sources and assumptions in the explanation
    6. Clearly state if you're using manufacturer data or estimation

    Reply ONLY with a JSON object containing these exact fields:
    {{
        "co2e_kg": <number>,
        "source": <if "manufacturer report" or "estimation">,
        "explanation": "<detailed explanation including data source>"
    }}
    Do not include any markdown formatting or additional JSON wrappers.
    """

    values = []
    last_valid_response = None

    for _ in range(num_calls):
        # temperature=0.0
        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
        )
        print(response)
        if not response or not response.choices:
            print("Error: the model response is not formatted or empty")
            continue

        try:
            raw_content = response.choices[0].message.content.strip()
            print(f"\nProcessing product: {product_data.get('title')[:60]}...")

            # Extract JSON object if there's additional text
            if "{" in raw_content and "}" in raw_content:
                start = raw_content.rfind("{")  # Get the last JSON object
                end = raw_content.rfind("}") + 1
                raw_content = raw_content[start:end]

            # Clean any remaining newlines or extra spaces
            clean_content = raw_content.replace('\n', ' ').strip()

            # Validate JSON before returning
            parsed = json.loads(clean_content)  # Test if it's valid JSON

            if parsed.get("co2e_kg") is not None:
                values.append(float(parsed.get("co2e_kg")))
                last_valid_response = parsed

        except Exception as e:
            print(f"Error processing response: {str(e)}")
            continue

    if not values:
        return json.dumps({
            "co2e_kg": None,
            "explanation": "Error: no valid response provided by the model."
        })

    # Apply distance-based weighting from the median to
    # mitigate the impact of outliers in LLM estimations
    weights = 1.0 / (np.abs(np.array(values) - np.median(np.array(values))) + 1e-5)
    avg_value = np.average(np.array(values), weights=weights)
    last_valid_response["co2e_kg"] = avg_value

    return json.dumps(last_valid_response)


def main(num_rows, input_file, output_file, model):
    # Load data from the json
    products = []

    # split metadata file into several parts due to the size of the original file
    with open(input_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_rows:
                # this is due to the limits of the API
                break
            product = json.loads(line.strip())
            products.append(product)

    BATCH_SIZE = 5
    with open(output_file, "w", encoding="utf-8") as out:
        out.write("[\n")
        first_item = True

        # Parallel execution
        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            futures = {executor.submit(estimate_co2_for_product, p, model): i for i, p in enumerate(products)}
            results_buffer = [None] * len(products)
            batch_indices = []

            for future in as_completed(futures):
                idx = futures[future]
                batch_indices.append(idx)
                product = products[idx]

                try:
                    llm_answer = future.result()
                    answer_data = json.loads(llm_answer)

                    result = {
                        "title": product.get("title"),
                        "parent_asin": product.get("parent_asin"),
                        "co2e_kg": answer_data.get("co2e_kg"),
                        "source": answer_data.get("source"),
                        "explanation": answer_data.get("explanation")
                    }

                except Exception as e:
                    print(f"Error processing product {product.get('title')[:60]}: {e}")
                    result = {
                        "title": product.get("title"),
                        "co2e_kg": None,
                        "explanation": f"Error processing response: {llm_answer}"
                    }

                results_buffer[idx] = result

                # Write results to JSON file
                if len(batch_indices) >= BATCH_SIZE:
                    for i in sorted(batch_indices):
                        if not first_item:
                            out.write(",\n")
                        out.write(json.dumps(results_buffer[i], ensure_ascii=False))
                        first_item = False
                    batch_indices = []  # Reset batch

            # Write residual results of the last batch
            for i in sorted(batch_indices):
                if not first_item:
                    out.write(",\n")
                out.write(json.dumps(results_buffer[i], ensure_ascii=False))
                first_item = False

        out.write("\n]")


# =============================================
# GPT-o3-mini (Real-time Inference)
# =============================================
model = "o3-mini"
#base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
client = OpenAI(api_key=getenv("OPENAI_API_KEY"))

main(
    num_rows=100,
    input_file="few_metadata/metadata.jsonl",
    output_file="few_results/gpt_o3_mini_results.json",
    model=model
)
