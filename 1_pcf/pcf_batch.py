import json
import os
import re
import time
from os import getenv
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Initialization
load_dotenv()
MODEL = "gemini-2.5-flash"
client = genai.Client(api_key=getenv("GEMINI_API_KEY"))

# Time (in seconds) to wait before
# checking if any new batch request is succeeded
CHECK_INTERVAL = 60


def save_results(input_file, output_file):
    # Keep track of the name-ASIN products mapping
    # and of the original order they appear
    with open(input_file, 'r') as f:
        input_data = [json.loads(l) for l in f if l.strip()]
    products_mapping = {d['parent_asin']: d.get('title', 'N/A') for d in input_data}
    products_order = [d['parent_asin'] for d in input_data]

    # Wait for all the batches being processed
    processed_batches = set()
    while True:
        for job in client.batches.list():
            if job.state.name == 'JOB_STATE_SUCCEEDED' and job.name not in processed_batches:
                print(f"\nSaving batch response ({job.name})...")
                content = client.files.download(file=job.dest.file_name).decode('utf-8')

                # Load existing results
                all_results = {}
                if os.path.exists(output_file):
                    with open(output_file, 'r') as f:
                        all_results = {item['parent_asin']: item for item in json.load(f)}

                # Parse the response
                for line in content.splitlines():
                    node = json.loads(line)
                    asin, raw_txt = node['custom_id'], node['response']['candidates'][0]['content']['parts'][0]['text']
                    clean_txt = re.sub(r'```json\s*|```', '', raw_txt).strip().replace('\n', ' ')
                    if clean_txt.count('{') > clean_txt.count('}'): clean_txt += '}'
                    try:
                        pred = json.loads(re.search(r'\{.*\}', clean_txt).group())
                    except Exception:
                        co2_match = re.search(r'"co2e_kg":\s*([\d\.]+)', clean_txt)
                        expl_match = re.search(r'"explanation":\s*"(.*?)"', clean_txt)
                        pred = {
                            "co2e_kg": float(co2_match.group(1)) if co2_match else None,
                            "source": "estimation",
                            "explanation": expl_match.group(1) if expl_match else "Parsing error"
                        }

                    all_results[asin] = {
                        "product_name": products_mapping.get(asin, "N/A"),
                        "parent_asin": asin,
                        "co2e_kg": pred.get("co2e_kg"),
                        "source": pred.get("source"),
                        "explanation": pred.get("explanation", "N/A")
                    }

                # Save the response
                final_list = [all_results[a] for a in products_order if a in all_results]
                with open(output_file, "w", encoding="utf-8") as out:
                    json.dump(final_list, out, indent=4, ensure_ascii=False)
                processed_batches.add(job.name)
                print(f"\nBatch response saved ({len(final_list)}/{len(products_order)})")

        # Check whether everything has been saved
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                if len(json.load(f)) >= len(products_order):
                    break

        time.sleep(CHECK_INTERVAL)


def estimate_co2_for_batch_of_products(product_data, llm_model=MODEL):
    """
    Example of LLM prompting to predict the CO2eq of batch of products
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

    # Batch request for the API
    batch_request = {
        "custom_id": product_data.get("parent_asin", "unknown"),
        "request": {
            "model": f"models/{llm_model}",
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generation_config": {
                "temperature": 0.0,
                "response_mime_type": "application/json"
            }
        }
    }
    return json.dumps(batch_request, ensure_ascii=False)


def main(start_row, num_rows, input_file, batch_file_name, output_file):
    # Extract products to insert into the batch request
    products = []
    with open(input_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            # Extract the products within the given boundaries
            if i < start_row:
                continue
            if len(products) >= num_rows:
                break
            product = json.loads(line.strip())
            products.append(product)

    # Create a JSON file with batch requests for that products
    print("-" * 15 + "(1)" + "-" * 15)
    print("Creating batch input file...")
    with open(batch_file_name, "w", encoding="utf-8") as f:
        for p in products:
            f.write(estimate_co2_for_batch_of_products(p) + "\n")
    print(f"Batch input file created: {batch_file_name}")
    print("-" * 33)

    # Upload the file to API
    print("-" * 15 + "(2)" + "-" * 15)
    print("Uploading batch input file to API...")
    uploaded_file = client.files.upload(
        file=batch_file_name,
        config=types.UploadFileConfig(mime_type='application/jsonl')
    )
    print(f"Batch input file uploaded to API: {uploaded_file.name} (ID)")
    print("-" * 33)

    # Create the batch job
    print("-" * 15 + "(3)" + "-" * 15)
    print("Creating batch job...")
    batch_job = client.batches.create(
        model=f"models/{MODEL}",
        src=uploaded_file.name
    )
    print("Batch job created:")
    print(f"\n- ID: {batch_job.name}")
    print(f"\n- Status: {batch_job.state}")
    print("-" * 33)

    # Wait for request processing completion
    # and save results
    print("-" * 15 + "(4)" + "-" * 15)
    save_results(input_file, output_file)
    print("-" * 33)


if __name__ == "__main__":
    main(start_row=..., num_rows=..., input_file=..., batch_file_name=..., output_file=...)