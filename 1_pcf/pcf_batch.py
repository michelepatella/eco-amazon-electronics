import json
from os import getenv
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = "gemini-2.5-flash"
BATCH_INPUT_FILE = "gemini_batch_input.jsonl"

# Initialize Google GenAI client
client = genai.Client(api_key=getenv("GEMINI_API_KEY"))


def estimate_co2_for_product(product_data, llm_model=MODEL):
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

    # Batch request for Gemini using native genai format
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


def main(num_rows):
    # Load data from the jsonl
    products = []

    # split metadata file into several parts due to the size of the original file
    with open("metadata_split/meta_3.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_rows:
                # this is due to the limits of the API
                break
            product = json.loads(line.strip())
            products.append(product)

    # Create the JSONL file for the batch
    print(f"Creating batch input file: {BATCH_INPUT_FILE}...")
    with open(BATCH_INPUT_FILE, "w", encoding="utf-8") as f:
        for p in products:
            f.write(estimate_co2_for_product(p) + "\n")

    # Upload the file to Gemini API
    print("Uploading file to Gemini API...")
    uploaded_file = client.files.upload(
        file=BATCH_INPUT_FILE,
        config=types.UploadFileConfig(mime_type='application/jsonl')
    )

    # Create the Batch Job
    print(f"File uploaded (ID: {uploaded_file.name}). Creating Batch Job...")
    batch_job = client.batches.create(
        model=f"models/{MODEL}",
        src=uploaded_file.name
    )

    print("-" * 30)
    print(f"BATCH JOB CREATED SUCCESSFULLY!")
    print(f"Batch ID: {batch_job.name}")
    print(f"Status: {batch_job.state}")
    print("-" * 30)


if __name__ == "__main__":
    main(num_rows=10000)