import json
from openai import OpenAI
from together import Together
import openai



API_KEY = "api_key"
MODEL = "model/model_version" 
BASE_URL = "base_url_provider_model"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

def estimate_co2_for_product(product_data, llm_model=MODEL):
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

    response = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    print(response)
    
    if not response or not response.choices:
        print("Error: the model response is not formatted or empty")
        return json.dumps({
            "co2e_kg": None,
            "explanation": "Error: no response provided by the model."
        })

    try:
        raw_content = response.choices[0].message.content.strip()
        print(f"\nProcessing product: {product_data.get('title', 'Unknown')[:60]}...")
        
        # Extract JSON object if there's additional text
        if "{" in raw_content and "}" in raw_content:
            start = raw_content.rfind("{")  # Get the last JSON object
            end = raw_content.rfind("}") + 1
            raw_content = raw_content[start:end]
            
        # Clean any remaining newlines or extra spaces
        raw_content = raw_content.replace('\n', ' ').strip()
        
        # Validate JSON before returning
        json.loads(raw_content)  # Test if it's valid JSON
        return raw_content
        
    except Exception as e:
        print(f"Error processing response: {str(e)}")
        return json.dumps({
            "co2e_kg": None,
            "explanation": f"Error processing response: {str(e)}"
        })

def main(num_rows):
    # Load data from the json
    products = []

    # split metadata file into several parts due to the size of the original file
    with open("dataset/metadata_split/meta_1.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_rows:
                # this is due to the limits of the API
                break
            product = json.loads(line.strip())
            products.append(product)

    results = []

    for product in products:
        try:
            llm_answer = estimate_co2_for_product(product)

            if isinstance(llm_answer, str):
                if "```json" in llm_answer:
                    llm_answer = llm_answer.split("```json")[1].split("```")[0].strip()
                elif "```" in llm_answer:
                    llm_answer = llm_answer.split("```")[1].strip()

                if "{" in llm_answer and "}" in llm_answer:
                    start = llm_answer.rfind("{")
                    end = llm_answer.rfind("}") + 1
                    llm_answer = llm_answer[start:end]

                llm_answer = llm_answer.strip()

            answer_data = json.loads(llm_answer)

            results.append({
                "product_name": product.get("title", "Senza nome"),
                "parent_asin": product.get("parent_asin", "Senza ASIN"),
                "co2e_kg": answer_data.get("co2e_kg"),
                "source": answer_data.get("source"),
                "explanation": answer_data.get("explanation")
            })

        except Exception as e:
            print(f"Error processing product {product.get('title', 'Unknown')[:60]}: {e}")
            results.append({
                "product_name": product.get("title", "Senza nome"),
                "co2e_kg": None,
                "explanation": f"Error processing response: {llm_answer}"
            })
    
    # Save as json
    with open("metadata.json", "a", encoding="utf-8") as out:
        json.dump(results, out, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main(num_rows=100)