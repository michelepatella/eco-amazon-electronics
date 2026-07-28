You are an expert in life cycle analysis (LCA) and CO2e emission calculation for {product_category} products.
You must estimate the CO2e emissions, based on the entire life cycle (cradle to grave), for the following {product_category} product.

Product data: {json.dumps(product_data, ensure_ascii=False)}

INSTRUCTIONS:
1. FIRST, check if there are any official carbon footprint reports or environmental product declarations (EPD) from the manufacturer for this specific product. If found, use these official values as your primary source.

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
6. Clearly state if you’re using manufacturer data or estimation

Reply ONLY with a JSON object containing these exact fields:
{
  "co2e_kg": <number>,
  "source": <if "manufacturer report" or "estimation">,
  "explanation": "<detailed explanation including data source>"
}
Do not include any markdown formatting or additional JSON wrappers.
