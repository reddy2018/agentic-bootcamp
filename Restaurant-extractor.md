Assignment B - Submission Form
In this assignment, you will build a Restaurant Information Extractor. The goal is to convert unstructured restaurant reviews into clean, validated JSON outputs. This will help you practice working with schemas, structured outputs, and retry logic.

🎯 Objective
• Convert restaurant reviews into JSON following a defined schema.

• Practice schema-first prompting and validation with LangChain.

• Implement retry logic when JSON output is invalid.

📥 Input
• A single-paragraph restaurant review written in plain text.

📤 Expected JSON Schema
{

"name": string → restaurant name

"cuisine": string → e.g., Indian, Italian, Japanese

"city": string → empty "" if not mentioned

"rating": number (0.0–5.0) → allow decimals

"price_range": "low" | "mid" | "high"

}

📏 Rules
• Output must be ONLY valid JSON (no markdown fences, no prose).

• Do not fabricate facts. If data is missing, leave it empty or null as specified.

• Rating must be a number between 0.0–5.0.

• Price range must strictly be one of: low | mid | high.


▶️ How to Run
Run with a custom review:

python restaurant_extractor.py --review "Loved dinner at Trattoria Bella in central Rome, pricey but amazing pasta. 4.5/5"

📦 Deliverables
• A working Python script or notebook that prints validated JSON outputs.

• At least 3 test runs with different reviews (already provided in tests/test_reviews.json).

• A short RESULTS.md file showing the JSON outputs.

📝 Example Reviews to Try
1. "Loved dinner at Trattoria Bella in central Rome… a bit pricey… 4.5/5."

2. "Tiny sushi bar in Tokyo… omakase only… amazing tuna… pricey."

3. "Budget-friendly dosa place in Bangalore… quick service… authentic taste."

💡 Tips for Success
• Always validate JSON output using Pydantic or a parser.

• Use retry logic to handle invalid JSON or schema errors.

• Keep temperature low (≈0.2) for stable structured output.

• Ensure outputs do not include extra text or formatting.