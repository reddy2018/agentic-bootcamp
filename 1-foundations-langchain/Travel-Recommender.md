Assignment A - Travel Recommender Submission Form
In this assignment, you will build a simple Travel Recommender Agent using LangChain and Google Gemini. The goal is to practice working with prompt templates and passing variables dynamically.

🎯 Objective
• Build a Travel Recommender that adapts its output based on user input variables.

• Practice using ChatPromptTemplate in LangChain.

• Run multiple test cases to see how outputs vary.

📥 Inputs (Variables)
• city — Destination city

• days — Trip duration (integer)

• budget — Budget level (low, moderate, high)

• traveler_type — Type of traveler (solo, family, adventure, luxury)

📤 Expected Output Format
Your output should be in plain text with 3 parts:

1. Opener → A 1–2 line intro about the city.

2. Itinerary → Day-by-day plan matching the number of days.

3. Tips → 2–3 short suggestions tailored to budget and traveler type.

▶️ How to Run
Run with specific inputs:

python travel_recommender.py --city "Goa" --days 3 --budget moderate --traveler_type family