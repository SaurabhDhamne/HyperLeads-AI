from flask import Flask, request, jsonify
import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

# -----------------------------
# ENV & GEMINI CONFIG
# -----------------------------
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

app = Flask(__name__)

# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "AI service running"})


# -----------------------------
# AI LEAD SCORING 
# -----------------------------
def score_lead_with_ai(industry, requirement):
    prompt = f"""
You are a B2B lead scoring AI.

Business Target Requirement:
{requirement}

Lead Industry:
{industry}

Score this lead from 0 to 100 based on relevance.

Return ONLY JSON in this format:
{{
  "lead_score": 75,
  "reason": "Short explanation"
}}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        if not response or not response.text:
            raise ValueError("Empty AI response")

        parsed = json.loads(response.text)

        return {
            "lead_score": int(parsed.get("lead_score", 0)),
            "reason": parsed.get("reason", "")
        }

    except Exception as e:
        print("AI SCORING ERROR:", e)
        return {
            "lead_score": 0,
            "reason": "AI service failed"
        }


@app.route("/score", methods=["POST"])
def score_lead():
    data = request.json or {}

    industry = data.get("industry", "")
    requirement = data.get("requirement", "")

    if not industry and not requirement:
        return jsonify({
            "lead_score": 0,
            "reason": "Insufficient data"
        })

    result = score_lead_with_ai(industry, requirement)

    return jsonify(result)


# -----------------------------
# EMAIL GENERATION
# -----------------------------
def generate_email_with_gemini(company, industry, requirement):
    prompt = f"""
Write a short B2B cold email.

Company: {company}
Industry: {industry}
Requirement: {requirement}

Tone: Professional, friendly, non-pushy.
Length: 40-60 words.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )

        if not response or not response.text:
            return None

        return response.text.strip()

    except Exception as e:
        print("EMAIL GENERATION ERROR:", e)
        return None


@app.route("/generate-email", methods=["POST"])
def generate_email_api():
    data = request.json or {}

    email_text = generate_email_with_gemini(
        company=data.get("company_name", ""),
        industry=data.get("industry", ""),
        requirement=data.get("requirement", "")
    )

    if not email_text:
        return jsonify({
            "error": "AI service unavailable"
        }), 503

    return jsonify({"email": email_text})


# -----------------------------
# LOCAL RUN
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
