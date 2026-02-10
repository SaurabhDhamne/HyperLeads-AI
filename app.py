from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv

# ✅ UPDATED Gemini import (new SDK style)
from google import genai

import spacy

# -----------------------------
# ENV & GEMINI CONFIG
# -----------------------------
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# -----------------------------
# SPACY (LAZY LOAD – VERY IMPORTANT)
# -----------------------------
nlp = None

def get_nlp():
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_md")
        except Exception:
            # fallback so service never crashes
            nlp = spacy.blank("en")
    return nlp

# -----------------------------
# TARGET KEYWORDS
# -----------------------------
TARGET_KEYWORDS = {
    "saas": ["software", "platform", "cloud", "subscription"],
    "ai": ["ai", "automation", "nlp", "model", "learning"],
    "cloud": ["cloud", "aws", "azure", "infrastructure"]
}

# -----------------------------
# NLP KEYWORD EXTRACTION
# -----------------------------
def extract_keywords(text: str):
    doc = get_nlp()(text.lower())
    keywords = set()

    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
            lemma = token.lemma_.replace("-", " ").strip()
            if lemma:
                keywords.add(lemma)

    return keywords

# -----------------------------
# GEMINI EMAIL GENERATOR
# -----------------------------
def generate_email_with_gemini(company, industry, keywords):
    prompt = f"""
You are a B2B sales assistant.

Write a short, professional cold email for a company named {company}
operating in the {industry} space.

Mention these keywords naturally: {", ".join(keywords)}.
Tone: professional, friendly, non-pushy.
Length: 80-120 words.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )

    return response.text.strip() if response and response.text else None

# -----------------------------
# FLASK APP
# -----------------------------
app = Flask(__name__)

# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "AI service running"})

# -----------------------------
# EMAIL GENERATION API
# -----------------------------
@app.route("/generate-email", methods=["POST"])
def generate_email_api():
    data = request.json or {}

    email_text = generate_email_with_gemini(
        company=data.get("company_name", ""),
        industry=data.get("industry", ""),
        keywords=data.get("keywords", [])
    )

    if not email_text:
        return jsonify({
            "error": "LLM quota exceeded or service unavailable"
        }), 503

    return jsonify({"email": email_text})

# -----------------------------
# LEAD SCORING API
# -----------------------------
@app.route("/score", methods=["POST"])
def score_lead():
    data = request.json or {}

    industry = data.get("industry", "").lower()
    website_text = data.get("website_text", "")

    combined_text = f"{industry} {website_text}"
    extracted_keywords = extract_keywords(combined_text)

    print("Combined text:", combined_text)
    print("Extracted keywords:", extracted_keywords)

    score = 20  # base score
    reasons = []

    for domain, keywords in TARGET_KEYWORDS.items():
        match_count = len(extracted_keywords.intersection(set(keywords)))
        if match_count > 0:
            score += match_count * 15
            reasons.append(f"{domain} keyword match ({match_count})")

    score = min(score, 100)

    return jsonify({
        "lead_score": score,
        "keywords_found": list(extracted_keywords),
        "reason": ", ".join(reasons) or "Low relevance"
    })

# -----------------------------
# LOCAL RUN
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
