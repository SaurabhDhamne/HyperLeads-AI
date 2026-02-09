from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
import google.generativeai as genai

# api key
load_dotenv()  # loads .env into environment variables

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

# cold email generator 
def generate_email_with_gemini(company, industry, keywords):
    prompt = f"""
    You are a B2B sales assistant.

    Write a short, professional cold email for a company named {company}
    operating in the {industry} space.

    Mention these keywords naturally: {", ".join(keywords)}.
    Tone: professional, friendly, non-pushy.
    Length: 80-120 words.
    """

    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    response = model.generate_content(prompt)

    return response.text.strip()

# keywords for NLP to score
import spacy

nlp = spacy.load("en_core_web_md")

TARGET_KEYWORDS = {
    "saas": ["software", "platform", "cloud", "subscription"],
    "ai": ["ai", "automation", "nlp", "model", "learning"],
    "cloud": ["cloud", "aws", "azure", "infrastructure"]
}


# lemmatization
def extract_keywords(text):
    doc = nlp(text.lower())
    keywords = set()

    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
            keywords.add(token.lemma_.replace("-", " "))

    return keywords


app = Flask(__name__)
# route for email generator
@app.route("/generate-email", methods=["POST"])
def generate_email_api():
    data = request.json

    email_text = generate_email_with_gemini(
        company=data["company_name"],
        industry=data["industry"],
        keywords=data.get("keywords", [])
    )

    if not email_text:
        return jsonify({
            "error": "LLM quota exceeded or service unavailable"
        }), 503

    return jsonify({"email": email_text})



@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "AI service running"})

@app.route("/score", methods=["POST"])
def score_lead():
    data = request.json

    industry = data.get("industry", "").lower()
    website_text = data.get("website_text", "")

    combined_text = f"{industry} {website_text}"
    extracted_keywords = extract_keywords(combined_text)
    print("Combined text:", combined_text)
    print("Extracted keywords:", extracted_keywords) #this is for debgging


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


if __name__ == "__main__":
    app.run(port=5001, debug=True)
