import re
import os
import logging
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------
# NLTK setup (download at runtime)
# ----------------------
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

stop_words = set(stopwords.words("english"))

# ----------------------
# Gemini AI setup
# ----------------------
try:
    api_key = os.environ.get("GOOGLE_API_KEY")  # <-- correct: environment variable name
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    logging.info("Gemini AI model configured successfully.")
except Exception as e:
    logging.error(f"FATAL: Error configuring Gemini AI: {e}")
    model = None

# ----------------------
# FAQ setup
# ----------------------
faqs = {
    "What are your business hours?": "Our business hours are from 9 AM to 6 PM, Monday to Friday.",
    "How can I track my order?": "You can track your order by visiting the 'Track Order' page on our website and entering your order ID.",
    "What is your return policy?": "We offer a 30-day return policy for all items in their original condition.",
    "Do you ship internationally?": "Yes, we ship to most countries worldwide.",
    "How do I contact customer support?": "You can contact our customer support team via email at support@example.com.",
    "What payment methods do you accept?": "We accept all major credit cards, PayPal, and Apple Pay."
}

# ----------------------
# Preprocessing & TF-IDF
# ----------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    return " ".join([word for word in tokens if word not in stop_words])

questions = list(faqs.keys())
processed_questions = [preprocess(q) for q in questions]
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(processed_questions)

# ----------------------
# Answering functions
# ----------------------
def get_general_answer(user_question):
    if not model:
        return "I'm sorry, my connection to the AI model is not configured. I can only answer our standard FAQs."
    try:
        response = model.generate_content(
            f"Please provide a concise, friendly answer to the question: {user_question}",
            generation_config={"temperature": 0.7}
        )
        logging.info(f"RAW GEMINI RESPONSE: {response}")
        if response.parts:
            return response.text
        else:
            logging.error(f"GEMINI call succeeded but returned no content. Feedback: {response.prompt_feedback}")
            return "I'm sorry, I couldn't generate a response for that."
    except Exception as e:
        logging.error(f"CRITICAL: Error calling Gemini API: {e}", exc_info=True)
        return "I'm sorry, I'm having trouble connecting to my knowledge base right now."

def get_best_answer(user_question):
    processed_user_question = preprocess(user_question)
    user_question_vector = vectorizer.transform([processed_user_question])
    similarities = cosine_similarity(user_question_vector, question_vectors)
    most_similar_index = similarities.argmax()
    
    if similarities[0, most_similar_index] > 0.6:
        return faqs[questions[most_similar_index]]
    
    return get_general_answer(user_question)

# ----------------------
# Flask App
# ----------------------
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.json.get("question", "")
    if not user_question:
        return jsonify({"answer": "Please ask a question."})
    answer = get_best_answer(user_question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
