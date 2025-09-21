import re
import os
import logging
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# NEW, SECURE CODE
import os # Make sure to have 'import os' at the top of your file

try:
    # Gets the API key from the hosting environment's secrets
    api_key = os.environ.get('AIzaSyDGjOKqAyVTVzEdj2lwPMNugj6r9JeGV94')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    print("Gemini AI model configured successfully.")
except Exception as e:
    print(f"FATAL: Error configuring Gemini AI: {e}")
    model = None
faqs = {
    "What are your business hours?": "Our business hours are from 9 AM to 6 PM, Monday to Friday.",
    "How can I track my order?": "You can track your order by visiting the 'Track Order' page on our website and entering your order ID.",
    "What is your return policy?": "We offer a 30-day return policy for all items in their original condition.",
    "Do you ship internationally?": "Yes, we ship to most countries worldwide.",
    "How do I contact customer support?": "You can contact our customer support team via email at support@example.com.",
    "What payment methods do you accept?": "We accept all major credit cards, PayPal, and Apple Pay."
}

stop_words = set(stopwords.words('english'))
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    return " ".join([word for word in tokens if word not in stop_words])

questions = list(faqs.keys())
processed_questions = [preprocess(q) for q in questions]
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(processed_questions)

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
            return "I'm sorry, I couldn't generate a response for that. It might be due to a safety filter."

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

app = Flask(__name__)
@app.route("/")
def index(): return render_template("index.html")
@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.json.get("question", "")
    if not user_question: return jsonify({"answer": "Please ask a question."})
    answer = get_best_answer(user_question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)

