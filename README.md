# AI-Powered FAQ Chatbot - CodeAlpha Internship Project

This is a full-stack web application that functions as a hybrid chatbot. It first attempts to answer user questions from a pre-defined set of FAQs using NLP techniques. If it cannot find a confident answer, it escalates the query to a powerful Large Language Model (Google's Gemini AI) to provide a general knowledge response.

---

## Features

- **Hybrid AI Model:**
  - **Fast FAQ Retrieval:** Uses TF-IDF and Cosine Similarity to instantly find the best match for common questions.
  - **General Knowledge:** Seamlessly calls the Gemini 1.5 Flash API for questions outside its specific knowledge base.
- **Modern Chat Interface:** A clean, responsive chat UI built with Tailwind CSS.
- **NLP Preprocessing:** Utilizes NLTK for text cleaning, tokenization, and stopword removal to improve matching accuracy.

---

## Tech Stack

- **Backend:** Python, Flask
- **Frontend:** HTML, Tailwind CSS, JavaScript
- **NLP & Matching:** `NLTK`, `scikit-learn`
- **Generative AI:** `google-generativeai` (for the Gemini API)

---

## Setup & Installation

1.  **Prerequisites:**
    - Python 3.8+
    - An API Key from [Google AI Studio](https://aistudio.google.com/)

2.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/CodeAlpha_AI_Chatbot.git](https://github.com/your-username/CodeAlpha_AI_Chatbot.git)
    cd CodeAlpha_AI_Chatbot
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK data:**
    Open a Python shell (`python`) and run:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

5.  **Configure API Key:**
    Open `app.py` and replace `"YOUR_API_KEY"` with your actual Google AI API key.

6.  **Run the application:**
    ```bash
    python app.py
    ```
    The chatbot will be live at `http://127.0.0.1:5000`.
