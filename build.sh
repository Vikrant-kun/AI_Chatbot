#!/usr/bin/env bash
# Exit on error to make debugging easier
set -o errexit

# Step 1: Install all the Python packages
pip install -r requirements.txt

# Step 2: Run a small Python script to download the necessary NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"