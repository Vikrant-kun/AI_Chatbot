#!/usr/bin/env bash
# Exit on error to make debugging easier
set -o errexit

# Step 1: Install all the Python packages
pip install -r requirements.txt

# Step 2: Run a small Python script to download the necessary NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---
### **Step 2: Commit and Push This Final Change**

Save the new `build.sh` file, then go to your terminal and run these commands to update your GitHub repository.

```bash
git add build.sh
git commit -m "Update build script for NLTK data"
git push origin main

