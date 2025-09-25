#!/bin/bash
# Render build script for ResumeFit Flask App

echo "Starting build process..."

# Install Python dependencies
pip install -r requirements.txt

# Download spaCy model explicitly
echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Download NLTK data
echo "Downloading NLTK data..."
python -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True) 
nltk.download('vader_lexicon', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
print('NLTK data downloaded successfully')
"

echo "Build completed successfully!"