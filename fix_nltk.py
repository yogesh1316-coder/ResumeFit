#!/usr/bin/env python3
"""
Quick fix script to download compatible NLTK data
"""

import nltk
import sys

def download_nltk_data():
    """Download NLTK data packages with better error handling"""
    packages = [
        'punkt',
        'punkt_tab',
        'stopwords', 
        'vader_lexicon',
        'averaged_perceptron_tagger',
        'maxent_ne_chunker',
        'words'
    ]
    
    print("Downloading NLTK data packages...")
    for package in packages:
        try:
            print(f"Downloading {package}...")
            nltk.download(package, quiet=True)
            print(f"✓ {package} downloaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not download {package}: {e}")
    
    print("\nTesting NLTK functionality...")
    
    # Test tokenization
    try:
        from nltk.tokenize import sent_tokenize, word_tokenize
        test_text = "Hello world. This is a test."
        sentences = sent_tokenize(test_text)
        words = word_tokenize(test_text)
        print(f"✓ Tokenization working: {len(sentences)} sentences, {len(words)} words")
    except Exception as e:
        print(f"✗ Tokenization error: {e}")
    
    # Test sentiment analysis
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores("This is a great resume!")
        print(f"✓ Sentiment analysis working: {scores}")
    except Exception as e:
        print(f"✗ Sentiment analysis error: {e}")
    
    # Test stopwords
    try:
        from nltk.corpus import stopwords
        stop_words = stopwords.words('english')
        print(f"✓ Stopwords loaded: {len(stop_words)} words")
    except Exception as e:
        print(f"✗ Stopwords error: {e}")
    
    print("\nNLTK setup complete!")

if __name__ == "__main__":
    download_nltk_data()