#!/usr/bin/env python3
"""
Debug script to test the main application functions
"""

import sys
import traceback

def test_imports():
    """Test all imports"""
    try:
        from flask import Flask, render_template, request, jsonify
        import pdfplumber
        import spacy
        import re
        from datetime import datetime
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        traceback.print_exc()
        return False

def test_spacy_model():
    """Test SpaCy model loading"""
    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
        print("✓ SpaCy model loaded successfully")
        return True
    except Exception as e:
        print(f"✗ SpaCy model error: {e}")
        traceback.print_exc()
        return False

def test_ml_training():
    """Test ML model training"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        
        # Training data
        training_texts = [
            "python machine learning data science tensorflow pandas numpy",
            "java spring boot microservices sql database",
            "javascript react nodejs frontend development",
            "marketing digital marketing social media analytics",
            "sales customer relationship management crm",
            "finance accounting financial analysis excel",
            "design ui ux graphic design photoshop",
            "project management agile scrum leadership"
        ]
        
        training_labels = [
            "Data Scientist",
            "Software Engineer", 
            "Frontend Developer",
            "Marketing Specialist",
            "Sales Representative",
            "Financial Analyst",
            "Designer",
            "Project Manager"
        ]
        
        # Initialize and train model
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X_train = vectorizer.fit_transform(training_texts)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, training_labels)
        
        print("✓ ML model training successful")
        return True
    except Exception as e:
        print(f"✗ ML training error: {e}")
        traceback.print_exc()
        return False

def test_app_initialization():
    """Test Flask app initialization"""
    try:
        from app import app
        print("✓ Flask app imported successfully")
        return True
    except Exception as e:
        print(f"✗ Flask app initialization error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=== DEBUG TEST RESULTS ===")
    
    tests = [
        ("Import Test", test_imports),
        ("SpaCy Model Test", test_spacy_model),
        ("ML Training Test", test_ml_training),
        ("App Initialization Test", test_app_initialization)
    ]
    
    failed_tests = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if not test_func():
            failed_tests.append(test_name)
    
    print(f"\n=== SUMMARY ===")
    if failed_tests:
        print(f"✗ {len(failed_tests)} test(s) failed:")
        for test in failed_tests:
            print(f"  - {test}")
    else:
        print("✓ All tests passed!")

if __name__ == "__main__":
    main()