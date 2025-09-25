"""
Test script to verify name extraction logic
"""
import re

def is_likely_name(text):
    """Test version of name detection"""
    if not text or not text.strip():
        return False
        
    words = text.strip().split()
    text_clean = text.strip()
    text_lower = text_clean.lower()
    
    # Basic validation
    if (len(words) < 1 or len(words) > 4 or
        any(char.isdigit() for char in text_clean) or
        len(text_clean) > 50):
        return False
    
    # Company indicators
    company_indicators = {
        'company', 'corp', 'corporation', 'inc', 'incorporated', 'ltd', 'limited', 'llc',
        'solutions', 'systems', 'technologies', 'services', 'consulting', 'software',
        'tech', 'it', 'information', 'technology', 'pvt', 'private', 'public',
        'enterprise', 'enterprises', 'group', 'international', 'global', 'india'
    }
    
    # Check if any word is a company indicator
    for word in words:
        if word.lower() in company_indicators:
            print(f"❌ '{text}' rejected - contains company indicator: {word.lower()}")
            return False
    
    # Pattern validation for names
    name_patterns = [
        r'^[A-Z][a-z]+(?:\s[A-Z][a-z]+){0,3}$',  # Title case names
        r'^[A-Z][a-z]+(?:\s[A-Z]\.){0,2}(?:\s[A-Z][a-z]+)?$'  # Names with middle initials
    ]
    
    if not any(re.match(pattern, text_clean) for pattern in name_patterns):
        print(f"❌ '{text}' rejected - doesn't match name patterns")
        return False
    
    # Check for all caps (usually company names)
    if text.isupper() and len(words) > 1:
        print(f"❌ '{text}' rejected - all caps (likely company name)")
        return False
    
    print(f"✅ '{text}' accepted as likely name")
    return True

# Test cases
test_cases = [
    "John Smith",
    "CODTECH IT SOLUTIONS", 
    "API",
    "Sarah Johnson",
    "TECH SOLUTIONS",
    "Michael Brown",
    "SOFTWARE SYSTEMS",
    "Emma Davis",
    "IT SERVICES",
    "David Wilson"
]

print("Testing name extraction logic:")
print("=" * 50)

for test_case in test_cases:
    print(f"\nTesting: '{test_case}'")
    is_likely_name(test_case)