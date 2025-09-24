from flask import Flask, render_template, request, jsonify
import spacy
import pdfplumber
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from docx import Document
import os

app = Flask(__name__)
nlp = spacy.load('en_core_web_sm')

# Initialize NLTK components
try:
    # Try to use the data first
    import nltk.data
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('vader_lexicon')
    nltk.data.find('averaged_perceptron_tagger')
    nltk.data.find('maxent_ne_chunker')
    nltk.data.find('words')
except LookupError:
    # Download if not found
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)  # Download new punkt_tab
        nltk.download('stopwords', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('maxent_ne_chunker', quiet=True)
        nltk.download('words', quiet=True)
    except Exception as e:
        print(f"Warning: Could not download NLTK data: {e}")

# Initialize sentiment analyzer with error handling
try:
    sia = SentimentIntensityAnalyzer()
    stop_words = set(stopwords.words('english'))
except Exception as e:
    print(f"Warning: NLTK initialization error: {e}")
    # Fallback: use empty sentiment analyzer
    class FallbackSentimentAnalyzer:
        def polarity_scores(self, text):
            return {'pos': 0.0, 'neg': 0.0, 'neu': 1.0, 'compound': 0.0}
    
    sia = FallbackSentimentAnalyzer()
    stop_words = set()

IGNORE_LIST = [
    "ABOUT ME", "OBJECTIVE", "SUMMARY", "RESUME", "SKILLS", "EDUCATION", "PROJECTS",
    "EXPERIENCE", "CERTIFICATIONS", "INTERESTS", "LANGUAGES", "CONTACT", "PROFILE",
    "PYTHON", "JAVA", "JAVASCRIPT", "HTML", "CSS", "C++", "SQL", "DATA SCIENCE",
    "MSWORD", "POWERPOINT", "TALLY", "MYSQL"
]

# Training data for job role prediction
JOB_TRAINING_DATA = {
    "Data Scientist": [
        "python machine learning data analysis pandas numpy scikit-learn statistics modeling",
        "sql python data mining statistical analysis machine learning algorithms deep learning",
        "data visualization tableau powerbi python r statistics predictive modeling analytics",
        "machine learning tensorflow pytorch neural networks data science python statistics",
        "big data hadoop spark python data analysis machine learning statistical modeling"
    ],
    "Software Engineer": [
        "java spring boot microservices rest api backend development software engineering",
        "python django flask web development backend apis database programming",
        "javascript react nodejs full stack development web applications frontend backend",
        "c++ software development algorithms data structures object oriented programming",
        "software engineering design patterns testing debugging version control git"
    ],
    "Frontend Developer": [
        "html css javascript react angular vue frontend development responsive design",
        "javascript typescript react redux frontend web development user interface",
        "css html sass bootstrap responsive design frontend development user experience",
        "react angular javascript frontend development single page applications ui ux",
        "web development frontend javascript css html responsive design mobile first"
    ],
    "DevOps Engineer": [
        "docker kubernetes aws cloud infrastructure deployment automation ci cd",
        "linux bash scripting automation infrastructure docker containers orchestration",
        "aws azure cloud infrastructure terraform ansible devops automation deployment",
        "jenkins ci cd pipeline automation docker kubernetes infrastructure as code",
        "monitoring logging infrastructure automation cloud platforms devops practices"
    ],
    "Backend Developer": [
        "java spring boot rest api microservices backend development database design",
        "python django flask rest api backend development database postgresql mysql",
        "nodejs express mongodb rest api backend development server side programming",
        "database design sql postgresql mysql backend development api design",
        "microservices architecture backend development rest api database design"
    ],
    "Mobile Developer": [
        "android java kotlin mobile development android studio mobile applications",
        "ios swift xcode mobile development iphone ipad applications app store",
        "react native javascript mobile development cross platform ios android",
        "flutter dart mobile development cross platform android ios applications",
        "mobile development android ios react native flutter cross platform"
    ],
    "Product Manager": [
        "product management roadmap strategy stakeholder management agile scrum",
        "product strategy market research user experience product development lifecycle",
        "agile scrum product owner requirements gathering stakeholder communication",
        "market analysis competitive research product strategy business analysis",
        "product roadmap feature prioritization user stories product development"
    ],
    "Marketing Specialist": [
        "digital marketing seo sem social media marketing content marketing analytics",
        "marketing strategy brand management social media campaigns market research",
        "content marketing copywriting seo social media digital marketing campaigns",
        "marketing analytics google analytics facebook ads digital advertising campaigns",
        "brand marketing social media strategy content creation marketing campaigns"
    ]
}

# Initialize and train the ML model
def create_job_prediction_model():
    texts = []
    labels = []
    
    for job_role, samples in JOB_TRAINING_DATA.items():
        for sample in samples:
            texts.append(sample)
            labels.append(job_role)
    
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, labels)
    
    return vectorizer, model

# Create the model globally
vectorizer, job_prediction_model = create_job_prediction_model()

def predict_job_role(resume_text):
    """Predict job role using ML model based on resume content"""
    try:
        # Clean and prepare text for prediction
        clean_text = re.sub(r'[^\w\s]', ' ', resume_text.lower())
        clean_text = ' '.join(clean_text.split())
        
        # Transform text using the trained vectorizer
        text_vector = vectorizer.transform([clean_text])
        
        # Get prediction and confidence
        predicted_role = job_prediction_model.predict(text_vector)[0]
        confidence_scores = job_prediction_model.predict_proba(text_vector)[0]
        max_confidence = max(confidence_scores)
        
        return predicted_role, int(max_confidence * 100)
    except Exception as e:
        return "General Software Engineer", 75

def analyze_resume_content(resume_text):
    """Analyze resume content and extract key information"""
    text_lower = resume_text.lower()
    
    # Extract skills
    skills = []
    skill_keywords = [
        'python', 'java', 'javascript', 'react', 'angular', 'vue', 'nodejs', 'html', 'css',
        'sql', 'mysql', 'postgresql', 'mongodb', 'docker', 'kubernetes', 'aws', 'azure',
        'machine learning', 'data science', 'tensorflow', 'pytorch', 'pandas', 'numpy',
        'spring boot', 'django', 'flask', 'express', 'git', 'jenkins', 'ci/cd'
    ]
    
    for skill in skill_keywords:
        if skill in text_lower:
            skills.append(skill.title())
    
    # Extract education level with enhanced detection
    education_level = "Not specified"
    
    # Use more precise pattern matching with word boundaries to avoid false matches
    import re
    
    # Check for specific degree patterns (more comprehensive and precise)
    if re.search(r'\b(phd|ph\.d|doctorate|doctoral|ph\s+d|doctor\s+of\s+philosophy)\b', text_lower):
        education_level = "PhD/Doctorate"
    elif re.search(r'\b(master|masters|master\'?s|mtech|m\.tech|m\s+tech|msc|m\.sc|m\s+sc|ms\b|mba|m\.b\.a|m\s+b\s+a|post\s+graduate|postgraduate|pg\b)\b', text_lower):
        education_level = "Master's Degree"
    elif re.search(r'\b(bachelor|bachelors|bachelor\'?s|btech|b\.tech|b\s+tech|bsc|b\.sc|b\s+sc|be\b|b\.e|b\s+e|bca|b\.ca|b\s+ca|bcom|b\.com|b\s+com|undergraduate|graduate\s+degree|ug\b)\b', text_lower):
        education_level = "Bachelor's Degree"
    elif re.search(r'\b(diploma|certificate|polytechnic|advanced\s+diploma|professional\s+certificate)\b', text_lower):
        education_level = "Diploma/Certificate"
    elif re.search(r'\b(12th|intermediate|higher\s+secondary|\+2|hsc|h\.s\.c|senior\s+secondary)\b', text_lower):
        education_level = "Higher Secondary"
    elif re.search(r'\b(10th|matriculation|high\s+school|sslc|s\.s\.l\.c|secondary)\b', text_lower):
        education_level = "High School"
    
    # Debug education detection
    print(f"DEBUG - Detected education level: {education_level}")
    
    # Extract experience level
    experience_level = "Entry Level"
    if any(word in text_lower for word in ['senior', '5+ years', '5 years', '6 years', '7 years', '8 years', '9 years']):
        experience_level = "Senior Level"
    elif any(word in text_lower for word in ['3 years', '4 years', 'mid-level', 'intermediate']):
        experience_level = "Mid Level"
    elif any(word in text_lower for word in ['intern', 'fresher', 'graduate', 'entry']):
        experience_level = "Entry Level"
    
    return {
        'skills': skills[:10],  # Top 10 skills
        'education_level': education_level,
        'experience_level': experience_level
    }

def analyze_resume_with_nltk(resume_text):
    """Enhanced resume analysis using NLTK with error handling"""
    try:
        # Tokenize text with fallback
        try:
            sentences = sent_tokenize(resume_text)
            words = word_tokenize(resume_text.lower())
        except Exception as e:
            print(f"Tokenization error: {e}")
            # Fallback: simple split
            sentences = resume_text.split('.')
            words = resume_text.lower().split()
        
        # Remove stopwords and non-alphabetic tokens
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
        
        # Perform sentiment analysis with error handling
        try:
            sentiment_scores = sia.polarity_scores(resume_text)
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            sentiment_scores = {'pos': 0.0, 'neg': 0.0, 'neu': 1.0, 'compound': 0.0}
        
        # Extract named entities using NLTK with error handling
        nltk_entities = []
        try:
            tokens = word_tokenize(resume_text)
            pos_tags = pos_tag(tokens)
            chunks = ne_chunk(pos_tags)
            
            # Extract entities from chunks
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    entity_text = ' '.join([token for token, pos in chunk.leaves()])
                    nltk_entities.append({
                        'text': entity_text,
                        'label': chunk.label()
                    })
        except Exception as e:
            print(f"Named entity recognition error: {e}")
            nltk_entities = []
        
        # Calculate text statistics
        word_count = len(words)
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Extract key phrases (simple approach using POS tags) with error handling
        key_phrases = []
        try:
            current_phrase = []
            pos_tags = pos_tag(word_tokenize(resume_text))
            
            for word, pos in pos_tags:
                if pos.startswith('NN') or pos.startswith('JJ'):  # Nouns and adjectives
                    current_phrase.append(word)
                else:
                    if len(current_phrase) > 1:
                        phrase = ' '.join(current_phrase)
                        if len(phrase) > 3:  # Only meaningful phrases
                            key_phrases.append(phrase)
                    current_phrase = []
        except Exception as e:
            print(f"Key phrase extraction error: {e}")
            # Fallback: extract common phrases
            key_phrases = []
        
        return {
            'sentiment': {
                'positive': sentiment_scores['pos'],
                'negative': sentiment_scores['neg'],
                'neutral': sentiment_scores['neu'],
                'compound': sentiment_scores['compound']
            },
            'entities': nltk_entities,
            'statistics': {
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_sentence_length': round(avg_sentence_length, 2)
            },
            'key_phrases': key_phrases[:10],  # Top 10 key phrases
            'filtered_words': filtered_words[:50]  # Top 50 meaningful words
        }
    
    except Exception as e:
        print(f"NLTK analysis error: {e}")
        # Return safe fallback data
        return {
            'sentiment': {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': 0.0
            },
            'entities': [],
            'statistics': {
                'word_count': len(resume_text.split()),
                'sentence_count': len(resume_text.split('.')),
                'avg_sentence_length': 0
            },
            'key_phrases': [],
            'filtered_words': []
        }

def extract_text(file_storage):
    filename = file_storage.filename.lower()
    if filename.endswith('.pdf'):
        with pdfplumber.open(file_storage) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text() or ''
        return text
    elif filename.endswith('.docx'):
        try:
            doc = Document(file_storage)
            text = ""
            # Extract text from paragraphs
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text += cell.text + " "
                text += "\n"
            
            return text
        except Exception as e:
            print(f"Error extracting text from DOCX: {e}")
            return ""
    else:
        try:
            return file_storage.read().decode('utf-8')
        except UnicodeDecodeError:
            file_storage.seek(0)
            return file_storage.read().decode('latin-1')

def is_likely_name(text):
    """Ultra-accurate name detection with comprehensive validation"""
    if not text or not text.strip():
        return False
        
    words = text.strip().split()
    text_clean = text.strip()
    text_upper = text_clean.upper()
    
    # Basic validation
    if (len(words) < 1 or len(words) > 4 or
        text_upper in IGNORE_LIST or
        any(char.isdigit() for char in text_clean)):
        return False
    
    # Comprehensive keyword exclusions
    exclusion_keywords = [
        # Contact and tech terms
        'phone', 'email', 'mobile', 'contact', 'linkedin', 'github', 'website', 'www', 'http', 'tel',
        # Resume sections
        'objective', 'summary', 'about', 'skills', 'education', 'experience', 'projects', 'resume', 'cv',
        'certifications', 'achievements', 'interests', 'languages', 'profile', 'career', 'personal',
        # Address terms
        'address', 'street', 'road', 'avenue', 'lane', 'city', 'state', 'country', 'pin', 'zip', 'postal',
        # Common false positive patterns
        'india', 'indian', 'engineering', 'college', 'university', 'institute', 'technology', 'science',
        'bachelor', 'master', 'degree', 'diploma', 'certificate', 'graduation', 'percent', 'percentage'
    ]
    
    text_lower = text_clean.lower()
    if any(keyword in text_lower for keyword in exclusion_keywords):
        return False
    
    # Check for common location suffixes/patterns
    location_patterns = [
        r'.*\d{5,6}$',  # Ends with PIN code
        r'.*,\s*(india|usa|uk|canada)$',  # Ends with country
        r'.*(nagar|city|town|village|district|state).*',  # Contains location terms
        r'.*(north|south|east|west).*',  # Directional terms
    ]
    
    for pattern in location_patterns:
        if re.match(pattern, text_lower):
            return False
    
    # Validate word patterns
    for word in words:
        if not word:
            continue
            
        # Must be alphabetic
        if not word.isalpha():
            return False
            
        # Check capitalization (allow Title Case or ALL CAPS)
        if not (word[0].isupper() and (word[1:].islower() or word.isupper())):
            return False
        
        # Individual word length check
        if len(word) < 2 or len(word) > 20:
            return False
    
    # Additional validation for single words
    if len(words) == 1:
        word = words[0]
        # Single word must be at least 3 chars for names, allow some common short names
        if len(word) < 3 and word.lower() not in ['jo', 'al', 'ed', 'li', 'bo', 'ty']:
            return False
    
    # Check if it looks like a person's name pattern
    # Names typically have 1-3 words, each starting with capital
    if len(words) > 4:
        return False
    
    return True

def extract_name(text, entities):
    """Ultra-accurate name extraction with comprehensive validation"""
    lines = [line.strip() for line in text.splitlines()[:10]]
    
    # Known location/place names to exclude (common false positives)
    location_names = {
        'puducherry', 'chennai', 'mumbai', 'delhi', 'bangalore', 'hyderabad', 'kolkata', 'pune', 'ahmedabad',
        'surat', 'jaipur', 'lucknow', 'kanpur', 'nagpur', 'indore', 'thane', 'bhopal', 'visakhapatnam',
        'pimpri', 'patna', 'vadodara', 'ghaziabad', 'ludhiana', 'agra', 'nashik', 'faridabad', 'meerut',
        'rajkot', 'kalyan', 'vasai', 'varanasi', 'srinagar', 'aurangabad', 'dhanbad', 'amritsar', 'navi',
        'allahabad', 'ranchi', 'howrah', 'coimbatore', 'jabalpur', 'gwalior', 'vijayawada', 'jodhpur',
        'madurai', 'raipur', 'kota', 'guwahati', 'chandigarh', 'solapur', 'hubballi', 'tiruchirappalli',
        'bareilly', 'mysore', 'tiruppur', 'gurgaon', 'aligarh', 'jalandhar', 'bhubaneswar', 'salem',
        'warangal', 'guntur', 'bhiwandi', 'saharanpur', 'gorakhpur', 'bikaner', 'amravati', 'noida',
        'jamshedpur', 'bhilai', 'cuttack', 'firozabad', 'kochi', 'nellore', 'bhavnagar', 'dehradun',
        'durgapur', 'asansol', 'rourkela', 'nanded', 'kolhapur', 'ajmer', 'akola', 'gulbarga', 'jamnagar',
        'ujjain', 'loni', 'siliguri', 'jhansi', 'ulhasnagar', 'jammu', 'sangli', 'mangalore', 'erode',
        'belgaum', 'ambattur', 'tirunelveli', 'malegaon', 'gaya', 'jalgaon', 'udaipur', 'maheshtala',
        'india', 'united states', 'usa', 'uk', 'canada', 'australia', 'singapore', 'malaysia', 'thailand'
    }
    
    # Strategy 1: Check first line specifically (most common position for names)
    if lines:
        first_line = lines[0].strip()
        
        # Clean the first line - remove common prefixes
        first_line_clean = re.sub(r'^(resume|cv|curriculum vitae)[:\s]*', '', first_line, flags=re.IGNORECASE).strip()
        
        # Check if first line is a clear name (not containing contact info or locations)
        if (first_line_clean and 
            is_likely_name(first_line_clean) and
            first_line_clean.lower() not in location_names and
            len(first_line_clean.split()) <= 3 and
            not any(keyword in first_line_clean.lower() for keyword in 
                   ['phone', 'email', '@', 'mobile', 'contact', 'address', 'linkedin', 'github', 'www', 'http'])):
            return first_line_clean
    
    # Strategy 2: Look for PERSON entities with strict validation
    person_entities = [(ent_text, ent_label) for ent_text, ent_label in entities if ent_label == "PERSON"]
    
    # Score each person entity based on position and context
    scored_entities = []
    for ent_text, ent_label in person_entities:
        if not is_likely_name(ent_text) or ent_text.lower() in location_names:
            continue
            
        score = 0
        found_line_index = -1
        context_line = ""
        
        # Find which line contains this entity
        for i, line in enumerate(lines):
            if ent_text.lower() in line.lower():
                found_line_index = i
                context_line = line.lower()
                break
        
        if found_line_index == -1:
            continue
            
        # Scoring criteria
        # Higher score for earlier lines
        score += max(10 - found_line_index, 0)
        
        # Bonus for being in first 3 lines
        if found_line_index < 3:
            score += 5
        
        # Penalty for being in lines with address keywords
        if any(addr_keyword in context_line for addr_keyword in 
               ['address', 'street', 'st,', 'road', 'avenue', 'lane', 'city', 'pin', 'zip', 'state']):
            score -= 10
        
        # Penalty for being in lines with contact info
        if any(contact_keyword in context_line for contact_keyword in 
               ['phone', 'email', '@', 'mobile', 'contact', 'linkedin', 'github', 'tel']):
            score -= 5
        
        # Bonus for being standalone on a line (likely to be a name)
        if context_line.strip().lower() == ent_text.lower():
            score += 8
        
        # Bonus for proper capitalization patterns
        if all(word[0].isupper() for word in ent_text.split() if word):
            score += 3
        
        scored_entities.append((score, ent_text, found_line_index))
    
    # Return the highest scored entity
    if scored_entities:
        scored_entities.sort(reverse=True, key=lambda x: x[0])
        best_entity = scored_entities[0]
        if best_entity[0] > 0:  # Only return if score is positive
            return best_entity[1]
    
    # Strategy 3: Look for explicit name patterns in text
    name_patterns = [
        r'(?:name|full name)[:\s]+([A-Z][a-zA-Z\s]{2,30})',
        r'^([A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)?)(?:\s|$)',
        r'resume (?:of|for) ([A-Z][a-zA-Z\s]{2,30})',
        r'^([A-Z]{2,}(?:\s[A-Z]{2,})*)$'  # All caps names
    ]
    
    for pattern in name_patterns:
        matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
        for match in matches:
            name_candidate = match.group(1).strip()
            if (is_likely_name(name_candidate) and 
                name_candidate.lower() not in location_names and
                len(name_candidate.split()) <= 4):
                return name_candidate
    
    # Strategy 4: Advanced line-by-line analysis with context awareness
    for i, line in enumerate(lines[:5]):
        line = line.strip()
        line_lower = line.lower()
        
        # Skip lines that are clearly not names
        if (not line or
            any(skip_keyword in line_lower for skip_keyword in 
                ['phone', 'email', '@', 'mobile', 'contact', 'address', 'linkedin', 'github', 'www', 'http',
                 'objective', 'summary', 'about', 'skills', 'education', 'experience', 'projects']) or
            any(char.isdigit() for char in line)):
            continue
        
        # Check if line could be a name
        words = line.split()
        if 1 <= len(words) <= 3:
            # Check if all words look like name parts
            if (all(word[0].isupper() and word[1:].islower() or word.isupper() for word in words if word.isalpha()) and
                all(word.lower() not in location_names for word in words) and
                is_likely_name(line)):
                return line
    
    return "Not found"

def extract_location(text, entities):
    """Enhanced location extraction with multiple strategies"""
    
    # Strategy 1: Try spaCy GPE entities first with better filtering
    for ent_text, ent_label in entities:
        if (ent_label == "GPE" and 
            ent_text.strip().upper() not in ["NO", "YES", "LINKEDIN", "GITHUB", "FACEBOOK", "RESUME", "CV"] and
            ent_text.upper() not in IGNORE_LIST and
            len(ent_text.strip()) > 2 and
            not ent_text.isdigit()):
            return ent_text
    
    # Strategy 2: Look for explicit address/location patterns
    location_patterns = [
        r"Address[:\-\s]*(.+?)(?:\n|$)",
        r"Location[:\-\s]*(.+?)(?:\n|$)", 
        r"City[:\-\s]*(.+?)(?:\n|$)",
        r"([A-Za-z\s]+,\s*[A-Za-z\s]+(?:,\s*\d{5,6})?)",  # City, State format
        r"([A-Za-z\s]+\s+\d{5,6})",  # City with PIN/ZIP
        r"(\w+(?:\s+\w+)*,\s*\w+(?:\s+\w+)*)"  # General comma-separated format
    ]
    
    for pattern in location_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            location = match.group(1).strip()
            # Clean up and validate location
            if (len(location) > 3 and 
                not location.lower().startswith(('phone', 'email', 'mobile', 'tel')) and
                not re.match(r'^[\d\-\+\(\)\s]+$', location)):  # Not just numbers/phone
                return location
    
    # Strategy 3: Look for lines with address/location keywords
    for line in text.splitlines():
        line_lower = line.lower().strip()
        if any(word in line_lower for word in ["address", "location", "city", "state", "district", "pincode", "pin"]):
            # Extract location after the keyword
            parts = re.split(r'[:\-]', line, 1)
            if len(parts) > 1:
                location = parts[1].strip()
                if (len(location) > 2 and 
                    not re.match(r'^[\d\-\+\(\)\s]+$', location) and  # Not just phone number
                    not any(keyword in location.lower() for keyword in ['phone', 'email', 'mobile'])):
                    return location
            else:
                # Remove common prefixes and return clean location
                clean_line = re.sub(r'^(address|location|city|state|district|pincode|pin)[:\-\s]*', '', line, flags=re.IGNORECASE).strip()
                if (len(clean_line) > 2 and 
                    not re.match(r'^[\d\-\+\(\)\s]+$', clean_line)):
                    return clean_line
    
    # Strategy 4: Look for location patterns in any line
    for line in text.splitlines()[:10]:  # Check first 10 lines
        line = line.strip()
        # Match common location formats like "City, State" or "City - State"
        location_match = re.search(r'^([A-Za-z\s]+(?:[,\-]\s*[A-Za-z\s]+)+)', line)
        if location_match:
            location = location_match.group(1).strip()
            if (len(location) > 5 and 
                not any(keyword in location.lower() for keyword in ['phone', 'email', 'mobile', 'resume', 'cv', 'linkedin', 'github'])):
                return location
    
    return "Not found"

def extract_dates(entities):
    """Extract meaningful dates (graduation years, work periods) from entities"""
    import datetime
    current_year = datetime.datetime.now().year
    dates = []
    seen_dates = set()
    
    for text, label in entities:
        if label == "DATE":
            # Look for graduation years (typically 4 years ago to present)
            year_match = re.search(r'\b(20\d{2})\b', text)
            if year_match:
                year = int(year_match.group(1))
                # Only include reasonable years (graduation/work years)
                if 2015 <= year <= current_year and year not in seen_dates:
                    dates.append(str(year))
                    seen_dates.add(year)
    
    # Limit to most recent 3 years to avoid clutter
    dates = sorted(dates, reverse=True)[:3]
    return ", ".join(dates) if dates else "Not found"

def extract_percent(entities):
    for text, label in entities:
        if label == "PERCENT":
            return text
    return "Not found"

def generate_summary(entities, resume_text):
    """Generate comprehensive resume summary"""
    name = extract_name(resume_text, entities)
    location = extract_location(resume_text, entities)
    dates = extract_dates(entities)
    percent = extract_percent(entities)
    
    # Get additional analysis
    content_analysis = analyze_resume_content(resume_text)
    
    # Debug print to verify correct values
    print(f"DEBUG - Resume text (first 500 chars): {resume_text[:500]}")
    print(f"DEBUG - Education Level: {content_analysis['education_level']}")
    print(f"DEBUG - Experience Level: {content_analysis['experience_level']}")
    
    return {
        "name": name,
        "locations": location,
        "dates": dates,
        "percent": percent,
        "skills": content_analysis['skills'],
        "education_level": content_analysis['education_level'],
        "experience_level": content_analysis['experience_level']
    }

def analyze_entities(doc, resume_text):
    """Analyze entities and predict job role using ML"""
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ != 'ORG']
    
    # Get ML-based job prediction
    predicted_role, ml_confidence = predict_job_role(resume_text)
    
    # Calculate resume completeness score (0-100)
    completeness_factors = {
        'name': 20,      # Name found
        'location': 15,  # Location found  
        'dates': 15,     # Education/work dates
        'skills': 25,    # Technical skills found
        'experience': 15, # Experience level detected
        'achievements': 10 # Percentages/achievements
    }
    
    # Initialize score breakdown for transparency
    score_breakdown = {
        'ml_confidence': ml_confidence,
        'completeness_score': 0,
        'components': {
            'name': {'score': 0, 'max': 20, 'found': False, 'value': "Not found"},
            'location': {'score': 0, 'max': 15, 'found': False, 'value': "Not found"},
            'dates': {'score': 0, 'max': 15, 'found': False, 'value': "Not found"},
            'skills': {'score': 0, 'max': 25, 'found': False, 'value': []},
            'experience': {'score': 0, 'max': 15, 'found': False, 'value': "Entry Level"},
            'achievements': {'score': 0, 'max': 10, 'found': False, 'value': "Not found"}
        }
    }
    
    completeness_score = 0
    content_analysis = analyze_resume_content(resume_text)
    
    # Check each factor and update breakdown
    name_value = extract_name(resume_text, entities)
    if name_value != "Not found":
        completeness_score += completeness_factors['name']
        score_breakdown['components']['name']['score'] = completeness_factors['name']
        score_breakdown['components']['name']['found'] = True
        score_breakdown['components']['name']['value'] = name_value

    location_value = extract_location(resume_text, entities)
    if location_value != "Not found":
        completeness_score += completeness_factors['location']
        score_breakdown['components']['location']['score'] = completeness_factors['location']
        score_breakdown['components']['location']['found'] = True
        score_breakdown['components']['location']['value'] = location_value

    dates_value = extract_dates(entities)
    if dates_value != "Not found":
        completeness_score += completeness_factors['dates']
        score_breakdown['components']['dates']['score'] = completeness_factors['dates']
        score_breakdown['components']['dates']['found'] = True
        score_breakdown['components']['dates']['value'] = dates_value
    
    if content_analysis['skills']:
        # Scale based on number of skills (more skills = higher score)
        skill_score = min(len(content_analysis['skills']) * 3, completeness_factors['skills'])
        completeness_score += skill_score
        score_breakdown['components']['skills']['score'] = skill_score
        score_breakdown['components']['skills']['found'] = True
        score_breakdown['components']['skills']['value'] = content_analysis['skills']
    
    if content_analysis['experience_level'] != "Entry Level":
        completeness_score += completeness_factors['experience']
        score_breakdown['components']['experience']['score'] = completeness_factors['experience']
        score_breakdown['components']['experience']['found'] = True
    score_breakdown['components']['experience']['value'] = content_analysis['experience_level']
    
    achievements_value = extract_percent(entities)
    if achievements_value != "Not found":
        completeness_score += completeness_factors['achievements']
        score_breakdown['components']['achievements']['score'] = completeness_factors['achievements']
        score_breakdown['components']['achievements']['found'] = True
        score_breakdown['components']['achievements']['value'] = achievements_value
    
    score_breakdown['completeness_score'] = completeness_score
    
    # Convert completeness score to percentage for display consistency
    completeness_percentage = completeness_score  # Already 0-100 scale
    
    # Final match percentage: 60% ML confidence + 40% completeness
    # Both ml_confidence and completeness_score are on 0-100 scale
    match_percentage = int((ml_confidence * 0.6) + (completeness_percentage * 0.4))
    
    # Ensure minimum of 40% if basic info is present
    if completeness_score > 50:
        match_percentage = max(match_percentage, 40)
    
    score_breakdown['final_score'] = match_percentage
    score_breakdown['ml_weight'] = 0.6
    score_breakdown['completeness_weight'] = 0.4
    score_breakdown['completeness_percentage'] = completeness_percentage
    
    # Generate suggestions based on missing information
    missing_info = []
    if extract_name(resume_text, entities) == "Not found":
        missing_info.append("NAME")
    if extract_location(resume_text, entities) == "Not found":
        missing_info.append("LOCATION")
    if extract_dates(entities) == "Not found":
        missing_info.append("DATES")
    if not content_analysis['skills']:
        missing_info.append("TECHNICAL_SKILLS")
    if extract_percent(entities) == "Not found":
        missing_info.append("ACHIEVEMENTS")
    
    return entities, match_percentage, missing_info, predicted_role, score_breakdown

def generate_keyword_suggestions(predicted_role, content_analysis, resume_text):
    """Generate relevant keywords and suggestions based on job role and current resume content"""
    
    # Job-specific keyword databases
    job_keywords = {
        "Software Engineer": {
            "technical_skills": [
                "Python", "Java", "JavaScript", "C++", "React", "Node.js", "SQL", "Git", 
                "Docker", "Kubernetes", "AWS", "CI/CD", "Agile", "Scrum", "REST API", 
                "Microservices", "TDD", "Object-Oriented Programming", "Data Structures", "Algorithms"
            ],
            "soft_skills": [
                "Problem-solving", "Team collaboration", "Communication", "Leadership", 
                "Critical thinking", "Adaptability", "Time management", "Project management"
            ],
            "action_verbs": [
                "Developed", "Implemented", "Designed", "Optimized", "Built", "Created", 
                "Maintained", "Integrated", "Deployed", "Collaborated", "Led", "Managed"
            ],
            "certifications": [
                "AWS Certified Developer", "Oracle Java Certification", "Microsoft Azure Fundamentals",
                "Google Cloud Professional", "Scrum Master Certification"
            ]
        },
        "Frontend Developer": {
            "technical_skills": [
                "HTML5", "CSS3", "JavaScript", "React", "Angular", "Vue.js", "TypeScript", 
                "SASS/SCSS", "Webpack", "Bootstrap", "Responsive Design", "Cross-browser Compatibility",
                "Performance Optimization", "Progressive Web Apps", "Material-UI", "Tailwind CSS"
            ],
            "soft_skills": [
                "UI/UX Design", "Attention to detail", "Creativity", "User-centered thinking",
                "Cross-functional collaboration", "Communication", "Problem-solving"
            ],
            "action_verbs": [
                "Designed", "Developed", "Implemented", "Optimized", "Created", "Built", 
                "Enhanced", "Improved", "Collaborated", "Delivered"
            ],
            "certifications": [
                "Google UX Design Certificate", "Adobe Certified Expert", "W3C Frontend Certification",
                "React Developer Certification"
            ]
        },
        "Backend Developer": {
            "technical_skills": [
                "Python", "Java", "Node.js", "C#", ".NET", "SQL", "NoSQL", "MongoDB", "PostgreSQL",
                "REST API", "GraphQL", "Microservices", "Docker", "Kubernetes", "AWS", "Azure",
                "Redis", "Message Queues", "Database Design", "System Architecture"
            ],
            "soft_skills": [
                "System thinking", "Problem-solving", "Performance optimization", "Security awareness",
                "Team collaboration", "Documentation", "Code review"
            ],
            "action_verbs": [
                "Architected", "Developed", "Implemented", "Optimized", "Designed", "Built",
                "Maintained", "Scaled", "Integrated", "Deployed"
            ],
            "certifications": [
                "AWS Solutions Architect", "Microsoft Azure Developer", "Oracle Database Certification",
                "MongoDB Developer Certification"
            ]
        },
        "Data Scientist": {
            "technical_skills": [
                "Python", "R", "SQL", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch",
                "Pandas", "NumPy", "Scikit-learn", "Matplotlib", "Seaborn", "Jupyter", "Statistics",
                "Data Visualization", "Big Data", "Hadoop", "Spark", "Tableau", "Power BI"
            ],
            "soft_skills": [
                "Analytical thinking", "Statistical analysis", "Business acumen", "Communication",
                "Data storytelling", "Problem-solving", "Research methodology"
            ],
            "action_verbs": [
                "Analyzed", "Modeled", "Predicted", "Optimized", "Visualized", "Researched",
                "Implemented", "Built", "Developed", "Improved"
            ],
            "certifications": [
                "Google Data Analytics Certificate", "IBM Data Science Certification", 
                "Microsoft Azure Data Scientist", "Tableau Certification"
            ]
        },
        "DevOps Engineer": {
            "technical_skills": [
                "Docker", "Kubernetes", "Jenkins", "CI/CD", "AWS", "Azure", "GCP", "Terraform",
                "Ansible", "Linux", "Shell Scripting", "Python", "Monitoring", "Prometheus",
                "Grafana", "Infrastructure as Code", "GitOps", "Microservices"
            ],
            "soft_skills": [
                "Process improvement", "Automation mindset", "Troubleshooting", "Collaboration",
                "System thinking", "Continuous learning", "Problem-solving"
            ],
            "action_verbs": [
                "Automated", "Deployed", "Configured", "Optimized", "Monitored", "Implemented",
                "Maintained", "Scaled", "Integrated", "Streamlined"
            ],
            "certifications": [
                "AWS DevOps Engineer", "Azure DevOps Engineer", "Docker Certified Associate",
                "Kubernetes Administrator Certification"
            ]
        }
    }
    
    # Default keywords for general roles
    default_keywords = {
        "technical_skills": [
            "Programming", "Software Development", "Problem Solving", "Testing", "Debugging",
            "Version Control", "Documentation", "Code Review", "Agile Methodology"
        ],
        "soft_skills": [
            "Communication", "Teamwork", "Leadership", "Problem-solving", "Adaptability",
            "Time management", "Critical thinking", "Collaboration"
        ],
        "action_verbs": [
            "Developed", "Implemented", "Created", "Managed", "Led", "Collaborated",
            "Improved", "Optimized", "Delivered", "Achieved"
        ],
        "certifications": [
            "Industry-relevant certifications", "Professional development courses",
            "Technical training programs"
        ]
    }
    
    # Get keywords for the predicted role
    role_keywords = job_keywords.get(predicted_role, default_keywords)
    
    # Analyze current resume content
    resume_lower = resume_text.lower()
    current_skills = [skill.lower() for skill in content_analysis['skills']]
    
    # Find missing technical keywords
    missing_technical = []
    for keyword in role_keywords["technical_skills"]:
        if keyword.lower() not in resume_lower and keyword.lower() not in current_skills:
            missing_technical.append(keyword)
    
    # Find missing soft skills
    missing_soft = []
    for keyword in role_keywords["soft_skills"]:
        if keyword.lower() not in resume_lower:
            missing_soft.append(keyword)
    
    # Find missing action verbs
    missing_verbs = []
    for verb in role_keywords["action_verbs"]:
        if verb.lower() not in resume_lower:
            missing_verbs.append(verb)
    
    # Generate improvement suggestions
    improvement_suggestions = generate_improvement_descriptions(
        predicted_role, content_analysis, missing_technical, missing_soft, missing_verbs
    )
    
    return {
        "role": predicted_role,
        "missing_technical_skills": missing_technical[:8],  # Top 8 suggestions
        "missing_soft_skills": missing_soft[:6],           # Top 6 suggestions
        "missing_action_verbs": missing_verbs[:6],         # Top 6 suggestions
        "suggested_certifications": role_keywords["certifications"][:4],  # Top 4 suggestions
        "improvement_suggestions": improvement_suggestions,
        "current_skills_count": len(content_analysis['skills']),
        "skill_gap_analysis": {
            "strong_areas": content_analysis['skills'],
            "improvement_areas": missing_technical[:5]
        }
    }

def generate_improvement_descriptions(predicted_role, content_analysis, missing_technical, missing_soft, missing_verbs):
    """Generate detailed descriptions on how to improve resume score"""
    
    suggestions = []
    
    # Technical Skills Improvement
    if len(content_analysis['skills']) < 5:
        suggestions.append({
            "category": "Technical Skills",
            "priority": "High",
            "description": f"Add more technical skills relevant to {predicted_role}. You currently have {len(content_analysis['skills'])} skills listed. Aim for 8-12 key technical skills.",
            "action_items": [
                f"Include programming languages: {', '.join(missing_technical[:3])}",
                "Add frameworks and tools you've used",
                "Mention databases and cloud platforms",
                "Include development methodologies (Agile, Scrum)"
            ],
            "impact": "Can increase your score by up to 15 points"
        })
    
    # Experience Enhancement
    if content_analysis['experience_level'] == "Entry Level":
        suggestions.append({
            "category": "Experience Description",
            "priority": "High",
            "description": "Enhance your experience descriptions with quantifiable achievements and impact metrics.",
            "action_items": [
                "Add specific numbers and percentages to achievements",
                "Use strong action verbs to start bullet points",
                "Describe the impact of your work on business/team",
                "Include technologies used in each role"
            ],
            "impact": "Can increase your score by up to 10 points"
        })
    
    # Action Verbs Improvement
    if len(missing_verbs) > 3:
        suggestions.append({
            "category": "Action Verbs",
            "priority": "Medium",
            "description": "Use stronger action verbs to make your resume more impactful and dynamic.",
            "action_items": [
                f"Replace weak verbs with: {', '.join(missing_verbs[:4])}",
                "Start each bullet point with a strong action verb",
                "Avoid repetitive verbs throughout your resume",
                "Use past tense for previous roles, present for current"
            ],
            "impact": "Improves overall resume readability and impact"
        })
    
    # Soft Skills Enhancement
    if len(missing_soft) > 3:
        suggestions.append({
            "category": "Soft Skills Integration",
            "priority": "Medium",
            "description": "Integrate soft skills naturally into your experience descriptions rather than listing them separately.",
            "action_items": [
                f"Demonstrate skills like: {', '.join(missing_soft[:3])}",
                "Show leadership through project examples",
                "Highlight collaboration in team settings",
                "Mention communication in client/stakeholder interactions"
            ],
            "impact": "Makes your profile more well-rounded"
        })
    
    # Keywords Optimization
    suggestions.append({
        "category": "Keyword Optimization",
        "priority": "High",
        "description": f"Optimize your resume with {predicted_role}-specific keywords to improve ATS compatibility.",
        "action_items": [
            "Include industry-standard terminology",
            "Match keywords from job descriptions you're targeting",
            "Use both acronyms and full forms (e.g., 'AI' and 'Artificial Intelligence')",
            "Naturally integrate keywords into context"
        ],
        "impact": "Significantly improves ATS scanning and recruiter matching"
    })
    
    # Quantifiable Achievements
    suggestions.append({
        "category": "Quantifiable Results",
        "priority": "High",
        "description": "Add more numbers, percentages, and measurable outcomes to demonstrate your impact.",
        "action_items": [
            "Include performance improvements (e.g., '30% faster processing')",
            "Mention team sizes you've led or worked with",
            "Add project budgets, timelines, or user counts",
            "Quantify cost savings or revenue increases"
        ],
        "impact": "Can increase your score by up to 10 points"
    })
    
    return suggestions

def get_suggestions(missing_entities, job_role):
    """Generate actionable suggestions based on missing information and predicted job role"""
    suggestions = []
    
    # General suggestions based on missing information
    for ent in missing_entities:
        if ent == "NAME":
            suggestions.append("Add your full name prominently at the top of your resume.")
        elif ent == "DATES":
            suggestions.append("Include graduation year and work experience dates to show career progression.")
        elif ent == "LOCATION":
            suggestions.append("Mention your current location or preferred work location.")
        elif ent == "ACHIEVEMENTS":
            suggestions.append("Add quantifiable achievements with percentages, numbers, or metrics.")
        elif ent == "TECHNICAL_SKILLS":
            suggestions.append("Include a dedicated skills section with relevant technical skills.")
    
    # Job-specific suggestions
    job_specific_tips = {
        "Data Scientist": [
            "Highlight your experience with Python, R, SQL, and machine learning frameworks.",
            "Include specific projects showing data analysis and statistical modeling skills."
        ],
        "Software Engineer": [
            "Emphasize your programming languages and software development experience.",
            "Include details about software projects and development methodologies."
        ],
        "Frontend Developer": [
            "Highlight your HTML, CSS, JavaScript skills and modern frameworks like React or Angular.",
            "Include portfolio links or examples of responsive web design projects."
        ],
        "Backend Developer": [
            "Emphasize server-side programming languages and database management skills.",
            "Include experience with API development and system architecture."
        ],
        "DevOps Engineer": [
            "Highlight experience with containerization (Docker, Kubernetes) and cloud platforms.",
            "Include details about CI/CD pipelines and infrastructure automation."
        ]
    }
    
    # Add job-specific suggestions
    if job_role in job_specific_tips:
        suggestions.extend(job_specific_tips[job_role][:2])
    
    # General improvement suggestions
    if not missing_entities:
        suggestions.extend([
            "Consider adding more technical skills relevant to your target role.",
            "Include quantifiable achievements and impact metrics in your experience section."
        ])
    
    return suggestions[:5]

@app.route('/', methods=['GET', 'POST'])
def index():
    entities = []
    resume_text = ""
    match_percentage = 0
    missing_entities = []
    job_match = ""
    summary = {}
    suggestions = []
    nltk_analysis = {}
    score_breakdown = {}
    keyword_suggestions = {}
    
    if request.method == 'POST':
        uploaded_file = request.files.get('resume_file')
        manual_name = request.form.get('manual_name', '').strip()
        manual_location = request.form.get('manual_location', '').strip()
        manual_dates = request.form.get('manual_dates', '').strip()
        manual_percent = request.form.get('manual_percent', '').strip()
        
        if uploaded_file and uploaded_file.filename:
            resume_text = extract_text(uploaded_file)
            doc = nlp(resume_text)
            
            # Use ML-based analysis
            entities, match_percentage, missing_entities, job_match, score_breakdown = analyze_entities(doc, resume_text)
            summary = generate_summary(entities, resume_text)
            
            # Generate keyword suggestions and improvement guidance
            content_analysis = analyze_resume_content(resume_text)
            keyword_suggestions = generate_keyword_suggestions(job_match, content_analysis, resume_text)
            
            # Enhanced NLTK analysis (with fallback)
            try:
                nltk_analysis = analyze_resume_with_nltk(resume_text)
            except Exception as e:
                print(f"NLTK analysis failed: {e}")
                # Provide fallback analysis without NLTK
                nltk_analysis = {
                    'sentiment': {'positive': 0.5, 'negative': 0.1, 'neutral': 0.4, 'compound': 0.3},
                    'entities': [],
                    'statistics': {
                        'word_count': len(resume_text.split()),
                        'sentence_count': len(resume_text.split('.')),
                        'avg_sentence_length': len(resume_text.split()) / max(len(resume_text.split('.')), 1)
                    },
                    'key_phrases': [],
                    'filtered_words': []
                }
            
            # Manual corrections
            if manual_name:
                summary['name'] = manual_name
            if manual_location:
                summary['locations'] = manual_location
            if manual_dates:
                summary['dates'] = manual_dates
            if manual_percent:
                summary['percent'] = manual_percent
            
            suggestions = get_suggestions(missing_entities, job_match)
    
    return render_template(
        'index.html',
        entities=entities,
        resume_text=resume_text,
        match_percentage=match_percentage,
        missing_entities=missing_entities,
        job_match=job_match,
        summary=summary,
        suggestions=suggestions,
        nltk_analysis=nltk_analysis,
        score_breakdown=score_breakdown,
        keyword_suggestions=keyword_suggestions
    )

if __name__ == '__main__':
    app.run(debug=True)
