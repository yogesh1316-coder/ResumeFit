# 🚀 ResumeFit - AI-Powered Resume Analyzer

**An intelligent platform designed to simplify and enhance the job-seeking process through advanced AI and Natural Language Processing.**

ResumeFit uses cutting-edge AI technology to analyze resumes and match them with suitable job roles. By comparing candidates' skills, experience, and qualifications with job descriptions, it ensures accurate and efficient recommendations, making recruitment faster, smarter, and more effective.

## ✨ Features

### 🎯 Core Functionality
- **Smart Resume Analysis**: AI-powered analysis using NLTK and spaCy
- **Job Role Matching**: Machine learning-based job prediction with 60% weight
- **Completeness Scoring**: Comprehensive resume evaluation with 40% weight
- **Skill Gap Analysis**: Identify missing skills and areas for improvement
- **Educational Level Detection**: Accurate degree recognition (Bachelor's, Master's, etc.)
- **Experience Assessment**: Automatic experience level categorization

### 🎨 Modern Web Interface
- **Responsive Design**: Mobile-friendly with hamburger menu navigation
- **Multi-Section Layout**: Home, Dashboard, About, and Contact sections
- **Professional Styling**: Glass-morphism effects and gradient backgrounds
- **Interactive Navigation**: Smooth section transitions
- **Real-time Analysis**: Dynamic results display with visual breakdowns

### 📊 Advanced Analytics
- **Transparent Scoring**: Clear breakdown of analysis components
- **Visual Feedback**: Progress bars and component cards
- **Detailed Suggestions**: Personalized improvement recommendations
- **Formula Visualization**: See exactly how scores are calculated

### 📧 Contact Integration
- **Gmail Integration**: Direct email composition in new tab
- **Contact Form**: Professional contact form with subject selection
- **Support Email**: Direct communication channel (hsegoy1316@gmail.com)

## 🛠️ Technologies Used

### Backend
- **Python 3.x** - Core programming language
- **Flask** - Web framework for API and routing
- **NLTK** - Natural language processing
- **spaCy** - Advanced NLP and entity recognition
- **scikit-learn** - Machine learning for job matching
- **pdfplumber** - PDF text extraction
- **numpy** - Numerical computing

### Frontend
- **HTML5** - Modern semantic markup
- **CSS3** - Advanced styling with gradients and animations
- **JavaScript (ES6)** - Interactive functionality
- **Font Awesome** - Professional icons
- **Responsive Design** - Mobile-first approach

### AI/ML Components
- **TF-IDF Vectorization** - Text feature extraction
- **Logistic Regression** - Job role classification
- **Named Entity Recognition** - Information extraction
- **Regular Expressions** - Pattern matching for education levels

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager
- Modern web browser

### 1. Clone the Repository
```bash
git clone https://github.com/SilentProgrammer-max/AI-Powered-Resume-Analyzer.git
cd AI-Powered-Resume-Analyzer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Required NLP Models
```bash
python fix_nltk.py
```

### 4. Start the Application
```bash
python start_app.py
```
*or*
```bash
python app.py
```

### 5. Access the Application
Open your browser and navigate to:
```
http://localhost:5000
```

## 📖 How to Use

### 1. **Upload Resume**
- Navigate to the Dashboard section
- Click "Choose File" and select your PDF resume
- Click "Analyze Resume" to start processing

### 2. **View Analysis Results**
- **Job Match**: See predicted job roles with confidence scores
- **Overall Score**: Combined score from ML analysis (60%) and completeness (40%)
- **Component Breakdown**: Detailed analysis of resume elements
- **Suggestions**: Personalized recommendations for improvement

### 3. **Explore Features**
- **Home**: Learn about ResumeFit's capabilities
- **About**: Understand the AI technology behind the platform
- **Contact**: Get in touch for support or feedback

## 🏗️ Project Structure

```
ResumeFit/
├── app.py                     # Main Flask application
├── start_app.py              # Application startup script
├── requirements.txt          # Python dependencies
├── debug_test.py            # Debug utilities
├── fix_nltk.py             # NLTK setup script
├── static/
│   ├── enhanced_style.css   # Modern CSS styling
│   └── script.js           # Interactive JavaScript
├── templates/
│   └── index.html          # Main web interface
└── README.md               # Project documentation
```

## 🎯 Scoring Algorithm

ResumeFit uses a sophisticated scoring system:

```
Final Score = (ML Job Match × 60%) + (Resume Completeness × 40%)
```

### Component Weights:
- **Name Detection**: 5 points
- **Location Information**: 10 points
- **Education Level**: 15 points
- **Experience Level**: 20 points
- **Contact Information**: 10 points
- **Skills & Keywords**: Variable based on relevance

## 🔧 Configuration

### Environment Variables
The application runs with the following default settings:
- **Host**: `127.0.0.1` (localhost)
- **Port**: `5000`
- **Debug Mode**: Enabled in development

### Supported File Formats
- **PDF**: Primary format for resume analysis
- **Text extraction**: Handles various PDF structures and layouts

## 🧪 Testing & Debugging

### Debug Tools
- **debug_test.py**: Test core functionality
- **Browser DevTools**: Inspect frontend behavior
- **Flask Debug Mode**: Detailed error information

### Testing Resume Analysis
Use the included test resume or upload your own PDF to verify:
- Text extraction accuracy
- Entity recognition performance
- Job matching predictions
- Score calculations

## 🤝 Contributing

We welcome contributions to ResumeFit! Here's how you can help:

1. **Fork the Repository**
2. **Create a Feature Branch**: `git checkout -b feature/amazing-feature`
3. **Make Your Changes**: Implement your feature or bug fix
4. **Test Thoroughly**: Ensure all functionality works correctly
5. **Commit Changes**: `git commit -m "Add amazing feature"`
6. **Push to Branch**: `git push origin feature/amazing-feature`
7. **Open Pull Request**: Submit your changes for review

### Development Guidelines
- Follow PEP 8 for Python code
- Use meaningful commit messages
- Test your changes thoroughly
- Update documentation as needed

## 📧 Support & Contact

- **Email**: hsegoy1316@gmail.com
- **GitHub Issues**: Report bugs or request features
- **Response Time**: We typically respond within 24 hours

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NLTK Team** - Natural Language Processing toolkit
- **spaCy Team** - Advanced NLP library
- **Flask Community** - Web framework support
- **scikit-learn** - Machine learning algorithms
- **Open Source Community** - Inspiration and resources

---

**ResumeFit** - *Bridging the gap between candidates and employers through intelligent resume analysis* 🎯✨
