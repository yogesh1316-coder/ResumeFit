// Enhanced Resume Analyzer JavaScript Interface

class ResumeAnalyzer {
    constructor() {
        this.currentFile = null;
        this.isAnalyzing = false;
        this.initializeEventListeners();
        this.setupDragAndDrop();
        this.initializeProgressSystem();
    }

    initializeEventListeners() {
        // File input change handler
        const fileInput = document.getElementById('resume_file');
        if (fileInput) {
            fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        }

        // Form submission handler
        const form = document.querySelector('form');
        if (form) {
            form.addEventListener('submit', this.handleFormSubmit.bind(this));
        }

        // Real-time manual correction handlers
        this.setupManualCorrectionHandlers();
    }

    setupDragAndDrop() {
        const dropZone = document.getElementById('drop-zone') || this.createDropZone();
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, this.preventDefaults.bind(this), false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, this.highlight.bind(this), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, this.unhighlight.bind(this), false);
        });

        dropZone.addEventListener('drop', this.handleDrop.bind(this), false);
    }

    createDropZone() {
        const fileInput = document.getElementById('resume_file');
        const dropZone = document.createElement('div');
        dropZone.id = 'drop-zone';
        dropZone.innerHTML = `
            <div class="drop-zone-content">
                <i class="upload-icon">📄</i>
                <p>Drag and drop your resume here or click to browse</p>
                <small>Supports PDF and DOCX files</small>
            </div>
        `;
        dropZone.style.cssText = `
            border: 2px dashed #007bff;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin: 20px 0;
            cursor: pointer;
            transition: all 0.3s ease;
        `;
        
        if (fileInput && fileInput.parentNode) {
            fileInput.parentNode.insertBefore(dropZone, fileInput);
            fileInput.style.display = 'none';
        }

        dropZone.addEventListener('click', () => fileInput.click());
        
        return dropZone;
    }

    initializeProgressSystem() {
        this.progressContainer = this.createProgressContainer();
    }

    createProgressContainer() {
        const container = document.createElement('div');
        container.id = 'progress-container';
        container.style.cssText = `
            display: none;
            margin: 20px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        `;
        
        container.innerHTML = `
            <div class="progress-header">
                <h4>Processing Resume...</h4>
                <div class="spinner"></div>
            </div>
            <div class="progress-bar-container">
                <div class="progress-bar" id="progress-bar"></div>
            </div>
            <div class="progress-steps">
                <div class="step" id="step-1">📤 Uploading file...</div>
                <div class="step" id="step-2">🔍 Extracting text...</div>
                <div class="step" id="step-3">🧠 Analyzing with ML...</div>
                <div class="step" id="step-4">📊 Generating insights...</div>
                <div class="step" id="step-5">✅ Complete!</div>
            </div>
        `;

        // Add CSS for progress system
        this.addProgressStyles();
        
        const form = document.querySelector('form');
        if (form) {
            form.appendChild(container);
        }
        
        return container;
    }

    addProgressStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .spinner {
                width: 20px;
                height: 20px;
                border: 2px solid #f3f3f3;
                border-top: 2px solid #007bff;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                display: inline-block;
                margin-left: 10px;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .progress-bar-container {
                width: 100%;
                height: 8px;
                background-color: #e9ecef;
                border-radius: 4px;
                margin: 15px 0;
                overflow: hidden;
            }

            .progress-bar {
                height: 100%;
                background: linear-gradient(90deg, #007bff, #0056b3);
                width: 0%;
                transition: width 0.3s ease;
                border-radius: 4px;
            }

            .progress-steps {
                margin-top: 15px;
            }

            .step {
                padding: 5px 0;
                opacity: 0.5;
                transition: opacity 0.3s ease;
            }

            .step.active {
                opacity: 1;
                font-weight: bold;
                color: #007bff;
            }

            .step.completed {
                opacity: 1;
                color: #28a745;
            }

            #drop-zone.highlight {
                border-color: #0056b3;
                background-color: #f8f9ff;
            }

            .analysis-card {
                background: white;
                border-radius: 8px;
                padding: 20px;
                margin: 15px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border: 1px solid #dee2e6;
            }

            .sentiment-indicator {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                font-weight: bold;
                margin: 2px;
            }

            .sentiment-positive { background: #d4edda; color: #155724; }
            .sentiment-negative { background: #f8d7da; color: #721c24; }
            .sentiment-neutral { background: #d1ecf1; color: #0c5460; }

            .key-phrase {
                display: inline-block;
                background: #e3f2fd;
                color: #1565c0;
                padding: 4px 8px;
                margin: 2px;
                border-radius: 12px;
                font-size: 0.9em;
            }

            .entity-tag {
                display: inline-block;
                background: #fff3cd;
                color: #856404;
                padding: 2px 6px;
                margin: 1px;
                border-radius: 8px;
                font-size: 0.8em;
            }
        `;
        document.head.appendChild(style);
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    highlight(e) {
        const dropZone = document.getElementById('drop-zone');
        if (dropZone) {
            dropZone.classList.add('highlight');
        }
    }

    unhighlight(e) {
        const dropZone = document.getElementById('drop-zone');
        if (dropZone) {
            dropZone.classList.remove('highlight');
        }
    }

    handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0) {
            const file = files[0];
            this.handleFileSelect({ target: { files: [file] } });
        }
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (!file) return;

        // Validate file type
        const validTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
        if (!validTypes.includes(file.type) && !file.name.toLowerCase().endsWith('.pdf') && !file.name.toLowerCase().endsWith('.docx')) {
            this.showAlert('Please select a PDF or DOCX file.', 'error');
            return;
        }

        // Validate file size (10MB limit)
        if (file.size > 10 * 1024 * 1024) {
            this.showAlert('File size must be less than 10MB.', 'error');
            return;
        }

        this.currentFile = file;
        this.updateFileInfo(file);
        
        // Auto-analyze if enabled
        if (document.getElementById('auto-analyze') && document.getElementById('auto-analyze').checked) {
            this.analyzeResume();
        }
    }

    updateFileInfo(file) {
        const fileName = file.name;
        const fileSize = (file.size / 1024).toFixed(1) + ' KB';
        
        let fileInfoDiv = document.getElementById('file-info');
        if (!fileInfoDiv) {
            fileInfoDiv = document.createElement('div');
            fileInfoDiv.id = 'file-info';
            fileInfoDiv.style.cssText = `
                margin: 10px 0;
                padding: 10px;
                background: #e8f5e8;
                border: 1px solid #c3e6c3;
                border-radius: 5px;
            `;
            
            const dropZone = document.getElementById('drop-zone');
            if (dropZone && dropZone.parentNode) {
                dropZone.parentNode.insertBefore(fileInfoDiv, dropZone.nextSibling);
            }
        }
        
        fileInfoDiv.innerHTML = `
            <strong>Selected File:</strong> ${fileName} (${fileSize})
            <button type="button" onclick="resumeAnalyzer.clearFile()" style="margin-left: 10px; padding: 2px 8px; background: #dc3545; color: white; border: none; border-radius: 3px; cursor: pointer;">Remove</button>
        `;
    }

    clearFile() {
        this.currentFile = null;
        const fileInput = document.getElementById('resume_file');
        if (fileInput) {
            fileInput.value = '';
        }
        
        const fileInfo = document.getElementById('file-info');
        if (fileInfo) {
            fileInfo.remove();
        }
    }

    handleFormSubmit(event) {
        if (!this.currentFile) {
            this.showAlert('Please select a resume file first.', 'warning');
            event.preventDefault();
            return false;
        }

        this.showProgress();
        this.simulateProgress();
        
        return true;
    }

    showProgress() {
        const progressContainer = document.getElementById('progress-container');
        if (progressContainer) {
            progressContainer.style.display = 'block';
            this.currentStep = 1;
            this.updateProgressStep(1);
        }
    }

    hideProgress() {
        const progressContainer = document.getElementById('progress-container');
        if (progressContainer) {
            progressContainer.style.display = 'none';
        }
    }

    simulateProgress() {
        const steps = [
            { step: 1, delay: 500, progress: 20 },
            { step: 2, delay: 1000, progress: 40 },
            { step: 3, delay: 1500, progress: 70 },
            { step: 4, delay: 2000, progress: 90 },
            { step: 5, delay: 2500, progress: 100 }
        ];

        steps.forEach(({ step, delay, progress }) => {
            setTimeout(() => {
                this.updateProgressStep(step);
                this.updateProgressBar(progress);
            }, delay);
        });
    }

    updateProgressStep(stepNumber) {
        // Reset all steps
        for (let i = 1; i <= 5; i++) {
            const stepElement = document.getElementById(`step-${i}`);
            if (stepElement) {
                stepElement.classList.remove('active', 'completed');
                if (i < stepNumber) {
                    stepElement.classList.add('completed');
                } else if (i === stepNumber) {
                    stepElement.classList.add('active');
                }
            }
        }
    }

    updateProgressBar(percentage) {
        const progressBar = document.getElementById('progress-bar');
        if (progressBar) {
            progressBar.style.width = percentage + '%';
        }
    }

    setupManualCorrectionHandlers() {
        const manualInputs = ['manual_name', 'manual_location', 'manual_dates', 'manual_percent'];
        
        manualInputs.forEach(inputId => {
            const input = document.getElementById(inputId);
            if (input) {
                input.addEventListener('input', this.handleManualCorrection.bind(this));
            }
        });
    }

    handleManualCorrection(event) {
        const input = event.target;
        const value = input.value.trim();
        
        // Show real-time feedback
        this.showManualCorrectionFeedback(input, value);
    }

    showManualCorrectionFeedback(input, value) {
        let feedback = input.parentNode.querySelector('.correction-feedback');
        
        if (!feedback) {
            feedback = document.createElement('div');
            feedback.className = 'correction-feedback';
            feedback.style.cssText = `
                font-size: 0.8em;
                margin-top: 5px;
                padding: 5px;
                border-radius: 3px;
            `;
            input.parentNode.appendChild(feedback);
        }
        
        if (value) {
            feedback.innerHTML = `✓ Manual correction applied: "${value}"`;
            feedback.style.background = '#d4edda';
            feedback.style.color = '#155724';
            feedback.style.border = '1px solid #c3e6cb';
        } else {
            feedback.remove();
        }
    }

    showAlert(message, type = 'info') {
        // Remove existing alerts
        const existingAlert = document.querySelector('.custom-alert');
        if (existingAlert) {
            existingAlert.remove();
        }

        const alert = document.createElement('div');
        alert.className = 'custom-alert';
        
        const colors = {
            'error': { bg: '#f8d7da', border: '#f5c6cb', text: '#721c24' },
            'warning': { bg: '#fff3cd', border: '#ffeaa7', text: '#856404' },
            'success': { bg: '#d4edda', border: '#c3e6cb', text: '#155724' },
            'info': { bg: '#d1ecf1', border: '#bee5eb', text: '#0c5460' }
        };
        
        const color = colors[type];
        alert.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            background: ${color.bg};
            border: 1px solid ${color.border};
            color: ${color.text};
            border-radius: 5px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            z-index: 1000;
            max-width: 400px;
        `;
        
        alert.innerHTML = `
            ${message}
            <button onclick="this.parentNode.remove()" style="
                float: right;
                background: none;
                border: none;
                font-size: 16px;
                cursor: pointer;
                margin-left: 10px;
                color: ${color.text};
            ">&times;</button>
        `;
        
        document.body.appendChild(alert);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alert.parentNode) {
                alert.remove();
            }
        }, 5000);
    }

    // Enhanced result display
    enhanceResultDisplay() {
        // Add sentiment analysis display
        this.displaySentimentAnalysis();
        
        // Add interactive skill pills
        this.enhanceSkillDisplay();
        
        // Add entity highlighting
        this.highlightEntities();
        
        // Add copy functionality
        this.addCopyFunctionality();
    }

    displaySentimentAnalysis() {
        // This would be called after results are loaded
        const sentimentData = window.nltk_analysis?.sentiment;
        if (!sentimentData) return;

        const sentimentContainer = document.createElement('div');
        sentimentContainer.className = 'analysis-card';
        sentimentContainer.innerHTML = `
            <h4>📊 Resume Sentiment Analysis</h4>
            <div class="sentiment-scores">
                <span class="sentiment-indicator sentiment-positive">
                    Positive: ${(sentimentData.positive * 100).toFixed(1)}%
                </span>
                <span class="sentiment-indicator sentiment-neutral">
                    Neutral: ${(sentimentData.neutral * 100).toFixed(1)}%
                </span>
                <span class="sentiment-indicator sentiment-negative">
                    Negative: ${(sentimentData.negative * 100).toFixed(1)}%
                </span>
            </div>
            <p><strong>Overall Score:</strong> ${sentimentData.compound.toFixed(2)}</p>
        `;

        const resultsSection = document.querySelector('.results-section') || document.body;
        resultsSection.appendChild(sentimentContainer);
    }

    addCopyFunctionality() {
        const copyButtons = document.querySelectorAll('[data-copy]');
        copyButtons.forEach(button => {
            button.addEventListener('click', () => {
                const textToCopy = button.getAttribute('data-copy');
                navigator.clipboard.writeText(textToCopy).then(() => {
                    this.showAlert('Copied to clipboard!', 'success');
                });
            });
        });
    }
}

// Initialize the Resume Analyzer when the page loads
document.addEventListener('DOMContentLoaded', function() {
    window.resumeAnalyzer = new ResumeAnalyzer();
    
    // If results are already present, enhance the display
    if (window.nltk_analysis) {
        setTimeout(() => {
            window.resumeAnalyzer.enhanceResultDisplay();
        }, 100);
    }
});

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}