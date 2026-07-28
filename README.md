# 🤖 AI Recruitment & Resume Ranking System

## Overview

The **AI Recruitment & Resume Ranking System** is an enterprise-grade hiring platform built with Python, Streamlit, PyTorch, Sentence-Transformers, and Groq LLMs.

It provides recruiters with high-accuracy candidate ranking (using dual BERT / TF-IDF models), resume authenticity & fraud detection analysis, fast AI candidate feedback generation, user authentication, and automated email communications.

---

## 🌟 Key Features

### 1. 🏆 Dual-Model Resume Ranking
- **🤖 BERT Mode (`Sentence-BERT / all-mpnet-base-v2`)**: Deep semantic representation matching between job descriptions and resumes across semantic similarity, technical skills match, experience relevance, and credibility.
- **⚡ TF-IDF Mode (`Scikit-Learn TF-IDF Vectorizer`)**: Near-instant keyword-frequency ranking with zero model download overhead.
- **Explainability Engine**: Detailed per-candidate score breakdowns (0–100%) and skill match analysis.

### 2. 🛡️ Fraud Detection & Credibility Dashboard
- **Risk Assessment**: Color-coded risk classification (Low, Medium, High Risk).
- **Interactive Visualizations**:
  - **Fraud Risk Distribution**: Donut pie charts displaying candidate risk breakdown.
  - **Multi-Score Comparison**: Grouped bar charts comparing Match Score, Skills Match, and Credibility across candidates.
  - **Fraud Risk Map**: Interactive scatter plots comparing match score against credibility.
  - **Skill Radar & Gauges**: Multi-dimensional candidate radar charts and individual authenticity gauges.
- **Fraud Indicators**: Flags keyword stuffing, skill exaggeration, experience consistency, and education verifiability.

### 3. 📝 Ultra-Fast AI Feedback Generation
- Powered by **Direct Groq SDK (`llama-3.1-8b-instant`)** for high-throughput feedback generation without agent overhead (~1-2 seconds per candidate).
- Real-time progress bar with per-candidate status updates.
- Generates tailored feedback reports for both shortlisted and rejected candidates including strengths, growth areas, actionable suggestions, and next steps.
- Exportable plain-text feedback reports.

### 4. 📧 Automated Candidate Communication
- Automated email generation (interview invitations, rejection updates, hiring updates).
- SMTP integration and simulated email delivery.

### 5. 🔐 Multi-User Authentication System
- Secure user authentication with encrypted password hashing (`SHA-256` with salting).
- Persistent JSON user storage supporting recruiter login and new user registration.

---

## 📁 Project Architecture

```text
ai-recruitment-system/
├── ats/                         # ATS Core Engine
│   ├── __init__.py
│   ├── ats_pipeline.py          # Orchestrator for BERT-based ATS workflow
│   ├── explainability.py        # Score breakdown & explanation generator
│   └── skill_extractor.py       # Technical & soft skill extraction engine
├── core/                        # Core ML & Math Modules
│   ├── __init__.py
│   ├── bert_embedder.py         # Sentence-Transformers embedding wrapper
│   ├── credibility_scorer.py    # Resume text quality & keyword stuffing heuristics
│   ├── scoring_engine.py        # Composite weighting engine
│   ├── similarity_engine.py     # Cosine similarity calculations
│   └── text_preprocessor.py     # Text cleaning & section parsing
├── Logs/                        # System Logs
│   └── app.log
├── app.py                       # Main Streamlit Dashboard UI
├── auth_utils.py                # User Authentication & Security
├── bert_resume_scorer.py        # BERT-based Scorer Interface
├── tfidf_resume_scorer.py       # TF-IDF-based Scorer Interface
├── feedback_generator.py        # Direct Groq SDK Fast Feedback Engine
├── email_automation.py          # Email Generation & Delivery
├── top_resume_selector.py       # Candidate Selection Agent
├── config.py                    # System & Scoring Weight Configurations
├── users.json                   # User Database Storage
├── requirements.txt             # Python Package Dependencies
├── .env.example                 # Environment Variable Template
└── README.md                    # System Documentation
```

---

## ⚡ Scoring Architecture & Weights

The system computes candidate scores using weighted component evaluation:

$$\text{Final Score} = (S_{\text{semantic}} \times 0.40) + (S_{\text{skills}} \times 0.30) + (S_{\text{experience}} \times 0.20) + (S_{\text{credibility}} \times 0.10)$$

- **Semantic Similarity (40%)**: Cosine similarity between resume and job description vectors.
- **Skills Match (30%)**: Ratio of matched vs required technical and domain skills.
- **Experience Relevance (20%)**: Extracted experience section alignment.
- **Credibility (10%)**: Document structure, certification presence, and anti-keyword-stuffing verification.

---

## 🛠️ Installation & Setup Guide

### 1. Prerequisites
- Python 3.9+ installed
- Git installed
- Groq API Key (Get one free at [console.groq.com](https://console.groq.com/))

### 2. Clone Repository
```bash
git clone https://github.com/Ravulaponnakrishnavamsi/AI-BASED-RESUME-RANKING-SYSTEM.git
cd AI-BASED-RESUME-RANKING-SYSTEM
```

### 3. Create & Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_optional
```

---

## 🚀 Running the Application

Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### Demo Credentials
- **Email:** `recruiter@ai.com`
- **Password:** `demo123`

---

## 📋 Technology Stack

- **Frontend / UI:** Streamlit, Streamlit Extras
- **NLP / ML:** PyTorch, Sentence-Transformers (`all-mpnet-base-v2`), Scikit-Learn
- **LLM / Inference:** Groq Cloud API (`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`)
- **Data & Plotting:** Pandas, Plotly Express, Plotly Graph Objects
- **Document Processing:** PyPDF2, Regex

---

