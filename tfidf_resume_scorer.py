"""
TF-IDF Resume Scorer Module
Fast, lightweight resume ranking using TF-IDF + Cosine Similarity.
No model download required — near-instant results.
Returns the same result dict shape as bert_resume_scorer.py for drop-in compatibility.
"""

import os
import re
import logging
from typing import List, Dict
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ─── Common tech skills for matching ───────────────────────────────────────────
COMMON_SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "kotlin", "swift",
    "react", "angular", "vue", "node", "django", "flask", "fastapi", "spring", "rails",
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "sqlite",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins", "ci/cd",
    "machine learning", "deep learning", "tensorflow", "pytorch", "scikit-learn", "nlp",
    "git", "linux", "rest api", "graphql", "microservices", "agile", "scrum",
    "html", "css", "bootstrap", "tailwind", "figma",
    "excel", "power bi", "tableau", "spark", "hadoop", "kafka",
    "communication", "leadership", "teamwork", "problem solving", "project management"
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_path=None, file_content=None) -> str:
    """Extract text from a PDF file."""
    try:
        if file_path:
            reader = PdfReader(file_path)
        elif file_content:
            reader = PdfReader(file_content)
        else:
            return ""
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        logging.error(f"PDF extraction error: {e}")
        return ""


def extract_candidate_name(text: str) -> str:
    """Heuristic: first line that looks like a name (2-4 capitalized words)."""
    for line in text.split("\n")[:6]:
        line = line.strip()
        words = line.split()
        if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w.isalpha()):
            return line
    return "Unknown"


def extract_skills(text: str, skill_list: List[str] = COMMON_SKILLS) -> List[str]:
    """Extract skills from text by matching against known skill keywords."""
    text_lower = text.lower()
    found = [skill for skill in skill_list if re.search(r'\b' + re.escape(skill) + r'\b', text_lower)]
    return found


def calculate_credibility(text: str) -> Dict:
    """Simple heuristic credibility scorer."""
    text_lower = text.lower()
    score = 50.0

    # Positive signals
    if re.search(r'\b(university|college|bachelor|master|phd|degree)\b', text_lower):
        score += 10
    if re.search(r'\b(certification|certified|license)\b', text_lower):
        score += 8
    if re.search(r'\b(github|linkedin|portfolio)\b', text_lower):
        score += 7
    if re.search(r'\b\d{4}\s*[-–]\s*(\d{4}|present)\b', text_lower):
        score += 10  # Date ranges (employment history)
    if re.search(r'\b(@|email)\b', text_lower):
        score += 5

    # Negative signals (keyword stuffing heuristic)
    word_count = len(text.split())
    if word_count > 50:
        unique_ratio = len(set(text.lower().split())) / word_count
        if unique_ratio < 0.4:
            score -= 15  # Very low unique word ratio → stuffing

    score = max(0.0, min(100.0, score))
    return {"score": score, "word_count": word_count}


def _tfidf_score(resume_text: str, jd_text: str,
                 vectorizer: TfidfVectorizer, jd_vector) -> float:
    """Compute TF-IDF cosine similarity between resume and JD vectors."""
    resume_vector = vectorizer.transform([resume_text])
    sim = cosine_similarity(resume_vector, jd_vector)[0][0]
    return float(sim) * 100.0  # Scale to 0-100


# ─── Main Scoring Function ────────────────────────────────────────────────────

def process_resumes_tfidf(job_description: str,
                          dir_path: str = None,
                          uploaded_files: List = None) -> List[Dict]:
    """
    Process and rank resumes using TF-IDF + Cosine Similarity.
    Near-instant — no model download.
    Returns same result shape as process_resumes_bert().

    Args:
        job_description: Job description text
        dir_path: Directory containing PDF resumes (optional)
        uploaded_files: List of uploaded file objects (optional)

    Returns:
        List of ranked resume dicts (highest score first)
    """
    resumes = []

    # ── Load resumes from directory ──
    if dir_path and os.path.isdir(dir_path):
        logging.info(f"TF-IDF: Loading resumes from {dir_path}")
        for filename in os.listdir(dir_path):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(dir_path, filename)
                text = extract_text_from_pdf(file_path=file_path)
                if text.strip():
                    resumes.append({
                        "text": text,
                        "filename": filename,
                        "candidate_name": extract_candidate_name(text)
                    })

    # ── Load uploaded files ──
    elif uploaded_files:
        logging.info(f"TF-IDF: Processing {len(uploaded_files)} uploaded files")
        for uf in uploaded_files:
            text = extract_text_from_pdf(file_content=uf)
            if text.strip():
                resumes.append({
                    "text": text,
                    "filename": uf.name,
                    "candidate_name": extract_candidate_name(text)
                })

    if not resumes:
        logging.warning("TF-IDF: No resumes found")
        return []

    # ── Fit TF-IDF on all docs (JD + resumes) ──
    all_texts = [job_description] + [r["text"] for r in resumes]
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    jd_vector = tfidf_matrix[0]  # First row is JD

    # Extract JD skills once
    jd_skills = extract_skills(job_description)

    results = []
    for idx, resume in enumerate(resumes):
        # ── Semantic similarity (TF-IDF cosine) ──
        resume_vector = tfidf_matrix[idx + 1]  # +1 because index 0 is JD
        semantic_score = float(cosine_similarity(resume_vector, jd_vector)[0][0]) * 100.0

        # ── Skills match ──
        resume_skills = extract_skills(resume["text"])
        matched = [s for s in jd_skills if s in resume_skills]
        missing = [s for s in jd_skills if s not in resume_skills]
        skills_score = (len(matched) / len(jd_skills) * 100.0) if jd_skills else 50.0

        # ── Credibility ──
        cred = calculate_credibility(resume["text"])
        cred_score = cred["score"]

        # ── Experience heuristic ──
        exp_years = 0
        exp_matches = re.findall(r'(\d+)\+?\s*years?', resume["text"].lower())
        if exp_matches:
            exp_years = max(int(x) for x in exp_matches)
        jd_exp_matches = re.findall(r'(\d+)\+?\s*years?', job_description.lower())
        jd_exp = int(jd_exp_matches[0]) if jd_exp_matches else 3
        exp_score = min(100.0, (exp_years / max(jd_exp, 1)) * 100.0)

        # ── Composite score (same weights as BERT config) ──
        final_score = (
            semantic_score * 0.40 +
            skills_score  * 0.30 +
            exp_score     * 0.20 +
            cred_score    * 0.10
        )

        results.append({
            "candidate_name": resume["candidate_name"],
            "filename": resume["filename"],
            "resume_text": resume["text"][:500] + "..." if len(resume["text"]) > 500 else resume["text"],
            "final_score": round(final_score, 2),
            "breakdown": {
                "final_score": round(final_score, 2),
                "breakdown": {
                    "semantic": round(semantic_score, 2),
                    "skills":   round(skills_score, 2),
                    "experience": round(exp_score, 2),
                    "credibility": round(cred_score, 2),
                }
            },
            "skills_detail": {
                "score": round(skills_score, 2),
                "matched_skills": matched,
                "missing_skills": missing,
                "total_jd_skills": len(jd_skills),
                "total_resume_skills": len(resume_skills)
            },
            "credibility_detail": {
                "score": round(cred_score, 2),
                "word_count": cred["word_count"]
            }
        })

    # ── Sort descending by final_score ──
    results.sort(key=lambda x: x["final_score"], reverse=True)

    # ── Add rank ──
    for i, r in enumerate(results, 1):
        r["rank"] = i

    logging.info(f"TF-IDF ranking complete. Top score: {results[0]['final_score']:.2f}")
    return results


if __name__ == "__main__":
    import time

    jd = """
    Senior Python Developer
    Required: Python 5+ years, Flask, AWS, SQL, Docker.
    Experience: 5+ years backend development.
    """

    resumes_test = [
        {
            "text": "John Doe | john@example.com\nSenior Python Developer 7 years. Flask REST APIs, AWS Lambda, PostgreSQL, Docker.\nEducation: BS Computer Science MIT 2016",
            "filename": "john.pdf",
            "candidate_name": "John Doe"
        },
        {
            "text": "Jane Smith | Python 1 year. Built small web apps with Flask. Student project on SQL.",
            "filename": "jane.pdf",
            "candidate_name": "Jane Smith"
        }
    ]

    # We can't test uploaded_files easily, so call the vectorizer directly
    all_texts = [jd] + [r["text"] for r in resumes_test]
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    jd_vector = tfidf_matrix[0]

    t0 = time.time()
    # Simulate process
    for i, r in enumerate(resumes_test):
        rv = tfidf_matrix[i + 1]
        score = float(cosine_similarity(rv, jd_vector)[0][0]) * 100
        print(f"{r['candidate_name']}: {score:.1f}")
    print(f"TF-IDF done in {time.time()-t0:.3f}s")
