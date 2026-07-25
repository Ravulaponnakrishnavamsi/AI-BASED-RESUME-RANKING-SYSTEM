"""
BERT Resume Scorer Module
Replaces the LLM-based resume_ranker.py with BERT-based scoring
"""

import os
import logging
from typing import List, Dict
from PyPDF2 import PdfReader
from ats.ats_pipeline import ATSPipeline
from ats.explainability import ExplainabilityEngine
from core.text_preprocessor import TextPreprocessor
from config import BERT_MODEL, SCORING_WEIGHTS

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# Initialize BERT ATS Pipeline
ats_pipeline = ATSPipeline(bert_model=BERT_MODEL, weights=SCORING_WEIGHTS)
explainer = ExplainabilityEngine()
preprocessor = TextPreprocessor()


def extract_text_from_pdf(file_path=None, file_content=None) -> str:
    """
    Extract text from PDF file
    
    Args:
        file_path: Path to PDF file (for directory-based processing)
        file_content: File content object (for uploaded files)
        
    Returns:
        Extracted text
    """
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
        logging.error(f"Error extracting PDF text: {str(e)}")
        return ""


def process_resumes_bert(job_description: str, dir_path: str = None, 
                         uploaded_files: List = None) -> List[Dict]:
    """
    Process and rank resumes using BERT-based ATS pipeline
    
    Args:
        job_description: Job description text
        dir_path: Directory containing resume PDFs (optional)
        uploaded_files: List of uploaded file objects (optional)
        
    Returns:
        List of ranked resume results
    """
    resumes = []
    
    # Process from directory
    if dir_path and os.path.isdir(dir_path):
        logging.info(f"Processing resumes from directory: {dir_path}")
        for filename in os.listdir(dir_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(dir_path, filename)
                resume_text = extract_text_from_pdf(file_path=file_path)
                
                if resume_text:
                    candidate_name = preprocessor.extract_candidate_name(resume_text)
                    resumes.append({
                        'text': resume_text,
                        'filename': filename,
                        'candidate_name': candidate_name
                    })
                    logging.info(f"Loaded resume: {filename}")
    
    # Process uploaded files
    elif uploaded_files:
        logging.info(f"Processing {len(uploaded_files)} uploaded resumes")
        for uploaded_file in uploaded_files:
            resume_text = extract_text_from_pdf(file_content=uploaded_file)
            
            if resume_text:
                candidate_name = preprocessor.extract_candidate_name(resume_text)
                resumes.append({
                    'text': resume_text,
                    'filename': uploaded_file.name,
                    'candidate_name': candidate_name
                })
                logging.info(f"Loaded resume: {uploaded_file.name}")
    
    if not resumes:
        logging.warning("No resumes found to process")
        return []
    
    # Rank resumes using BERT pipeline
    logging.info(f"Ranking {len(resumes)} resumes with BERT")
    ranked_results = ats_pipeline.rank_resumes(resumes, job_description)
    
    logging.info(f"Ranking complete. Top score: {ranked_results[0]['final_score']:.2f}")
    
    return ranked_results


def get_top_candidates(job_description: str, dir_path: str = None, 
                      uploaded_files: List = None, n: int = 3) -> List[Dict]:
    """
    Get top N candidates using BERT-based ranking
    
    Args:
        job_description: Job description text
        dir_path: Directory path (optional)
        uploaded_files: Uploaded files (optional)
        n: Number of top candidates to return
        
    Returns:
        Top N ranked candidates with explanations
    """
    all_results = process_resumes_bert(job_description, dir_path, uploaded_files)
    
    if not all_results:
        return []
    
    top_n = all_results[:n]
    
    # Add explanations
    for result in top_n:
        result['explanation'] = explainer.generate_explanation(result)
    
    return top_n


def generate_ranking_report(job_description: str, dir_path: str = None, 
                           uploaded_files: List = None) -> str:
    """
    Generate a comprehensive ranking report
    
    Args:
        job_description: Job description text
        dir_path: Directory path (optional)
        uploaded_files: Uploaded files (optional)
        
    Returns:
        Markdown-formatted ranking report
    """
    results = process_resumes_bert(job_description, dir_path, uploaded_files)
    
    if not results:
        return "**No resumes found to rank.**"
    
    # Generate summary
    summary = explainer.generate_ranking_summary(results, top_n=10)
    
    # Add detailed explanations for top 3
    summary += "\n\n---\n\n## Detailed Analysis - Top 3 Candidates\n\n"
    
    for result in results[:3]:
        summary += f"### {result['rank']}. {result['candidate_name']}\n\n"
        summary += explainer.generate_explanation(result)
        summary += "\n\n---\n\n"
    
    return summary


if __name__ == "__main__":
    # Test the BERT resume scorer
    test_jd = """
    Senior Python Developer
    
    Required Skills:
    - Python (5+ years)
    - Flask or Django
    - AWS cloud
    - SQL databases
    - Docker
    
    Experience: 5+ years backend development
    """
    
    # Test with sample resume
    test_resume = """
    John Doe
    john@example.com | (555) 123-4567
    
    Experience:
    Senior Python Developer at TechCorp (2018-2023)
    - Built REST APIs with Flask
    - Deployed on AWS
    - Managed PostgreSQL databases
    
    Skills: Python, Flask, AWS, SQL, Docker, Git
    
    Education:
    BS Computer Science, MIT
    """
    
    test_data = [{
        'text': test_resume,
        'filename': 'john_doe.pdf',
        'candidate_name': 'John Doe'
    }]
    
    # Use the ATS pipeline directly for testing
    from ats.ats_pipeline import ATSPipeline
    pipeline = ATSPipeline(bert_model="all-MiniLM-L6-v2")
    results = pipeline.rank_resumes(test_data, test_jd)
    
    print(f"Test Result:")
    print(f"Candidate: {results[0]['candidate_name']}")
    print(f"Final Score: {results[0]['final_score']:.2f}")
    print(f"Breakdown: {results[0]['breakdown']['breakdown']}")
