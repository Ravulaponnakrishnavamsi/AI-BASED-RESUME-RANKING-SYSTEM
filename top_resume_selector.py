from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os
import logging
import json
import re

load_dotenv()
logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=2000
)

top_resume_selector = Agent(
    role="Top Resume Selector Agent",
    goal="Deeply analyze resumes and identify the top 3 candidates that best match the job description with detailed insights",
    backstory="An expert AI recruiter with deep understanding of candidate qualifications, skills matching, and experience evaluation. Specializes in fair, unbiased candidate assessment.",
    llm=llm,
    verbose=True,
    allow_delegation=False
)

def extract_text_from_pdf(file_path=None, file_content=None):
    """Extract text from PDF resume"""
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

def extract_candidate_name(resume_text):
    """Extract candidate name from resume text (simple heuristic)"""
    lines = resume_text.split('\n')
    # Usually name is in first few lines
    for line in lines[:5]:
        line = line.strip()
        # Simple heuristic: name is usually 2-4 words, capitalized
        words = line.split()
        if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
            return line
    return "Unknown Candidate"

def process_resumes_for_selection(job_description, dir_path=None, uploaded_files=None):
    """Process all resumes and prepare data for analysis"""
    resumes_data = []
    
    if dir_path and os.path.isdir(dir_path):
        for filename in os.listdir(dir_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(dir_path, filename)
                resume_text = extract_text_from_pdf(file_path=file_path)
                candidate_name = extract_candidate_name(resume_text)
                
                resumes_data.append({
                    "file_name": filename,
                    "file_path": file_path,
                    "candidate_name": candidate_name,
                    "resume_text": resume_text
                })
    
    elif uploaded_files:
        for uploaded_file in uploaded_files:
            resume_text = extract_text_from_pdf(file_content=uploaded_file)
            candidate_name = extract_candidate_name(resume_text)
            
            # For uploaded files, we'll use a temporary path representation
            temp_path = f"uploaded/{uploaded_file.name}"
            
            resumes_data.append({
                "file_name": uploaded_file.name,
                "file_path": temp_path,
                "candidate_name": candidate_name,
                "resume_text": resume_text
            })
    
    return resumes_data

def create_top_resume_task(job_description, dir_path=None, uploaded_files=None):
    """Create task to select and analyze top 3 resumes"""
    resumes_data = process_resumes_for_selection(job_description, dir_path, uploaded_files)
    
    if not resumes_data:
        return None
    
    # Build comprehensive prompt for AI analysis
    resumes_summary = []
    for idx, resume in enumerate(resumes_data, 1):
        resumes_summary.append(
            f"Resume {idx}:\n"
            f"File: {resume['file_name']}\n"
            f"Candidate: {resume['candidate_name']}\n"
            f"Content: {resume['resume_text'][:1500]}...\n"  # Truncate for token limits
        )
    
    prompt = f"""
You are an expert AI recruiter. Analyze the following {len(resumes_data)} resumes for this job description:

JOB DESCRIPTION:
{job_description}

RESUMES TO ANALYZE:
{chr(10).join(resumes_summary)}

YOUR TASK:
1. Carefully analyze each resume against the job requirements
2. Score each candidate on a scale of 0-100 based on:
   - Skills match (40 points)
   - Experience relevance and years (30 points)
   - Education and certifications (15 points)
   - Projects and achievements (15 points)
3. Select the TOP 3 candidates with highest scores
4. For each top candidate, provide:
   - Overall match score (0-100)
   - Candidate name
   - Years of experience
   - Key strengths (3-5 bullet points)
   - Skills that match the job requirements
   - Brief match summary explaining why they're a good fit

IMPORTANT:
- Be fair and unbiased
- Focus on qualifications, not demographics
- Rank in DESCENDING order (best first)
- Provide actionable insights

OUTPUT FORMAT (return ONLY valid JSON):
{{
    "top_candidates": [
        {{
            "rank": 1,
            "file_name": "filename.pdf",
            "file_path": "/path/to/file",
            "score": 95,
            "candidate_name": "John Doe",
            "experience_years": 10,
            "key_strengths": ["strength 1", "strength 2", "strength 3"],
            "match_summary": "Brief explanation of why this is a great match",
            "skills_matched": ["skill1", "skill2", "skill3"]
        }}
    ],
    "all_candidates": [
        {{
            "candidate_name": "All candidates including rejected ones",
            "file_name": "filename.pdf",
            "score": 85,
            "is_shortlisted": true
        }}
    ]
}}
"""
    
    return Task(
        description=prompt,
        agent=top_resume_selector,
        expected_output="A JSON object with top 3 ranked candidates and detailed analysis for each, plus all candidates list"
    ), resumes_data

def parse_ai_response(ai_output, resumes_data):
    """Parse AI response and ensure proper formatting"""
    try:
        # Try to extract JSON from the response
        output_str = str(ai_output)
        
        # Find JSON in the output
        json_match = re.search(r'\{.*\}', output_str, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            
            # Ensure file paths are correct
            for candidate in result.get("top_candidates", []):
                # Find matching resume data
                for resume in resumes_data:
                    if resume["file_name"] == candidate.get("file_name"):
                        candidate["file_path"] = resume["file_path"]
                        break
            
            # Build all_candidates list if not present
            if "all_candidates" not in result:
                result["all_candidates"] = []
                top_files = [c["file_name"] for c in result.get("top_candidates", [])]
                
                for resume in resumes_data:
                    is_shortlisted = resume["file_name"] in top_files
                    result["all_candidates"].append({
                        "candidate_name": resume["candidate_name"],
                        "file_name": resume["file_name"],
                        "file_path": resume["file_path"],
                        "resume_text": resume["resume_text"],
                        "is_shortlisted": is_shortlisted,
                        "score": 0  # Will be filled by feedback generator
                    })
            
            return result
        else:
            raise ValueError("No JSON found in AI response")
    
    except Exception as e:
        logging.error(f"Error parsing AI response: {str(e)}")
        # Return a fallback structure
        return {
            "top_candidates": [],
            "all_candidates": [],
            "error": str(e)
        }

if __name__ == "__main__":
    # Test the module
    test_jd = "Senior Python Developer with 5+ years experience in Flask, AWS, and SQL"
    task, resumes_data = create_top_resume_task(test_jd, dir_path="./test_resumes")
    
    if task:
        crew = Crew(agents=[top_resume_selector], tasks=[task], verbose=True)
        result = crew.kickoff()
        parsed_result = parse_ai_response(result, resumes_data)
        print(json.dumps(parsed_result, indent=2))
