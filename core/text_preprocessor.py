"""
Text Preprocessor Module
Clean and normalize text for BERT embedding
"""

import re
from typing import Dict, List


class TextPreprocessor:
    """
    Clean and normalize text for BERT embedding
    """
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Remove noise and normalize text
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s.,!?@\-()]', '', text)
        
        # Normalize case (BERT handles case, but normalize for display)
        text = text.strip()
        
        return text
    
    @staticmethod
    def extract_sections(resume_text: str) -> Dict[str, str]:
        """
        Extract key sections from resume
        
        Args:
            resume_text: Full resume text
            
        Returns:
            Dictionary with keys: experience, skills, education, summary
        """
        sections = {
            "experience": "",
            "skills": "",
            "education": "",
            "summary": ""
        }
        
        if not resume_text:
            return sections
        
        # Simple keyword-based extraction (case-insensitive)
        text_lower = resume_text.lower()
        
        # Experience section
        exp_match = re.search(
            r'(experience|work history|employment)(.*?)(education|skills|$)',
            resume_text,
            re.IGNORECASE | re.DOTALL
        )
        if exp_match:
            sections["experience"] = exp_match.group(2).strip()
        
        # Skills section
        skill_match = re.search(
            r'(skills|technical skills|core competencies)(.*?)(education|experience|$)',
            resume_text,
            re.IGNORECASE | re.DOTALL
        )
        if skill_match:
            sections["skills"] = skill_match.group(2).strip()
        
        # Education section
        edu_match = re.search(
            r'(education)(.*?)(experience|skills|$)',
            resume_text,
            re.IGNORECASE | re.DOTALL
        )
        if edu_match:
            sections["education"] = edu_match.group(2).strip()
        
        # Summary/Objective section (often at the beginning)
        summary_match = re.search(
            r'(summary|objective|profile)(.*?)(experience|skills|education|$)',
            resume_text,
            re.IGNORECASE | re.DOTALL
        )
        if summary_match:
            sections["summary"] = summary_match.group(2).strip()
        
        return sections
    
    @staticmethod
    def extract_candidate_name(resume_text: str) -> str:
        """
        Extract candidate name from resume (simple heuristic)
        
        Args:
            resume_text: Full resume text
            
        Returns:
            Candidate name or "Unknown"
        """
        if not resume_text:
            return "Unknown"
        
        # Simple heuristic: first non-empty line that looks like a name
        lines = resume_text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            # Name pattern: 2-4 capitalized words, 2-30 chars each
            name_pattern = r'^([A-Z][a-z]{1,30}\s){1,3}[A-Z][a-z]{1,30}$'
            if re.match(name_pattern, line):
                return line
        
        return "Unknown"


if __name__ == "__main__":
    # Test the preprocessor
    sample_resume = """
    John Doe
    john@example.com | (555) 123-4567
    
    Summary:
    Experienced Python developer with strong backend skills.
    
    Experience:
    Senior Python Developer at TechCorp (2018-2023)
    - Built REST APIs
    - Deployed on AWS
    
    Skills: Python, Flask, AWS, SQL
    
    Education:
    BS Computer Science, MIT
    """
    
    preprocessor = TextPreprocessor()
    cleaned = preprocessor.clean_text(sample_resume)
    sections = preprocessor.extract_sections(sample_resume)
    name = preprocessor.extract_candidate_name(sample_resume)
    
    print(f"Candidate: {name}")
    print(f"Skills: {sections['skills'][:50]}...")
    print(f"Experience: {sections['experience'][:50]}...")
