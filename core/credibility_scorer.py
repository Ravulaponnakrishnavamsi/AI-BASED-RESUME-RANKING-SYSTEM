"""
Credibility Scorer Module
Assess resume quality based on completeness and formatting
"""

import re
from typing import Dict, List


class CredibilityScorer:
    """
    Assess resume quality based on completeness and formatting
    """
    
    @staticmethod
    def calculate_credibility_score(resume_text: str, 
                                    extracted_sections: Dict[str, str]) -> Dict:
        """
        Calculate credibility score (0-100) based on resume quality
        
        Args:
            resume_text: Full resume text
            extracted_sections: Dictionary with experience, skills, education, summary
            
        Returns:
            Dictionary containing:
                - score: overall credibility score
                - breakdown: component scores
                - flags: list of quality issues
        """
        scores = {
            'formatting': 0.0,
            'completeness': 0.0,
            'education': 0.0,
            'contact_info': 0.0
        }
        
        flags = []
        
        if not resume_text:
            return {
                'score': 0.0,
                'breakdown': scores,
                'flags': ['Empty resume']
            }
        
        # 1. Formatting quality (30%)
        has_multiple_sections = len([s for s in extracted_sections.values() if s]) >= 2
        reasonable_length = 500 < len(resume_text) < 10000
        
        formatting_score = 0
        if has_multiple_sections:
            formatting_score += 50
        else:
            flags.append("Missing key sections")
        
        if reasonable_length:
            formatting_score += 50
        elif len(resume_text) < 500:
            flags.append("Resume too short")
        else:
            flags.append("Resume too long")
        
        scores['formatting'] = formatting_score
        
        # 2. Completeness (30%)
        required_sections = ['experience', 'skills', 'education']
        has_all_sections = all([extracted_sections.get(s) for s in required_sections])
        
        if has_all_sections:
            scores['completeness'] = 100
        else:
            missing = [s for s in required_sections if not extracted_sections.get(s)]
            scores['completeness'] = 50
            flags.append(f"Missing sections: {', '.join(missing)}")
        
        # 3. Education verification (20%)
        edu_keywords = r'(bachelor|master|phd|doctorate|degree|university|college|b\.s\.|m\.s\.|b\.a\.|m\.a\.)'
        has_education = bool(re.search(edu_keywords, resume_text, re.IGNORECASE))
        
        if has_education:
            scores['education'] = 100
        else:
            scores['education'] = 40
            flags.append("No education information found")
        
        # 4. Contact info quality (20%)
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        
        has_email = bool(re.search(email_pattern, resume_text))
        has_phone = bool(re.search(phone_pattern, resume_text))
        
        contact_score = 0
        if has_email:
            contact_score += 50
        else:
            flags.append("Missing email address")
        
        if has_phone:
            contact_score += 50
        else:
            flags.append("Missing phone number")
        
        scores['contact_info'] = contact_score
        
        # Calculate weighted average
        final_score = (
            scores['formatting'] * 0.3 +
            scores['completeness'] * 0.3 +
            scores['education'] * 0.2 +
            scores['contact_info'] * 0.2
        )
        
        return {
            'score': round(final_score, 2),
            'breakdown': scores,
            'flags': flags
        }


if __name__ == "__main__":
    # Test the credibility scorer
    scorer = CredibilityScorer()
    
    # Test with a good resume
    good_resume = """
    John Doe
    john.doe@example.com | (555) 123-4567
    
    Summary:
    Experienced developer with strong skills.
    
    Experience:
    Senior Developer at TechCorp (2018-2023)
    - Built REST APIs
    
    Skills: Python, Flask, AWS
    
    Education:
    Bachelor of Science in Computer Science, MIT, 2018
    """
    
    from core.text_preprocessor import TextPreprocessor
    preprocessor = TextPreprocessor()
    sections = preprocessor.extract_sections(good_resume)
    
    result = scorer.calculate_credibility_score(good_resume, sections)
    
    print(f"Credibility Score: {result['score']}")
    print(f"Breakdown: {result['breakdown']}")
    print(f"Flags: {result['flags']}")
    
    # Test with a poor resume
    poor_resume = "John Doe. Some experience."
    poor_sections = preprocessor.extract_sections(poor_resume)
    poor_result = scorer.calculate_credibility_score(poor_resume, poor_sections)
    
    print(f"\nPoor Resume Score: {poor_result['score']}")
    print(f"Flags: {poor_result['flags']}")
