"""
Skill Extractor Module
Extract and match skills using BERT + pattern matching
"""

import re
from typing import List, Set, Dict
import numpy as np


class SkillExtractor:
    """
    Extract skills and calculate match scores
    """
    
    # Common technical skills (expandable list)
    COMMON_SKILLS = {
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'rust',
        'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'redis',
        'aws', 'azure', 'gcp', 'cloud', 'devops',
        'docker', 'kubernetes', 'k8s', 'terraform', 'ansible',
        'react', 'angular', 'vue', 'node.js', 'express',
        'flask', 'django', 'fastapi', 'spring', 'asp.net',
        'machine learning', 'deep learning', 'nlp', 'computer vision', 'data science',
        'tensorflow', 'pytorch', 'scikit-learn', 'keras',
        'git', 'github', 'gitlab', 'ci/cd', 'jenkins',
        'agile', 'scrum', 'kanban', 'jira',
        'rest api', 'graphql', 'microservices', 'api',
        'html', 'css', 'sass', 'bootstrap', 'tailwind',
        'redis', 'rabbitmq', 'kafka', 'elasticsearch',
        'linux', 'unix', 'windows', 'bash', 'powershell'
    }
    
    @staticmethod
    def extract_skills(text: str) -> Set[str]:
        """
        Extract skills from text using pattern matching
        
        Args:
            text: Resume or JD text
            
        Returns:
            Set of skills found
        """
        if not text:
            return set()
        
        text_lower = text.lower()
        found_skills = set()
        
        # Find common skills
        for skill in SkillExtractor.COMMON_SKILLS:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.add(skill)
        
        # Extract from "Skills:" section if present
        skills_section = re.search(
            r'(skills|technical skills|core competencies):?\s*(.*?)(?:\n\n|experience|education|$)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        
        if skills_section:
            skills_text = skills_section.group(2)
            # Split by common delimiters
            skill_tokens = re.split(r'[,;|\n•·]', skills_text)
            for token in skill_tokens:
                cleaned = token.strip().lower()
                # Add if it's a reasonable length and not just numbers
                if 2 < len(cleaned) < 30 and not cleaned.isdigit():
                    found_skills.add(cleaned)
        
        return found_skills
    
    def calculate_skill_match(self, resume_text: str, jd_text: str, 
                             embedder) -> Dict:
        """
        Calculate skill match score using both exact match and semantic similarity
        
        Args:
            resume_text: Resume text
            jd_text: Job description text
            embedder: BERTEmbedder instance
            
        Returns:
            Dictionary containing:
                - score: combined skill match score (0-100)
                - matched_skills: list of matched skills
                - missing_skills: list of missing required skills
                - semantic_skill_score: BERT similarity of skills
                - exact_match_score: exact overlap score
        """
        resume_skills = self.extract_skills(resume_text)
        jd_skills = self.extract_skills(jd_text)
        
        # If no skills detected in JD, return neutral score
        if not jd_skills:
            return {
                'score': 75.0,
                'matched_skills': [],
                'missing_skills': [],
                'semantic_skill_score': 75.0,
                'exact_match_score': 75.0
            }
        
        # Exact match calculation
        matched = resume_skills & jd_skills
        missing = jd_skills - resume_skills
        
        exact_match_ratio = len(matched) / len(jd_skills) if jd_skills else 0.0
        exact_match_score = exact_match_ratio * 100
        
        # Semantic similarity of skill embeddings
        semantic_score = 50.0  # Default
        
        if resume_skills and jd_skills:
            # Create skill text for embedding
            resume_skill_text = ", ".join(sorted(resume_skills))
            jd_skill_text = ", ".join(sorted(jd_skills))
            
            try:
                resume_skill_emb = embedder.embed_text(resume_skill_text)
                jd_skill_emb = embedder.embed_text(jd_skill_text)
                
                # Import here to avoid circular dependency
                from core.similarity_engine import SimilarityEngine
                semantic_score = SimilarityEngine.cosine_similarity_score(
                    resume_skill_emb, jd_skill_emb
                )
            except Exception as e:
                # Fallback to exact match if embedding fails
                semantic_score = exact_match_score
        
        # Combined score (70% exact match, 30% semantic)
        # This gives more weight to exact matches while still considering semantic similarity
        combined_score = (exact_match_score * 0.7) + (semantic_score * 0.3)
        
        return {
            'score': round(combined_score, 2),
            'matched_skills': sorted(list(matched)),
            'missing_skills': sorted(list(missing)),
            'semantic_skill_score': round(semantic_score, 2),
            'exact_match_score': round(exact_match_score, 2)
        }


if __name__ == "__main__":
    # Test the skill extractor
    extractor = SkillExtractor()
    
    resume_text = """
    John Doe
    Skills: Python, Flask, AWS, SQL, Docker, React
    
    Experience:
    Built REST APIs using Python and Flask
    Deployed on AWS with Docker containers
    """
    
    jd_text = """
    Job: Senior Backend Developer
    Required Skills: Python, Flask, AWS, SQL, Kubernetes
    """
    
    resume_skills = extractor.extract_skills(resume_text)
    jd_skills = extractor.extract_skills(jd_text)
    
    print(f"Resume skills: {resume_skills}")
    print(f"JD skills: {jd_skills}")
    print(f"Matched: {resume_skills & jd_skills}")
    print(f"Missing: {jd_skills - resume_skills}")
