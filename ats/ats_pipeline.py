"""
ATS Pipeline Module
Main orchestrator for the BERT-based ATS workflow
"""

from typing import List, Dict, Optional
import logging
from core.bert_embedder import BERTEmbedder
from core.text_preprocessor import TextPreprocessor
from core.similarity_engine import SimilarityEngine
from core.scoring_engine import ScoringEngine
from core.credibility_scorer import CredibilityScorer
from ats.skill_extractor import SkillExtractor
import numpy as np

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class ATSPipeline:
    """
    Main ATS pipeline orchestrator
    Coordinates all BERT-based scoring components
    """
    
    def __init__(self, bert_model: str = "all-mpnet-base-v2", 
                 weights: Dict[str, float] = None):
        """
        Initialize the ATS pipeline
        
        Args:
            bert_model: Sentence-BERT model name
            weights: Custom scoring weights (optional)
        """
        logging.info(f"Initializing ATS Pipeline with model: {bert_model}")
        
        self.embedder = BERTEmbedder(model_name=bert_model)
        self.preprocessor = TextPreprocessor()
        self.similarity_engine = SimilarityEngine()
        self.scoring_engine = ScoringEngine(weights=weights)
        self.credibility_scorer = CredibilityScorer()
        self.skill_extractor = SkillExtractor()
        
        logging.info("ATS Pipeline initialized successfully")
    
    def process_single_resume(self, resume_text: str, jd_text: str, 
                             jd_embedding: Optional[np.ndarray] = None,
                             candidate_name: str = None) -> Dict:
        """
        Process a single resume against job description
        
        Args:
            resume_text: Full resume text
            jd_text: Job description text
            jd_embedding: Pre-computed JD embedding (optional, for efficiency)
            candidate_name: Candidate name (optional)
            
        Returns:
            Dictionary containing:
                - resume_text: truncated resume text
                - candidate_name: extracted or provided name
                - final_score: composite score
                - breakdown: detailed score breakdown
                - skills_detail: skill matching details
                - credibility_detail: resume quality details
        """
        logging.info(f"Processing resume for candidate: {candidate_name or 'Unknown'}")
        
        # Preprocess text
        clean_resume = self.preprocessor.clean_text(resume_text)
        clean_jd = self.preprocessor.clean_text(jd_text)
        
        # Extract candidate name if not provided
        if not candidate_name:
            candidate_name = self.preprocessor.extract_candidate_name(clean_resume)
        
        # Extract sections
        resume_sections = self.preprocessor.extract_sections(clean_resume)
        
        # Generate embeddings
        resume_embedding = self.embedder.embed_resume(clean_resume)
        if jd_embedding is None:
            jd_embedding = self.embedder.embed_jd(clean_jd)
        
        # Component 1: Semantic similarity
        semantic_score = self.similarity_engine.cosine_similarity_score(
            resume_embedding, jd_embedding
        )
        logging.info(f"  Semantic score: {semantic_score:.2f}")
        
        # Component 2: Skills match
        skills_result = self.skill_extractor.calculate_skill_match(
            resume_text=clean_resume,
            jd_text=clean_jd,
            embedder=self.embedder
        )
        skills_score = skills_result['score']
        logging.info(f"  Skills score: {skills_score:.2f}")
        
        # Component 3: Experience relevance
        experience_score = self._calculate_experience_score(
            resume_sections['experience'], clean_jd, jd_embedding
        )
        logging.info(f"  Experience score: {experience_score:.2f}")
        
        # Component 4: Credibility
        credibility_result = self.credibility_scorer.calculate_credibility_score(
            clean_resume, resume_sections
        )
        credibility_score = credibility_result['score']
        logging.info(f"  Credibility score: {credibility_score:.2f}")
        
        # Calculate composite score
        component_scores = {
            'semantic': semantic_score,
            'skills': skills_score,
            'experience': experience_score,
            'credibility': credibility_score
        }
        
        final_result = self.scoring_engine.calculate_composite_score(component_scores)
        logging.info(f"  Final score: {final_result['final_score']:.2f}")
        
        return {
            'resume_text': resume_text[:500] + "..." if len(resume_text) > 500 else resume_text,
            'candidate_name': candidate_name,
            'final_score': final_result['final_score'],
            'breakdown': final_result,
            'skills_detail': skills_result,
            'credibility_detail': credibility_result
        }
    
    def _calculate_experience_score(self, experience_text: str, 
                                    jd_text: str, jd_embedding: np.ndarray) -> float:
        """
        Calculate experience relevance score
        
        Args:
            experience_text: Extracted experience section
            jd_text: Job description text
            jd_embedding: JD embedding vector
            
        Returns:
            Experience relevance score (0-100)
        """
        if not experience_text or not experience_text.strip():
            # No experience section found, return default score
            logging.warning("No experience section found in resume")
            return 50.0
        
        # Generate embedding for experience section
        exp_embedding = self.embedder.embed_text(experience_text)
        
        # Calculate similarity with JD
        score = self.similarity_engine.cosine_similarity_score(exp_embedding, jd_embedding)
        
        return score
    
    def rank_resumes(self, resumes: List[Dict[str, str]], jd_text: str) -> List[Dict]:
        """
        Rank multiple resumes against a job description
        
        Args:
            resumes: List of dictionaries with keys:
                - 'text': resume text (required)
                - 'filename': resume filename (optional)
                - 'candidate_name': candidate name (optional)
            jd_text: Job description text
            
        Returns:
            Sorted list of scored resumes (highest score first)
        """
        logging.info(f"Ranking {len(resumes)} resumes")
        
        # Generate JD embedding once (efficiency optimization)
        clean_jd = self.preprocessor.clean_text(jd_text)
        jd_embedding = self.embedder.embed_jd(clean_jd)
        
        results = []
        for idx, resume_data in enumerate(resumes, 1):
            logging.info(f"Processing resume {idx}/{len(resumes)}")
            
            result = self.process_single_resume(
                resume_text=resume_data['text'],
                jd_text=clean_jd,
                jd_embedding=jd_embedding,
                candidate_name=resume_data.get('candidate_name')
            )
            
            # Add metadata
            result['filename'] = resume_data.get('filename', 'Unknown')
            if not result.get('candidate_name') or result['candidate_name'] == 'Unknown':
                result['candidate_name'] = resume_data.get('filename', 'Unknown').replace('.pdf', '')
            
            results.append(result)
        
        # Sort by final score (descending)
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Add rank
        for idx, result in enumerate(results, 1):
            result['rank'] = idx
        
        logging.info(f"Ranking complete. Top score: {results[0]['final_score']:.2f}")
        
        return results
    
    def get_top_n(self, resumes: List[Dict[str, str]], jd_text: str, n: int = 3) -> List[Dict]:
        """
        Get top N candidates
        
        Args:
            resumes: List of resume dictionaries
            jd_text: Job description
            n: Number of top candidates to return
            
        Returns:
            Top N scored resumes
        """
        all_results = self.rank_resumes(resumes, jd_text)
        return all_results[:n]
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update scoring weights
        
        Args:
            new_weights: New weight dictionary
        """
        self.scoring_engine.set_weights(new_weights)
        logging.info(f"Updated scoring weights: {new_weights}")
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current scoring weights"""
        return self.scoring_engine.get_weights()


if __name__ == "__main__":
    # Test the ATS pipeline
    pipeline = ATSPipeline(bert_model="all-MiniLM-L6-v2")
    
    # Sample data
    resume = """
    John Doe
    john.doe@email.com | (555) 123-4567
    
    Summary:
    Experienced Python developer with 5 years in backend development
    
    Experience:
    Senior Python Developer at TechCorp (2018-2023)
    - Built REST APIs using Flask
    - Deployed applications on AWS
    - Managed SQL databases
    - Implemented Docker containerization
    
    Skills: Python, Flask, AWS, SQL, Docker, Git, REST API
    
    Education:
    BS Computer Science, MIT, 2018
    """
    
    jd = """
    Senior Backend Developer
    
    We are looking for an experienced backend developer with strong Python skills.
    
    Required Skills:
    - Python (5+ years)
    - Flask or Django
    - AWS cloud experience
    - SQL databases
    - Docker/Kubernetes
    
    Experience: 5+ years in backend development
    """
    
    # Test single resume processing
    result = pipeline.process_single_resume(resume, jd)
    
    print(f"Candidate: {result['candidate_name']}")
    print(f"Final Score: {result['final_score']:.2f}")
    print(f"Breakdown: {result['breakdown']['breakdown']}")
    print(f"Matched Skills: {result['skills_detail']['matched_skills']}")
    print(f"Missing Skills: {result['skills_detail']['missing_skills']}")
