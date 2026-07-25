"""
Configuration for AI Recruitment System
"""

# BERT Model Configuration
BERT_MODEL = "all-mpnet-base-v2"  # Options: "all-mpnet-base-v2" (accurate) or "all-MiniLM-L6-v2" (fast)

# TF-IDF Configuration
TFIDF_MAX_FEATURES = 5000          # Max vocabulary size for TF-IDF vectorizer
DEFAULT_RANKING_MODEL = "BERT"     # "BERT" or "TF-IDF"

# Scoring Weights (must sum to 1.0)
SCORING_WEIGHTS = {
    'semantic': 0.40,      # Overall semantic similarity between resume and JD
    'skills': 0.30,        # Technical skills match
    'experience': 0.20,    # Experience relevance
    'credibility': 0.10    # Resume quality and completeness
}

# Optional LLM Features
USE_LLM_FOR_COMMUNICATION = True  # Set to False to disable LLM for emails/feedback

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "app.log"

# ATS Settings
TOP_N_CANDIDATES = 4  # Number of top candidates to display by default
MIN_SCORE_THRESHOLD = 50.0  # Minimum score to consider a candidate

# Performance
ENABLE_CACHING = True  # Cache embeddings for faster re-processing
BATCH_SIZE = 10  # Number of resumes to process in parallel
