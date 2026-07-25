"""
Scoring Engine Module
Calculate composite scores from multiple components
"""

from typing import Dict, List
import numpy as np


class ScoringEngine:
    """
    Calculate final composite score from multiple components
    """
    
    DEFAULT_WEIGHTS = {
        'semantic': 0.40,
        'skills': 0.30,
        'experience': 0.20,
        'credibility': 0.10
    }
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize scoring engine with custom or default weights
        
        Args:
            weights: Dictionary of component weights (must sum to 1.0)
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        
        # Validate weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")
    
    def calculate_composite_score(self, component_scores: Dict[str, float]) -> Dict:
        """
        Calculate weighted composite score
        
        Args:
            component_scores: Dictionary with keys matching weight keys
                Example: {
                    'semantic': 85.0,
                    'skills': 90.0,
                    'experience': 75.0,
                    'credibility': 80.0
                }
        
        Returns:
            Dictionary containing:
                - final_score: weighted average
                - breakdown: original component scores
                - contributions: weighted contributions
                - weights: weights used
        """
        final_score = 0.0
        contributions = {}
        
        # Calculate weighted sum
        for component, score in component_scores.items():
            weight = self.weights.get(component, 0.0)
            contribution = score * weight
            final_score += contribution
            contributions[component] = round(contribution, 2)
        
        return {
            'final_score': round(final_score, 2),
            'breakdown': component_scores,
            'contributions': contributions,
            'weights': self.weights
        }
    
    def set_weights(self, new_weights: Dict[str, float]):
        """
        Update scoring weights
        
        Args:
            new_weights: New weight dictionary (must sum to 1.0)
        """
        weight_sum = sum(new_weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")
        
        self.weights = new_weights.copy()
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights"""
        return self.weights.copy()


if __name__ == "__main__":
    # Test the scoring engine
    engine = ScoringEngine()
    
    # Test with sample scores
    scores = {
        'semantic': 85.0,
        'skills': 90.0,
        'experience': 75.0,
        'credibility': 80.0
    }
    
    result = engine.calculate_composite_score(scores)
    
    print(f"Final Score: {result['final_score']}")
    print(f"Breakdown: {result['breakdown']}")
    print(f"Contributions: {result['contributions']}")
    print(f"Weights: {result['weights']}")
    
    # Test with custom weights
    custom_weights = {
        'semantic': 0.50,
        'skills': 0.25,
        'experience': 0.15,
        'credibility': 0.10
    }
    
    engine2 = ScoringEngine(weights=custom_weights)
    result2 = engine2.calculate_composite_score(scores)
    print(f"\nWith custom weights, Final Score: {result2['final_score']}")
