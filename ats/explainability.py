"""
Explainability Engine Module
Generate human-readable explanations for ATS scores
"""

from typing import Dict, List


class ExplainabilityEngine:
    """
    Generate natural language explanations for scores
    """
    
    @staticmethod
    def generate_explanation(result: Dict) -> str:
        """
        Generate detailed explanation for a scored resume
        
        Args:
            result: Output from ATSPipeline.process_single_resume()
            
        Returns:
            Markdown-formatted natural language explanation
        """
        final_score = result['final_score']
        breakdown = result['breakdown']['breakdown']
        skills_detail = result.get('skills_detail', {})
        credibility_detail = result.get('credibility_detail', {})
        
        # Build explanation
        lines = []
        
        # Overall assessment
        lines.append(f"## Overall Match: {final_score:.1f}/100\n")
        
        if final_score >= 85:
            lines.append("🎯 **Excellent match** for this position\n")
        elif final_score >= 70:
            lines.append("✅ **Good match** for this position\n")
        elif final_score >= 55:
            lines.append("⚠️ **Moderate match** - some gaps identified\n")
        else:
            lines.append("❌ **Weak match** - significant gaps\n")
        
        # Semantic similarity
        sem_score = breakdown.get('semantic', 0)
        lines.append(f"### Semantic Alignment: {sem_score:.1f}/100")
        
        if sem_score >= 85:
            lines.append("✅ Strong overall alignment with job requirements")
        elif sem_score >= 70:
            lines.append("⚠️ Moderate alignment with job requirements")
        else:
            lines.append("❌ Weak alignment with job requirements")
        
        lines.append("")
        
        # Skills analysis
        skills_score = breakdown.get('skills', 0)
        matched_skills = skills_detail.get('matched_skills', [])
        missing_skills = skills_detail.get('missing_skills', [])
        
        lines.append(f"### Skills Match: {skills_score:.1f}/100")
        
        if matched_skills:
            lines.append(f"✅ **Matched skills ({len(matched_skills)})**: {', '.join(matched_skills[:8])}")
            if len(matched_skills) > 8:
                lines.append(f"   _... and {len(matched_skills) - 8} more_")
        else:
            lines.append("⚠️ No exact skill matches found")
        
        if missing_skills:
            lines.append(f"⚠️ **Missing skills ({len(missing_skills)})**: {', '.join(missing_skills[:5])}")
            if len(missing_skills) > 5:
                lines.append(f"   _... and {len(missing_skills) - 5} more_")
        
        lines.append("")
        
        # Experience relevance
        exp_score = breakdown.get('experience', 0)
        lines.append(f"### Experience Relevance: {exp_score:.1f}/100")
        
        if exp_score >= 80:
            lines.append("✅ Highly relevant experience detected")
        elif exp_score >= 60:
            lines.append("⚠️ Some relevant experience detected")
        else:
            lines.append("❌ Limited relevant experience detected")
        
        lines.append("")
        
        # Credibility assessment
        cred_score = breakdown.get('credibility', 0)
        cred_flags = credibility_detail.get('flags', [])
        
        lines.append(f"### Resume Quality: {cred_score:.1f}/100")
        
        if cred_score >= 80:
            lines.append("✅ Well-formatted, professional resume")
        elif cred_score >= 60:
            lines.append("⚠️ Resume has minor quality issues")
        else:
            lines.append("❌ Resume has significant quality issues")
        
        if cred_flags:
            lines.append(f"**Issues**: {', '.join(cred_flags[:3])}")
        
        return "\n".join(lines)
    
    @staticmethod
    def generate_ranking_summary(results: List[Dict], top_n: int = 10) -> str:
        """
        Generate summary for all ranked resumes
        
        Args:
            results: List of ranked resume results
            top_n: Number of top candidates to show
            
        Returns:
            Markdown-formatted ranking summary
        """
        lines = []
        lines.append(f"# Ranking Summary\n")
        lines.append(f"**Total Candidates**: {len(results)}\n")
        
        # Show top N
        display_count = min(top_n, len(results))
        lines.append(f"## Top {display_count} Candidates\n")
        
        for result in results[:display_count]:
            rank = result.get('rank', '?')
            candidate = result.get('candidate_name', 'Unknown')
            score = result.get('final_score', 0)
            filename = result.get('filename', '')
            
            # Score badge
            if score >= 85:
                badge = "🎯"
            elif score >= 70:
                badge = "✅"
            elif score >= 55:
                badge = "⚠️"
            else:
                badge = "❌"
            
            lines.append(f"{rank}. {badge} **{candidate}** - {score:.1f}/100")
            if filename:
                lines.append(f"   _File: {filename}_")
            lines.append("")
        
        # Score distribution
        if len(results) > 5:
            scores = [r['final_score'] for r in results]
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            
            lines.append(f"## Score Distribution\n")
            lines.append(f"- **Average**: {avg_score:.1f}")
            lines.append(f"- **Highest**: {max_score:.1f}")
            lines.append(f"- **Lowest**: {min_score:.1f}")
        
        return "\n".join(lines)
    
    @staticmethod
    def generate_comparison(result1: Dict, result2: Dict) -> str:
        """
        Generate side-by-side comparison of two candidates
        
        Args:
            result1: First candidate result
            result2: Second candidate result
            
        Returns:
            Markdown-formatted comparison
        """
        lines = []
        lines.append("# Candidate Comparison\n")
        
        name1 = result1.get('candidate_name', 'Candidate 1')
        name2 = result2.get('candidate_name', 'Candidate 2')
        
        score1 = result1['final_score']
        score2 = result2['final_score']
        
        breakdown1 = result1['breakdown']['breakdown']
        breakdown2 = result2['breakdown']['breakdown']
        
        lines.append(f"| Metric | {name1} | {name2} |")
        lines.append("|--------|---------|---------|")
        lines.append(f"| **Overall Score** | {score1:.1f} | {score2:.1f} |")
        lines.append(f"| Semantic | {breakdown1.get('semantic', 0):.1f} | {breakdown2.get('semantic', 0):.1f} |")
        lines.append(f"| Skills | {breakdown1.get('skills', 0):.1f} | {breakdown2.get('skills', 0):.1f} |")
        lines.append(f"| Experience | {breakdown1.get('experience', 0):.1f} | {breakdown2.get('experience', 0):.1f} |")
        lines.append(f"| Quality | {breakdown1.get('credibility', 0):.1f} | {breakdown2.get('credibility', 0):.1f} |")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Test the explainability engine
    explainer = ExplainabilityEngine()
    
    # Mock result
    result = {
        'candidate_name': 'John Doe',
        'filename': 'john_doe_resume.pdf',
        'final_score': 87.5,
        'breakdown': {
            'final_score': 87.5,
            'breakdown': {
                'semantic': 92.0,
                'skills': 85.0,
                'experience': 88.0,
                'credibility': 80.0
            }
        },
        'skills_detail': {
            'score': 85.0,
            'matched_skills': ['python', 'flask', 'aws', 'sql'],
            'missing_skills': ['docker'],
            'semantic_skill_score': 82.0
        },
        'credibility_detail': {
            'score': 80.0,
            'flags': []
        }
    }
    
    explanation = explainer.generate_explanation(result)
    print(explanation)
