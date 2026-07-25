"""
Fast Feedback Generator — Direct Groq SDK (no CrewAI overhead)
Generates per-candidate feedback in ~1-2s instead of 15-20s with CrewAI.
"""

from groq import Groq
from dotenv import load_dotenv
import os
import logging
import json
import re

load_dotenv()

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Single reusable Groq client
_groq_client = None

def get_groq_client() -> Groq:
    """Lazy-initialize Groq client (singleton)"""
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


def _call_groq(prompt: str, max_tokens: int = 800) -> str:
    """
    Direct Groq API call — no CrewAI, no agent spin-up.
    Uses llama-3.1-8b-instant for maximum speed.
    """
    client = get_groq_client()
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an empathetic HR career coach. "
                    "Return ONLY valid JSON, no extra text, no markdown fences."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def _extract_json(raw: str) -> dict:
    """Extract first JSON object from a string."""
    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError("No JSON object found in response")


def generate_single_feedback(candidate: dict, job_description: str, is_shortlisted: bool) -> dict:
    """
    Generate feedback for ONE candidate via direct Groq call.
    Returns feedback dict or error fallback.
    Fast: ~1-3 seconds per candidate.
    """
    resume_snippet = candidate.get("resume_text", "")[:600]
    status_word = "SHORTLISTED" if is_shortlisted else "NOT SELECTED"
    tone = "enthusiastic and professional" if is_shortlisted else "supportive, constructive, motivating"

    prompt = f"""Generate feedback for this job applicant.

JOB: {job_description[:400]}

CANDIDATE: {candidate.get('candidate_name', 'Candidate')}
STATUS: {status_word}
SCORE: {candidate.get('score', 0):.1f}%
RESUME SNIPPET: {resume_snippet}

Tone: {tone}

Return ONLY this JSON (no extra text):
{{
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "areas_for_improvement": ["area 1", "area 2"],
  "overall_message": "2-3 sentence personalized message",
  "suggestions": ["suggestion 1", "suggestion 2"],
  "next_steps": "1-2 sentence next steps"
}}"""

    try:
        raw = _call_groq(prompt, max_tokens=500)
        feedback = _extract_json(raw)
        return {
            "candidate_name": candidate.get("candidate_name", "Unknown"),
            "status": "shortlisted" if is_shortlisted else "not_selected",
            "feedback": feedback
        }
    except Exception as e:
        logging.error(f"Feedback error for {candidate.get('candidate_name')}: {e}")
        # Fast fallback — no API needed
        return _fallback_feedback(candidate, is_shortlisted)


def _fallback_feedback(candidate: dict, is_shortlisted: bool) -> dict:
    """Generate template feedback when API fails — instant, no latency."""
    score = candidate.get("score", 0)
    name = candidate.get("candidate_name", "Candidate")
    skills = candidate.get("skills_matched", [])

    if is_shortlisted:
        msg = (f"Congratulations {name}! Your profile strongly aligns with our requirements. "
               f"Your score of {score:.1f}% reflects excellent qualifications.")
        next_steps = "We will contact you within 3-5 business days to schedule next steps."
        strengths = skills[:3] if skills else ["Strong technical background", "Relevant experience", "Good profile match"]
    else:
        msg = (f"Thank you for applying, {name}. While we are moving forward with other candidates, "
               f"we see genuine potential in your profile and encourage you to keep developing.")
        next_steps = "Continue building your skills and consider applying for future roles that match your profile."
        strengths = skills[:2] if skills else ["Shows initiative", "Relevant educational background"]

    return {
        "candidate_name": name,
        "status": "shortlisted" if is_shortlisted else "not_selected",
        "feedback": {
            "strengths": strengths,
            "areas_for_improvement": ["Gain more hands-on project experience", "Strengthen domain-specific skills"],
            "overall_message": msg,
            "suggestions": ["Build portfolio projects", "Earn relevant certifications"],
            "next_steps": next_steps
        }
    }


def generate_feedback(all_candidates: list, job_description: str,
                      progress_callback=None) -> dict:
    """
    Generate feedback for ALL candidates.
    
    Args:
        all_candidates: List of candidate dicts
        job_description: Job description text
        progress_callback: Optional callable(current, total, candidate_name) for UI progress
    
    Returns:
        {"shortlisted": [...], "rejected": [...]}
    """
    shortlisted_feedback = []
    rejected_feedback = []

    total = len(all_candidates)

    for idx, candidate in enumerate(all_candidates):
        is_shortlisted = candidate.get("is_shortlisted", False)
        name = candidate.get("candidate_name", f"Candidate {idx+1}")

        logging.info(f"Generating feedback for {name} ({idx+1}/{total})")

        if progress_callback:
            progress_callback(idx, total, name)

        fb = generate_single_feedback(candidate, job_description, is_shortlisted)

        if is_shortlisted:
            shortlisted_feedback.append(fb)
        else:
            rejected_feedback.append(fb)

    if progress_callback:
        progress_callback(total, total, "Complete")

    return {
        "shortlisted": shortlisted_feedback,
        "rejected": rejected_feedback
    }


def format_feedback_for_download(feedback_data: dict) -> str:
    """Format feedback as readable text for download."""
    output = []
    output.append("=" * 80)
    output.append("CANDIDATE FEEDBACK REPORT")
    output.append("=" * 80)
    output.append("")

    def _write_section(title, items):
        output.append(title)
        output.append("-" * 80)
        for fb in items:
            output.append(f"\nCandidate: {fb['candidate_name']}")
            output.append(f"Status: {fb['status'].upper()}")
            f = fb.get("feedback", {})
            output.append(f"\nOverall Message:\n{f.get('overall_message', '')}")
            output.append("\nStrengths:")
            for s in f.get("strengths", []):
                output.append(f"  ✓ {s}")
            output.append("\nAreas for Improvement:")
            for a in f.get("areas_for_improvement", []):
                output.append(f"  → {a}")
            output.append("\nSuggestions:")
            for sg in f.get("suggestions", []):
                output.append(f"  • {sg}")
            output.append(f"\nNext Steps:\n{f.get('next_steps', '')}")
            output.append("\n" + "-" * 80)

    if feedback_data.get("shortlisted"):
        _write_section("SHORTLISTED CANDIDATES", feedback_data["shortlisted"])

    if feedback_data.get("rejected"):
        output.append("")
        _write_section("REJECTED CANDIDATES", feedback_data["rejected"])

    return "\n".join(output)


if __name__ == "__main__":
    import time
    test_candidates = [
        {
            "candidate_name": "John Doe",
            "file_name": "john_doe.pdf",
            "is_shortlisted": True,
            "score": 92.5,
            "resume_text": "Senior Python Developer with 8 years experience in Flask, AWS, PostgreSQL.",
            "skills_matched": ["Python", "Flask", "AWS"]
        },
        {
            "candidate_name": "Jane Smith",
            "file_name": "jane_smith.pdf",
            "is_shortlisted": False,
            "score": 58.0,
            "resume_text": "Junior Python Developer with 1 year experience.",
            "skills_matched": ["Python"]
        }
    ]

    test_jd = "Senior Python Developer with 5+ years experience in Flask, AWS, and SQL"

    print("Generating feedback (direct Groq SDK)...")
    t0 = time.time()
    result = generate_feedback(test_candidates, test_jd)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")
    print(json.dumps(result, indent=2))
