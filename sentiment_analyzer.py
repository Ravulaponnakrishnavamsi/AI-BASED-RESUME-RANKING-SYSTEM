from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-70b-8192",
    temperature=0.5,
    max_tokens=1000
)

sentiment_analyzer = Agent(
    role="Sentiment Analyzer",
    goal="Analyze the sentiment of interview transcripts to assess confidence and emotional tone",
    backstory="An expert in natural language processing and emotional analysis",
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# Function to create a sentiment analysis task
def create_sentiment_task(transcript):
    return Task(
        description=f"Analyze the sentiment of this interview transcript: '{transcript}'. Provide scores (0-100) for confidence and emotional tone (e.g., positive, neutral, negative) with reasoning.",
        agent=sentiment_analyzer,
        expected_output="Sentiment analysis with confidence score (0-100), emotional tone score (0-100), and a brief explanation."
    )

if __name__ == "__main__":
    # Example transcript for testing
    sample_transcript = """
    Interviewer: Tell me about your experience with Python.
    Candidate: I've used Python for 4 years, mostly with Flask for web apps. My AWS experience is limited to a few months.
    Interviewer: How did you handle a challenge with Flask?
    Candidate: I once debugged a performance issue by optimizing database queries, which improved response time by 30%.
    """
    task = create_sentiment_task(sample_transcript)
    crew = Crew(agents=[sentiment_analyzer], tasks=[task], verbose=True)
    result = crew.kickoff()
    print("Sentiment Analysis:")
    print(result)