from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import logging

load_dotenv()
logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-70b-8192",
    temperature=0.5,
    max_tokens=1000
)

hire_recommendation_agent = Agent(
    role="Hire Recommendation Agent",
    goal="Analyze transcripts for fair hiring decisions",
    backstory="An expert in candidate evaluation",
    llm=llm,
    verbose=True,
    allow_delegation=False
)

def create_hire_recommendation_task(transcript):
    prompt = f"Analyze this transcript: '{transcript}'. Provide strengths, weaknesses, and a Hire/No-Hire decision. Ensure fairness by avoiding bias based on gender, age, or ethnicity. Flag any biased language or assumptions."
    return Task(
        description=prompt,
        agent=hire_recommendation_agent,
        expected_output="Analysis with strengths, weaknesses, Hire/No-Hire, and bias flags."
    )