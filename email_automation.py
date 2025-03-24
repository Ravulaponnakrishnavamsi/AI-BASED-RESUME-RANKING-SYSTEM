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

email_automation = Agent(
    role="Email Automation Agent",
    goal="Generate personalized, professional emails for candidates",
    backstory="An expert in crafting tailored email communication",
    llm=llm,
    verbose=True,
    allow_delegation=False
)

def simulate_email_api(email_content, recipient):
    logging.info(f"Simulated email sent to {recipient}")
    return {"status": "success", "message": f"Email sent to {recipient}"}

def create_email_task(candidate_name, email_type, job_title, details=None, recipient_email=None):
    if email_type == "interview_invite":
        prompt = f"Generate a personalized, professional email inviting {candidate_name} to an interview for the {job_title} position. Include the scheduled time: {details}. Address it to {candidate_name} and ensure it’s friendly yet formal. Use {recipient_email} as the recipient’s email in the salutation if provided."
    else:  
        prompt = f"Generate a personalized email updating the hiring team about {candidate_name}’s status for the {job_title} position. Include details: {details}. Keep it concise and professional."
    return Task(
        description=prompt,
        agent=email_automation,
        expected_output="A fully drafted, personalized email."
    )

if __name__ == "__main__":
    task = create_email_task("John Doe", "interview_invite", "Senior Python Developer", "March 25, 2025, 10:00 AM", "john.doe@example.com")
    crew = Crew(agents=[email_automation], tasks=[task], verbose=True)
    email_content = crew.kickoff()
    result = simulate_email_api(email_content, "john.doe@example.com")
    print("Email Content:", email_content)
    print("API Response:", result)