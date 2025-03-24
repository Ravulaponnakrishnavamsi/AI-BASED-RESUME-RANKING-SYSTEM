from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import logging

load_dotenv()
logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-70b-8192",
    temperature=0.5,
    max_tokens=1000
)

interview_scheduler = Agent(
    role="Interview Scheduler",
    goal="Schedule interviews based on candidate availability and provide summaries",
    backstory="An expert in coordinating schedules for AI-driven interviews",
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# Simulated Google Calendar API
def simulate_calendar_api(candidate_name, job_title, start_time):
    event = {
        "summary": f"Interview: {candidate_name} for {job_title}",
        "start": start_time.isoformat(),
        "end": (start_time + timedelta(hours=1)).isoformat()
    }
    logging.info(f"Simulated Calendar Event Created: {event}")
    return {"status": "success", "event": event}

# Task to suggest a time based on candidate availability
def create_schedule_time_task(job_title, candidate_name, candidate_availability):
    prompt = f"Given the job title: {job_title} and candidate {candidate_name}'s availability: {candidate_availability}, suggest an interview time in the format 'March 25, 2025, 10:00 AM'. The interviewer is an AI chatbot, so only the candidate’s availability matters. Pick a suitable time within the provided range."
    return Task(
        description=prompt,
        agent=interview_scheduler,
        expected_output="A suggested interview time in the format 'March 25, 2025, 10:00 AM'."
    )

# Task to generate a summary
def create_schedule_summary_task(candidate_name, job_title, scheduled_time):
    prompt = f"Generate a concise summary for an interview scheduled for {candidate_name} for the role of {job_title} at {scheduled_time}. Keep it short and professional."
    return Task(
        description=prompt,
        agent=interview_scheduler,
        expected_output="A brief summary of the scheduled interview."
    )

# Main scheduling function
def create_schedule_task(job_title, candidate_name, candidate_availability):
    time_task = create_schedule_time_task(job_title, candidate_name, candidate_availability)
    return time_task  # Return only the time task; summary is generated separately in app.py

if __name__ == "__main__":
    task = create_schedule_task(
        "Senior Python Developer", 
        "John Doe", 
        "March 25, 2025, 9 AM - 12 PM"
    )
    crew = Crew(agents=[interview_scheduler], tasks=[task], verbose=True)
    time = crew.kickoff()
    scheduled_time = datetime.strptime(time, "%B %d, %Y, %I:%M %p")
    result = simulate_calendar_api("John Doe", "Senior Python Developer", scheduled_time)
    print("Scheduled Time:", time)
    print("Calendar Response:", result)