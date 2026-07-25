from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

load_dotenv()
logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
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

def send_email_via_smtp(email_content, recipient):
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    
    if not sender_email or not sender_password:
        return {"status": "error", "message": "Missing SENDER_EMAIL or SENDER_PASSWORD in .env"}

    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient
        msg['Subject'] = "Recruitment Alert"
        msg.attach(MIMEText(str(email_content), 'plain'))

        # Connect to Gmail SMTP (default)
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient, text)
        server.quit()
        
        logging.info(f"Email sent successfully to {recipient}")
        return {"status": "success", "message": f"Email successfully sent to {recipient}"}
    except Exception as e:
        logging.error(f"Failed to send email: {str(e)}")
        return {"status": "error", "message": f"Failed to send email: {str(e)}"}

def create_email_task(candidate_name, email_type, job_title, details=None, recipient_email=None):
    if email_type == "interview_invite":
        prompt = f"Generate a personalized, professional email inviting {candidate_name} to an interview for the {job_title} position. Include the scheduled time: {details}. Address it to {candidate_name} and ensure it’s friendly yet formal. Use {recipient_email} as the recipient’s email in the salutation if provided. and company name is Wipro hr name jhonathon and contact number is 9876543210 and location hydrabad"
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
    result = send_email_via_smtp(email_content, "john.doe@example.com")
    print("Email Content:", email_content)
    print("API Response:", result)