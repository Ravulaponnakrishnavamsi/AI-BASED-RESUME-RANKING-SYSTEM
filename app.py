import streamlit as st
from crewai import Crew
from jd_generator import jd_generator, create_jd_task
from resume_ranker import resume_ranker, create_resume_rank_task
from email_automation import email_automation, create_email_task, simulate_email_api
from interview_scheduler import interview_scheduler, create_schedule_task, simulate_calendar_api, create_schedule_summary_task
from interview_agent import interview_agent, create_interview_task, evaluate_response_task
from hire_recommendation import hire_recommendation_agent, create_hire_recommendation_task
from sentiment_analyzer import sentiment_analyzer, create_sentiment_task
from datetime import datetime
import os
import logging
from streamlit_extras.add_vertical_space import add_vertical_space

# Setup logging
logging.basicConfig(filename="Logs/app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

st.set_page_config(page_title="AI Recruitment System", layout="wide")
st.title("AI Recruitment System")

# MCP Context
if "mcp_context" not in st.session_state:
    st.session_state.mcp_context = {
        "job_description": None,
        "ranked_resumes": None,
        "scheduled_time": None,
        "interview_transcript": None
    }

# Sidebar for transparency
st.sidebar.markdown("""
### AI Capabilities & Limitations
- **Powered by llama3-70b-8192**: Generates human-like text but may produce inaccuracies.
- **RAG**: Enhances outputs with web data, limited by search quality.
- **Simulation**: Email and calendar APIs are simulated.

### Ethical Hiring
Data is processed in-memory and not stored unless saved by the user.
""")

tabs = st.tabs([
    "JD Generator", "Resume Ranker", "Email Automation", "Interview Scheduler", 
    "Interview Agent", "Hire Recommendation", "Sentiment Analyzer"
])

# Tab 1: JD Generator
with tabs[0]:
    st.header("JD Generator")
    with st.expander("Template & Inputs", expanded=True):
        st.markdown("**Upload a JD template** *(optional, defaults to Templates/jd_template.txt)*")
        template_file = st.file_uploader("Upload JD Template (.txt)", type=["txt"], key="jd_template")
        if template_file:
            with open("Templates/jd_template.txt", "wb") as f:
                f.write(template_file.read())
        job_title = st.text_input("Job Title", "e.g., Senior Python Developer", key="jd_job_title", help="Enter the job title")
        skills = st.text_area("Required Skills", "e.g., Python, Flask, SQL, AWS", key="jd_skills")
        experience_level = st.text_input("Experience Level", "e.g., 5+ years", key="jd_experience")
    if st.button("Generate Job Description", key="jd_button", help="Generate a detailed job description"):
        if job_title and skills and experience_level:
            with st.spinner("Generating detailed JD..."):
                try:
                    jd_task = create_jd_task(job_title, skills, experience_level)
                    crew = Crew(agents=[jd_generator], tasks=[jd_task], verbose=True)
                    result = crew.kickoff()
                    st.session_state.mcp_context["job_description"] = result
                    st.subheader("Generated Job Description")
                    st.markdown(result)  # Use markdown to render formatted output
                    logging.info(f"JD generated for {job_title}")
                except Exception as e:
                    st.error(f"Error: {str(e)}. See Logs/app.log for details.")
                    logging.error(f"JD generation failed: {str(e)}")
        else:
            st.error("Fill in all fields.")
            logging.warning("JD generation attempted with missing fields")

# Tab 2: Resume Ranker
with tabs[1]:
    st.header("Resume Ranker")
    with st.expander("Inputs", expanded=True):
        job_desc = st.text_area("Job Description", value=st.session_state.mcp_context["job_description"] or "", key="resume_job_desc", help="Paste or enter job description")
        dir_path = st.text_input("Directory Path", "e.g., D:/resumes", key="dir_path")
        uploaded_files = st.file_uploader("Upload Resume PDFs", type=["pdf"], accept_multiple_files=True, key="resume_files")
    if st.button("Rank Resumes", key="resume_button", help="Rank uploaded or directory resumes"):
        if job_desc:
            if (dir_path and os.path.isdir(dir_path)) or uploaded_files:
                with st.spinner("Ranking resumes..."):
                    try:
                        task = create_resume_rank_task(job_desc, dir_path, uploaded_files)
                        if task:
                            crew = Crew(agents=[resume_ranker], tasks=[task], verbose=True)
                            result = crew.kickoff()
                            st.session_state.mcp_context["ranked_resumes"] = result
                            st.subheader("Ranked Resumes")
                            st.write(result)
                            logging.info("Resumes ranked")
                        else:
                            st.error("No valid resumes found.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}. See Logs/app.log for details.")
                        logging.error(f"Resume ranking failed: {str(e)}")
            else:
                st.error("Provide a directory path or upload resumes.")
        else:
            st.error("Provide a job description.")

 
# Tab 3: Email Automation (Personalized AI-Generated Content)
with tabs[2]:
    st.header("Email Automation")
    with st.expander("Email Details", expanded=True):
        st.markdown("**Enter details for a personalized email**")
        candidate_name = st.text_input("Candidate Name", "e.g., John Doe", key="email_candidate", help="Candidate's full name")
        job_title_email = st.text_input("Job Title", "e.g., Senior Python Developer", key="email_job_title", help="Job title for the email")
        email_type = st.selectbox("Email Type", ["interview_invite", "hiring_team_update"], key="email_type", help="Choose email purpose")
        details = st.text_area("Details", value=st.session_state.mcp_context["scheduled_time"] or "e.g., March 25, 2025, 10 AM", key="email_details", help="e.g., interview time or update details")
        recipient_email = st.text_input("Recipient Email", "e.g., john.doe@example.com", key="email_recipient", help="Recipient's email address")
    if st.button("Send Email", key="email_button", help="Generate and simulate sending a personalized email"):
        if candidate_name and job_title_email and details and recipient_email:
            with st.spinner("Generating personalized email..."):
                try:
                    task = create_email_task(candidate_name, email_type, job_title_email, details, recipient_email)
                    crew = Crew(agents=[email_automation], tasks=[task], verbose=True)
                    email_content = crew.kickoff()
                    result = simulate_email_api(email_content, recipient_email)
                    st.subheader("Email Content")
                    st.write(email_content)
                    st.subheader("API Response")
                    st.write(result)
                    logging.info(f"Email simulated for {recipient_email}")
                except Exception as e:
                    st.error(f"Error: {str(e)}. See Logs/app.log for details.")
                    logging.error(f"Email generation failed: {str(e)}")
        else:
            st.error("Fill in all fields.")

# Tab 4: Interview Scheduler (Candidate Availability Only)
with tabs[3]:
    st.header("Interview Scheduler")
    with st.expander("Scheduling Details", expanded=True):
        st.markdown("**Enter candidate availability** ")
        candidate_name_sched = st.text_input("Candidate Name", "e.g., John Doe", key="sched_candidate", help="Candidate's full name")
        job_title_sched = st.text_input("Job Title", "e.g., Senior Python Developer", key="sched_job_title", help="Job title for the interview")
        candidate_avail = st.text_area("Candidate Availability", "e.g., March 25, 2025, 9 AM - 12 PM", key="sched_candidate_avail", help="e.g., March 25, 2025, 9 AM - 12 PM")
    if st.button("Schedule Interview", key="sched_button", help="Schedule the interview based on candidate availability"):
        if candidate_name_sched and job_title_sched and candidate_avail:
            with st.spinner("Scheduling interview..."):
                try:
                    time_task = create_schedule_task(job_title_sched, candidate_name_sched, candidate_avail)
                    crew = Crew(agents=[interview_scheduler], tasks=[time_task], verbose=True)
                    scheduled_time_str = crew.kickoff()
                    scheduled_time = datetime.strptime(scheduled_time_str, "%B %d, %Y, %I:%M %p")
                    calendar_result = simulate_calendar_api(candidate_name_sched, job_title_sched, scheduled_time)
                    st.session_state.mcp_context["scheduled_time"] = scheduled_time_str
                    
                    summary_task = create_schedule_summary_task(candidate_name_sched, job_title_sched, scheduled_time_str)
                    crew = Crew(agents=[interview_scheduler], tasks=[summary_task], verbose=True)
                    summary = crew.kickoff()
                    
                    st.subheader("Scheduled Time")
                    st.write(scheduled_time_str)
                    st.subheader("Calendar Response")
                    st.write(calendar_result)
                    st.subheader("Interview Summary")
                    st.write(summary)
                    logging.info(f"Interview scheduled for {candidate_name_sched} with summary")
                except Exception as e:
                    st.error(f"Error: {str(e)}. See Logs/app.log for details.")
                    logging.error(f"Scheduling failed: {str(e)}")
        else:
            st.error("Fill in all fields.")
            logging.warning("Scheduling attempted with missing fields")
# Tab 5: Interview Agent
with tabs[4]:
    st.header("Interview Agent")
    with st.expander("Job Description", expanded=True):
        st.markdown("**Enter the job description** *(e.g., Senior Python Developer requiring...)*")
        job_desc_interview = st.text_area("Job Description", value=st.session_state.mcp_context["job_description"] or "", key="interview_job_desc")
    if "interview_history" not in st.session_state:
        st.session_state.interview_history = []
    if "current_question" not in st.session_state:
        st.session_state.current_question = None
    if st.button("Start Interview", key="start_interview", help="Begin the interview"):
        if job_desc_interview:
            with st.spinner("Generating Initial Question..."):
                try:
                    task = create_interview_task(job_desc_interview)
                    crew = Crew(agents=[interview_agent], tasks=[task], verbose=True)
                    question = crew.kickoff()
                    st.session_state.current_question = question
                    st.session_state.interview_history = [{"role": "agent", "content": str(question)}]
                    logging.info("Interview started")
                except Exception as e:
                    st.error(f"Error: {str(e)}. See Logs/app.log for details.")
                    logging.error(f"Interview start failed: {str(e)}")
        else:
            st.error("Provide a job description.")
    if st.session_state.interview_history:
        st.subheader("Conversation History")
        for message in st.session_state.interview_history:
            st.write(f"**{message['role'].capitalize()}**: {message['content']}")
    if st.session_state.current_question:
        candidate_response = st.text_area("Your Response", key="candidate_response", value="", height=100)
        if st.button("Submit Response", key="submit_response", help="Submit your answer"):
            if candidate_response:
                with st.spinner("Generating Follow-up..."):
                    try:
                        st.session_state.interview_history.append({"role": "candidate", "content": candidate_response})
                        eval_task = evaluate_response_task(job_desc_interview, st.session_state.interview_history, candidate_response)
                        crew = Crew(agents=[interview_agent], tasks=[eval_task], verbose=True)
                        follow_up = crew.kickoff()
                        st.session_state.current_question = follow_up
                        st.session_state.interview_history.append({"role": "agent", "content": str(follow_up)})
                        st.session_state.mcp_context["interview_transcript"] = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.interview_history])
                        st.experimental_rerun()
                        logging.info("Follow-up question generated")
                    except Exception as e:
                        st.error(f"Error: {str(e)}. See Logs/app.log for details.")
                        logging.error(f"Follow-up generation failed: {str(e)}")
            else:
                st.error("Provide a response.")

# Tab 6: Hire Recommendation
with tabs[5]:
    st.header("Hire Recommendation Agent")
    with st.expander("Transcript Input", expanded=True):
        transcript_default = st.session_state.mcp_context["interview_transcript"] or ""
        transcript_file = st.file_uploader("Upload Transcript (.txt)", type=["txt"], key="hire_transcript")
        transcript_text = st.text_area("Or Paste Transcript", value=transcript_default, key="hire_transcript_text")
    if st.button("Generate Recommendation", key="hire_button", help="Analyze transcript for hiring decision"):
        transcript = transcript_file.read().decode("utf-8") if transcript_file else transcript_text
        if transcript:
            with st.spinner("Analyzing Transcript..."):
                try:
                    task = create_hire_recommendation_task(transcript)
                    crew = Crew(agents=[hire_recommendation_agent], tasks=[task], verbose=True)
                    result = crew.kickoff()
                    st.subheader("Hiring Recommendation")
                    st.write(result)
                    logging.info("Hiring recommendation generated")
                except Exception as e:
                    st.error(f"Error: {str(e)}. See Logs/app.log for details.")
                    logging.error(f"Hire recommendation failed: {str(e)}")
        else:
            st.error("Upload a file or paste a transcript.")

# Tab 7: Sentiment Analyzer
with tabs[6]:
    st.header("Sentiment Analyzer")
    with st.expander("Transcript Input", expanded=True):
        sentiment_default = st.session_state.mcp_context["interview_transcript"] or ""
        sentiment_file = st.file_uploader("Upload Transcript (.txt)", type=["txt"], key="sentiment_transcript")
        sentiment_text = st.text_area("Or Paste Transcript", value=sentiment_default, key="sentiment_transcript_text")
    if st.button("Analyze Sentiment", key="sentiment_button", help="Analyze transcript sentiment"):
        transcript = sentiment_file.read().decode("utf-8") if sentiment_file else sentiment_text
        if transcript:
            with st.spinner("Analyzing Sentiment..."):
                try:
                    task = create_sentiment_task(transcript)
                    crew = Crew(agents=[sentiment_analyzer], tasks=[task], verbose=True)
                    result = crew.kickoff()
                    st.subheader("Sentiment Analysis")
                    st.write(result)
                    logging.info("Sentiment analysis completed")
                except Exception as e:
                    st.error(f"Error: {str(e)}. See Logs/app.log for details.")
                    logging.error(f"Sentiment analysis failed: {str(e)}")
        else:
            st.error("Upload a file or paste a transcript.")
    add_vertical_space(2)