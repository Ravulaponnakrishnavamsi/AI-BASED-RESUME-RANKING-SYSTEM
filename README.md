---
license: apache-2.0
title: ai-recruitment-system
sdk: streamlit
emoji: 🚀
colorTo: red
short_description: ai-recruitment-system
---



# AI Recruitment System

## Overview

The **AI Recruitment System** is an advanced, AI-driven hiring platform built in Python to automate and optimize the recruitment process. Powered by Llama 3.x (via Groq’s API) and the CrewAI multi-agent framework, it offers a suite of tools accessible through a Streamlit web interface. The system handles everything from creating detailed job descriptions to conducting AI-driven interviews, making it an efficient solution for modern hiring needs.

Key features include:
- **Detailed Job Description Generation**: Produces comprehensive job postings with multiple sections.
- **Resume Ranking**: Evaluates resumes for job fit with bias mitigation.
- **Personalized Email Automation**: Sends AI-generated, tailored emails (e.g., interview invites).
- **Interview Scheduling**: Schedules interviews based on candidate availability, with an AI chatbot interviewer.
- **Interview Agent**: Conducts interactive interviews and evaluates responses.
- **Hire Recommendation**: Analyzes transcripts for hiring decisions.
- **Sentiment Analysis**: Assesses candidate sentiment from interviews.

The system incorporates a **Model Context Protocol (MCP)** to maintain state across its components. MCP uses Streamlit’s session state (`st.session_state.mcp_context`) to store critical outputs—like job descriptions, ranked resumes, scheduled times, and interview transcripts—enabling seamless data flow between tabs without persistent storage. This enhances workflow efficiency and ensures context is preserved throughout the hiring process.

The system prioritizes ethical AI practices, such as bias avoidance and in-memory data processing for privacy (via MCP), and uses simulated APIs for email and calendar functions.


## Project Structure
ai-recruitment-system/
├── Agents/
│   ├── jd_generator.py         # Generates detailed job descriptions
│   ├── resume_ranker.py        # Ranks resumes with fairness
│   ├── email_automation.py     # Crafts personalized emails
│   ├── interview_scheduler.py  # Schedules interviews based on candidate availability
│   ├── interview_agent.py      # Conducts AI-driven interviews
│   ├── hire_recommendation.py  # Provides hiring recommendations
│   ├── sentiment_analyzer.py   # Analyzes sentiment in transcripts
├── Templates/
│   ├── jd_template.txt         # Default JD template with detailed sections
├── Logs/
│   ├── app.log                 # Log file for debugging
├── app.py                      # Streamlit UI for the system
├── requirements.txt            # Dependencies
├── .env                        # GROQ_API_KEY
└── README.md                   # This file



## Agent Functionality

The system uses a multi-agent architecture, with each agent specializing in a recruitment task. Below is a detailed explanation of their roles and how they leverage the Model Context Protocol (MCP):

### 1. JD Generator (`jd_generator.py`)
- **Role**: Generates detailed, professional job descriptions.
- **Functionality**: 
  - Fetches job-related data from trusted web sources (e.g., `.edu`, `.org`, `.gov`) using Google search and RecursiveUrlLoader.
  - Builds a FAISS vector store for contextual relevance from web content.
  - Uses a customizable template and inputs (job title, skills, experience level) to create a comprehensive JD with sections: Company Overview, Job Overview, Responsibilities (5-7 items), Required Skills and Qualifications (5-7 items), Preferred Skills, Benefits, and Application Process.
  - Stores the output in MCP (`mcp_context["job_description"]`) for use in other tabs.
- **Output**: A markdown-formatted, detailed job description.

### 2. Resume Ranker (`resume_ranker.py`)
- **Role**: Ranks resumes based on job fit.
- **Functionality**: 
  - Extracts text from PDF resumes (uploaded or from a directory).
  - Compares resumes to the job description (optionally sourced from MCP) and web context.
  - Assigns scores (0-100) with reasoning, flagging potential bias (e.g., gender, age, ethnicity) for fairness.
  - Stores the ranked list in MCP (`mcp_context["ranked_resumes"]`).
- **Output**: A ranked list of resumes with scores and bias checks.

### 3. Email Automation (`email_automation.py`)
- **Role**: Generates and simulates sending personalized emails.
- **Functionality**: 
  - Takes inputs: candidate name, job title, email type (interview invite or team update), details (e.g., interview time from MCP), and recipient email.
  - Uses Llama 3.x to craft fully personalized, professional emails tailored to the context and recipient.
  - Simulates email delivery with a mock API.
- **Output**: A drafted email and simulated API response.

### 4. Interview Scheduler (`interview_scheduler.py`)
- **Role**: Schedules interviews based on candidate availability.
- **Functionality**: 
  - Accepts candidate name, job title, and availability (e.g., "March 25, 2025, 9 AM - 12 PM").
  - Since the interviewer is an AI chatbot (always available), it selects a time within the candidate’s range.
  - Generates a concise summary of the scheduled interview.
  - Stores the scheduled time in MCP (`mcp_context["scheduled_time"]`).
  - Simulates calendar integration via a mock API.
- **Output**: Scheduled time, calendar response, and a summary.

### 5. Interview Agent (`interview_agent.py`)
- **Role**: Conducts AI-driven interviews.
- **Functionality**: 
  - Uses Retrieval-Augmented Generation (RAG) with web data and the job description (from MCP) to generate job-specific questions.
  - Engages in a conversational loop, evaluating responses and asking follow-ups.
  - Stores the transcript in MCP (`mcp_context["interview_transcript"]`) for downstream analysis.
- **Output**: Interview questions and a full transcript.

### 6. Hire Recommendation (`hire_recommendation.py`)
- **Role**: Provides hiring recommendations from transcripts.
- **Functionality**: 
  - Analyzes interview transcripts (from MCP) for strengths, weaknesses, and a Hire/No-Hire decision.
  - Ensures fairness by avoiding bias (e.g., gender, age, ethnicity) and flagging issues.
- **Output**: A detailed analysis with a hiring recommendation.

### 7. Sentiment Analyzer (`sentiment_analyzer.py`)
- **Role**: Assesses candidate sentiment in interviews.
- **Functionality**: 
  - Evaluates the tone and sentiment (e.g., positive, neutral, negative) of interview transcripts (from MCP).
  - Offers insights into candidate confidence and engagement.
- **Output**: A sentiment analysis report.

## How Llama 3.x Powers the Solution

Llama 3.x, accessed via Groq’s API, is the backbone of the AI Recruitment System, providing advanced natural language processing capabilities. Its integration drives the system’s automation, personalization, and analytical features, enhanced by the Model Context Protocol (MCP) for state management. Here’s how it contributes:

### 1. Detailed Text Generation
- **Agents**: JD Generator, Email Automation, Interview Scheduler (summary), Interview Agent.
- **Role**: Llama 3.x generates rich, context-aware text:
  - **JD Generator**: Produces detailed job descriptions with multiple sections (e.g., Responsibilities, Benefits), incorporating web context and user inputs into a professional, markdown-formatted output stored in MCP.
  - **Email Automation**: Creates personalized emails tailored to the candidate, job, and context (e.g., using MCP’s scheduled time), replacing static templates with dynamic content.
  - **Interview Scheduler**: Generates concise, readable summaries of scheduled interviews, saved to MCP.
  - **Interview Agent**: Crafts dynamic, job-specific questions and follow-ups based on the job description (from MCP) and candidate responses.

### 2. Contextual Analysis and Reasoning
- **Agents**: Resume Ranker, Interview Agent, Hire Recommendation, Sentiment Analyzer.
- **Role**: Llama 3.x interprets and evaluates complex text inputs:
  - **Resume Ranker**: Analyzes resume content against job descriptions (from MCP) and web context, providing scores and bias-aware reasoning, stored in MCP.
  - **Interview Agent**: Assesses candidate responses for relevance and depth, using RAG and MCP data for informed questioning, with transcripts saved to MCP.
  - **Hire Recommendation**: Evaluates transcripts (from MCP) for strengths, weaknesses, and hiring decisions, ensuring fairness.
  - **Sentiment Analyzer**: Detects emotional tone and sentiment in transcripts (from MCP) with nuanced understanding.

### 3. Task Automation via CrewAI
- **Agents**: All agents.
- **Role**: Llama 3.x powers the CrewAI framework, enabling autonomous task execution:
  - Each agent processes specific prompts (e.g., "Generate a detailed JD," "Schedule an interview") using Llama’s reasoning and generation capabilities, with MCP ensuring context continuity.
  - Groq’s API ensures fast inference, critical for real-time features like the Interview Agent.

### 4. Ethical AI Practices
- **Bias Mitigation**: In Resume Ranker and Hire Recommendation, Llama 3.x is instructed to flag and avoid bias based on gender, age, or ethnicity, supporting ethical hiring.
- **Privacy via MCP**: MCP stores data in-memory (e.g., `mcp_context`), avoiding persistent storage for privacy.
- **Transparency**: The Streamlit sidebar highlights Llama 3.x’s role and limitations (e.g., potential inaccuracies).

### Technical Details
- **Model**: Llama 3.x (70B parameters, 8192 token context) via `langchain_groq.ChatGroq`.
- **Parameters**: Temperature=0.5 for balanced output, max_tokens=2000 (increased for detailed JDs) in JD Generator, 1000 elsewhere.
- **Enhancements**: RAG (via FAISS and HuggingFace embeddings) augments JD Generator and Interview Agent with web-sourced context, integrated with MCP.

## Setup Instructions

1. **Extract the ZIP File**:
   - Download the `ai-recruitment-system.zip` file.
   - Extract it to a directory of your choice using a tool like WinZip, 7-Zip, or your OS’s built-in unzip feature:

2. **Set Up Environment**:
  Create a .env file in the root directory with your Groq API key:

    echo GROQ_API_KEY=<your-api-key> > .env

3. **Install Dependencies**:
  Ensure Python 3.8+ is installed, then run:

      pip install -r requirements.txt

4. **Run Application**:
    streamlit run app.py