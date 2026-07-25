# AI Recruitment System

## Overview

The **AI Recruitment System** is an advanced, AI-driven hiring platform built in Python to automate and optimize the recruitment process. (via Groq’s API) , it offers a suite of tools accessible through a Streamlit web interface. The system handles everything from creating detailed job descriptions to conducting AI-driven interviews, making it an efficient solution for modern hiring needs.

Key features include:
- **Detailed Job Description Generation**: Produces comprehensive job postings with multiple sections.
- **Resume Ranking**: Evaluates resumes for job fit with bias mitigation.
- **Personalized Email Automation**: Sends AI-generated, tailored emails (e.g., interview invites).
  


The system prioritizes ethical AI practices, such as bias avoidance and in-memory data processing for privacy (via MCP), and uses simulated APIs for email and calendar functions.



## Functionality

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
