from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from googlesearch import search
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import os
import logging
import re
load_dotenv()
logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-70b-8192",
    temperature=0.5,
    max_tokens=2000  
)

jd_generator = Agent(
    role="JD Generator",
    goal="Generate detailed, professional job descriptions with comprehensive sections",
    backstory="An expert in crafting in-depth job postings for recruitment",
    llm=llm,
    verbose=True,
    allow_delegation=False
)
def html_to_text(html_content: str) -> str:
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract text with proper spacing
    text = soup.get_text(separator=" ").strip()

    # Remove excessive multiple spaces
    text = re.sub(r'\s+', ' ', text) 
def fetch_related_content(job_title, skills, experience_level):
    query = f"{job_title} job description {skills} {experience_level} site:*.edu | site:*.org | site:*.gov -inurl:(signup | login)"
    urls = list(search(query, num_results=5))
    documents = []
    for url in urls:
        try:
            loader = RecursiveUrlLoader(url=url,extractor=html_to_text,max_depth=1,
                                headers={"User-Agent": "Mozilla/5.0"}) 
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            logging.error(f"Error loading {url}: {e}")
    return documents

def store_in_vdb(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vdb = FAISS.from_documents(chunks, embeddings)
    return vdb

def create_jd_task(job_title, skills, experience_level, template_path="jd_template.txt"):
    documents = fetch_related_content(job_title, skills, experience_level)
    vdb = store_in_vdb(documents) if documents else None
    context = vdb.similarity_search(f"Job description for {job_title}", k=3) if vdb else []
    context_text = "\n".join([doc.page_content for doc in context]) or "No context available."
    
    with open(template_path, "r") as f:
        template = f.read()
    
    prompt = f"""
    Using the provided template: '{template}' and web-sourced context: '{context_text}', generate a detailed job description for the position of {job_title}. The JD should be comprehensive and professional, including the following sections:
    - **Job Title**: Incorporate {job_title}.
    - **Experience Level**: Specify {experience_level}.
    - **Company Overview**: A brief description of a fictional company and its mission.
    - **Job Overview**: A summary of the role’s purpose and impact.
    - **Responsibilities**: A detailed list (5-7 items) of key duties, incorporating {skills}.
    - **Required Skills and Qualifications**: A detailed list (5-7 items) including {skills} and {experience_level}-relevant qualifications.
    - **Preferred Skills**: Optional skills that enhance candidacy (2-3 items).
    - **Benefits**: A list of typical benefits (e.g., health insurance, remote work).
    - **Application Process**: Instructions for applying (e.g., submit resume and cover letter).
    Ensure the output is well-structured, uses markdown formatting, and is tailored to the inputs while drawing inspiration from the context.
    """
    return Task(
        description=prompt,
        agent=jd_generator,
        expected_output="A detailed job description in markdown format with multiple sections."
    )

if __name__ == "__main__":
    job_title = "Senior Python Developer"
    skills = "Python, Flask, SQL, AWS"
    experience_level = "5+ years"
    task = create_jd_task(job_title, skills, experience_level)
    crew = Crew(agents=[jd_generator], tasks=[task], verbose=True)
    result = crew.kickoff()
    print("Generated Job Description:\n", result)