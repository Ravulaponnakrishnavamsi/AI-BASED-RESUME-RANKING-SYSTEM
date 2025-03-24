from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from googlesearch import search
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os
import logging
from bs4 import BeautifulSoup
import re
load_dotenv()
logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-70b-8192",
    temperature=0.5,
    max_tokens=1000
)

resume_ranker = Agent(
    role="Resume Ranker",
    goal="Rank resumes based on job fit with fairness",
    backstory="An expert in evaluating resumes fairly",
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
def extract_text_from_pdf(file_path=None, file_content=None):
    if file_path:
        reader = PdfReader(file_path)
    elif file_content:
        reader = PdfReader(file_content)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def fetch_related_content(job_description):
    query = f"{job_description} site:*.edu | site:*.org | site:*.gov -inurl:(signup | login)"
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
    return FAISS.from_documents(chunks, embeddings)

def process_resumes(job_description, dir_path=None, uploaded_files=None):
    resumes = []
    if dir_path and os.path.isdir(dir_path):
        for filename in os.listdir(dir_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(dir_path, filename)
                resume_text = extract_text_from_pdf(file_path=file_path)
                resumes.append(f"Resume: {filename}\nContent: {resume_text}")
    elif uploaded_files:
        for uploaded_file in uploaded_files:
            resume_text = extract_text_from_pdf(file_content=uploaded_file)
            resumes.append(f"Resume: {uploaded_file.name}\nContent: {resume_text}")
    return resumes

def create_resume_rank_task(job_description, dir_path=None, uploaded_files=None):
    resumes = process_resumes(job_description, dir_path, uploaded_files)
    if not resumes:
        return None
    documents = fetch_related_content(job_description)
    vdb = store_in_vdb(documents) if documents else None
    context = vdb.similarity_search(job_description, k=3) if vdb else []
    context_text = "\n".join([doc.page_content for doc in context]) or "No context."
    prompt = f"Rank these resumes: {', '.join(resumes)} for '{job_description}' using context: '{context_text}'. Ensure fairness by avoiding bias based on gender, age, or ethnicity. Flag any potential bias in reasoning."
    return Task(
        description=prompt,
        agent=resume_ranker,
        expected_output="A ranked list with scores (0-100), reasoning, and bias flags."
    )