
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from googlesearch import search
from dotenv import load_dotenv
import os
from bs4 import BeautifulSoup
import re
load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-70b-8192",
    temperature=0.7,  
    max_tokens=1000
)

interview_agent = Agent(
    role="Interview Agent",
    goal="Conduct interactive AI-driven interviews with questions enriched by real-world data",
    backstory="A skilled interviewer leveraging industry insights for precise questioning",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    memory=True  
)
def html_to_text(html_content: str) -> str:
    soup = BeautifulSoup(html_content, 'html.parser')
    
    
    text = soup.get_text(separator=" ").strip()

    # Remove excessive multiple spaces
    text = re.sub(r'\s+', ' ', text) 
# Fetch related interview questions from Google
def fetch_related_questions(job_description):
    query = f"{job_description} interview questions site:*.edu | site:*.org | site:*.gov -inurl:(signup | login)"
    urls = list(search(query, num_results=5))
    
    documents = []
    for url in urls:
        try:
            loader = RecursiveUrlLoader(url=url,extractor=html_to_text,max_depth=1,
                                headers={"User-Agent": "Mozilla/5.0"}) 
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {url}: {e}")
    return documents

# Store content in VDB
def store_in_vdb(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vdb = FAISS.from_documents(chunks, embeddings)
    return vdb

# Generate enriched interview question
def generate_enriched_question(job_description, conversation_history, vdb):
    query = f"Interview questions for {job_description}"
    if conversation_history:
        query += f" based on prior responses: {conversation_history[-1]['content']}"
    similar_docs = vdb.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in similar_docs])
    
    if conversation_history is None or not conversation_history:
        prompt = f"Using this job description: '{job_description}' and context from similar interview questions: '{context}', ask an initial open-ended question to assess the candidate's experience."
    else:
        prompt = f"Given this job description: '{job_description}', conversation history: {conversation_history}, and context from similar questions: '{context}', ask a relevant follow-up question."
    
    return Task(
        description=prompt,
        agent=interview_agent,
        expected_output="A single interview question tailored to the job description and conversation context."
    )

# Start or continue the interview with RAG
def create_interview_task(job_description, conversation_history=None):
    documents = fetch_related_questions(job_description)
    if not documents:
        # Fallback to basic question generation
        if conversation_history is None or not conversation_history:
            prompt = f"Start an interview for a candidate applying to this job: {job_description}. Ask an initial open-ended question to assess their experience."
        else:
            prompt = f"Given this job description: {job_description}, and the conversation history: {conversation_history}, ask a relevant follow-up question."
        return Task(
            description=prompt,
            agent=interview_agent,
            expected_output="A single interview question tailored to the job description and conversation context."
        )
    
    vdb = store_in_vdb(documents)
    return generate_enriched_question(job_description, conversation_history, vdb)

# Evaluate response and generate follow-up with RAG
def evaluate_response_task(job_description, conversation_history, candidate_response):
    documents = fetch_related_questions(job_description)
    if not documents:
        return Task(
            description=f"Evaluate this candidate response: '{candidate_response}' for the job: {job_description}. Based on the conversation history: {conversation_history}, generate a follow-up question.",
            agent=interview_agent,
            expected_output="A follow-up question based on the candidate's response."
        )
    
    vdb = store_in_vdb(documents)
    query = f"Follow-up interview questions for {job_description} based on response: {candidate_response}"
    similar_docs = vdb.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in similar_docs])
    
    return Task(
        description=f"Evaluate this candidate response: '{candidate_response}' for the job: {job_description}. Using conversation history: {conversation_history} and context: '{context}', generate a follow-up question.",
        agent=interview_agent,
        expected_output="A follow-up question based on the candidate's response."
    )

if __name__ == "__main__":
    job_description = "Senior Python Developer requiring Python, Flask, SQL, AWS, and 5+ years experience."
    task = create_interview_task(job_description)
    crew = Crew(agents=[interview_agent], tasks=[task], verbose=True)
    question = crew.kickoff()
    print("Initial Question:", question)

    conversation_history = [{"role": "agent", "content": str(question)}]
    candidate_response = "I've worked with Python and Flask for 4 years, mostly on web apps, but my AWS experience is limited to a few months."
    eval_task = evaluate_response_task(job_description, conversation_history, candidate_response)
    crew = Crew(agents=[interview_agent], tasks=[eval_task], verbose=True)
    follow_up = crew.kickoff()
    print("Follow-up Question:", follow_up)