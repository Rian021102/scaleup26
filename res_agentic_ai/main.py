from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv

# Assuming these imports are properly defined as per your system setup
from phi.knowledge.langchain import LangChainKnowledgeBase
from phi.model.openai import OpenAIChat
from phi.agent import Agent
from langchain.embeddings import OpenAIEmbeddings
from phi.tools.python import PythonTools
from phi.model.anthropic import Claude
from langchain_community.vectorstores import DeepLake

# Load environment variables
load_dotenv()

app = FastAPI(title="Reservoir Engineering Assistant API")

# Environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("api_key")
ACTIVELOOP_TOKEN = os.getenv("ACTIVELOOP_TOKEN")

# Validate environment variables
if not api_key:
    raise ValueError("API_KEY environment variable is not set")
if not ACTIVELOOP_TOKEN:
    raise ValueError("ACTIVELOOP_TOKEN environment variable is not set")

# Initialize OpenAI embeddings
try:
    embeddings_model = OpenAIEmbeddings()
except Exception as e:
    raise Exception(f"Failed to initialize OpenAI embeddings: {str(e)}")

# Global variable for database instance
_db_instance = None

def get_db_instance():
    global _db_instance
    if _db_instance is None:
        _db_instance = DeepLake(dataset_path="hub://rian/reseng", embedding=embeddings_model, read_only=True)
    return _db_instance

# Initialize agents
db = get_db_instance()
retriever = db.as_retriever()
knowledge_base = LangChainKnowledgeBase(retriever=retriever)

knowledge_agent = Agent(
    name="RAG Agent",
    role="A reservoir engineer who is responsible to retrieve information from the knowledge base",
    instructions="Answer the following question based only on the provided context. Your answers must be based on the document and please provide detailed answers. If you don't know the answer, just say you don't know. Don't try to make up an answer",
    model=Claude(id="claude-3-7-sonnet-20250219", api_key=api_key),
    knowledge=knowledge_base,
    add_context=True,
    search_knowledge=True,
    markdown=True,
)

# Pydantic model for request
class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(query: Query):
    try:
        response = knowledge_agent.run(query.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Reservoir Engineering Assistant API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
