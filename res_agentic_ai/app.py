from phi.knowledge.langchain import LangChainKnowledgeBase
from phi.agent import Agent, RunResponse
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from phi.utils.pprint import pprint_run_response
from phi.tools.python import PythonTools
from phi.model.anthropic import Claude
import os
from langchain_community.vectorstores import DeepLake



openai_api_key = st.secrets["OPENAI_API_KEY"]
api_key = st.secrets["api_key"]
ACTIVELOOP_TOKEN = st.secrets["ACTIVELOOP_TOKEN"]

# Validate environment variables
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
if not ACTIVELOOP_TOKEN:
    raise ValueError("ACTIVELOOP_TOKEN environment variable is not set")

# # Initialize OpenAI embeddings with error handling
try:
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
except Exception as e:
    raise Exception(f"Failed to initialize OpenAI embeddings: {str(e)}")



# Global variable to store the database instance
_db_instance = None

def get_db_instance():
    global _db_instance
    if _db_instance is None:
        _db_instance = DeepLake(dataset_path="hub://rian/reseng", embedding=embeddings_model, read_only=True)
    return _db_instance

# Set up the retriever
db = get_db_instance()
retriever = db.as_retriever()
knowledge_base = LangChainKnowledgeBase(retriever=retriever)

knowledge_agent = Agent(
    name="RAG Agent",
    role="A reservoir engineer who is responsible to retrieve information from the knowledge base",
    instructions=" Answer the following question based only on the provided context.Your answers must be based on the document and please provide detailed answers. If you don't know the answer, just say you don't know. Don't try to make up an answer",
    model=Claude(id="claude-3-5-sonnet-20241022", api_key=api_key),
    knowledge=knowledge_base,
    add_context=True,
    search_knowledge=True,
    markdown=True,
)


formula_agent = Agent(
    description='A math expert in mathematical formula',
    name="Formula Agent",
    role='An expert in writing complex mathematical formula and re-arrenging the formula to answer the questions',
    instructions="Provide the formula from the  knowledge_agent. You need to re-arrenging the formula to answer the questions when needed",
    model=Claude(id="claude-3-5-sonnet-20241022", api_key=api_key),
    add_context=True,
    markdown=True,

)

latex_agent = Agent(
    description='An expert in re-writing formula in Latex format',
    name="Latex Agent", 
    role="An expert in re-writing formula asked in query from the formula_agent in Latex format", 
    instructions="Provide formula in Latex format",
    model=Claude(id="claude-3-5-sonnet-20241022", api_key=api_key),
    markdown=True,
)



python_agent = Agent(
    description='An expert in writing python code to provide answers when there is numerical input related to formula',
    name="Python Agent",
    role="An python developer who exeperts in writing code in python to provide answers when there is query with numerical inputs",
    model=Claude(id="claude-3-5-sonnet-20241022", api_key=api_key),
    instructions="Provide the python agent from the mathematical formula agent to write the python code for the formula and use it to provide answers when query consists of numerical inputs",
    tools=[PythonTools()],
)

answering_agent = Agent(
    name="Answering Agent",
    role="You are the head of the reservoir engineering team, who gets answer from the reservoir engineer and provide the calculation from the formula agent, if needed based on the contextual question",
    instructions=[
        "1. Retrieve information based on query from vector database",
        "2. When answers have formula give to formula agent to write the formula. If the questions need to re-arrange the formula please do so.",
        "3. Route to latex agent to convert the formula to latex format",
        "4. When the question/query have numerical inputs, pass the formula to python_agent to write the code to answers the query",],
    team=[knowledge_agent, formula_agent, latex_agent, python_agent],
    show_tool_calls=True,
    markdown=True,
)


# Streamlit UI
st.set_page_config(page_title="Reservoir Engineering AI", page_icon=":earth_americas:", layout="wide")
st.title("Reservoir Engineering AI")
#set up the sidebar
st.sidebar.title("Ask a question")
st.sidebar.write("This A.I app simply to help you to answer the question related to reservoir engineering")
st.sidebar.write("Hi, My name is Rian Rachmanto")
st.sidebar.write("Let's connect : https://www.linkedin.com/in/rian-rachmanto/")
#set an area for the user to input the question
question = st.sidebar.text_area("Type your question here",
                                height=100)

if st.sidebar.button("Ask"):
    response: RunResponse = knowledge_agent.run(question)
    st.write(response.content)