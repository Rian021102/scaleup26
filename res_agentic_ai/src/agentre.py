from langchain_openai import ChatOpenAI
from phi.knowledge.langchain import LangChainKnowledgeBase
from phi.model.openai import OpenAIChat
from phi.agent import Agent
from langchain.embeddings import OpenAIEmbeddings
from phi.utils.pprint import pprint_run_response
from phi.tools.python import PythonTools
import os
from langchain_community.vectorstores import DeepLake
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai
from phi.model.google import Gemini
from google import genai



api_key = os.getenv("OPENAI_API_KEY")
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ACTIVELOOP_TOKEN = os.getenv("ACTIVELOOP_TOKEN")
embeddings_model = OpenAIEmbeddings()


db = DeepLake(dataset_path="hub://rian/reseng", embedding=embeddings_model, read_only=True)

# Set up the retriever
retriever = db.as_retriever()



knowledge_base = LangChainKnowledgeBase(retriever=retriever)

knowledge_agent = Agent(
    name="RAG Agent",
    role="You are a reservoir engineer. You have to answer questions within the context of the documents. If you don't know the answer, just say you don't know, do not try to make up answers",
    instructions="Use the knowledge base to answer questions.",
    model=OpenAIChat(id="gpt-4"),
    knowledge=knowledge_base,
    add_context=True,
    search_knowledge=True,
    markdown=True,
)

formula_agent = Agent(
    name="Formula Agent",
    role="You are a Python coder who works with reservoir engineer to write code for formula in reservoir engineering. Not only you write the code, you can also answer question when given inputs with Python formula you write",
    instructions="Use formula provided to write the python code and calculate using the formula you write when asked",
    tools=[PythonTools()],
    model=OpenAIChat(id="gpt-4"),
)

latex_agent = Agent(name="Latex Agent", 
                    role="You are a Latex expert who can convert the formula to Latex", 
                    model=OpenAIChat(id="gpt-4"))


answering_agent = Agent(
    name="Answering Agent",
    role="You are the head of the reservoir engineering team, who gets answer from the reservoir engineer and provide the calculation from the formula agent, if needed based on the contextual question",
    instructions=[
        "1. Provide answers within the context of the documents.",
        "2. When asked for formula, get the answer from the document, then route the question to latext agent to write the formula in correct mathematical latex format.",
        "3. When asked for formula, get the answer from the document, then route the question from latex agent to the Formula Agent, to write the python code for the formula.",
        "4. When provided with the inputs, use the python code formula the provide the answer.",
        "5. When you calculate don't forget inside the formula need to add delimiter for each input",],
    team=[knowledge_agent,latex_agent, formula_agent],
    show_tool_calls=True,
    markdown=True,
)

response=answering_agent.run("What is the gas solubility of the reservoir with known bubble point pressure using the Standing-Katz correlation?")
pprint_run_response(response, markdown=True)