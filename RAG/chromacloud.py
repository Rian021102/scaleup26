import os
from pathlib import Path
import streamlit as st
import chromadb
import pymupdf
from dotenv import load_dotenv
from openai import OpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

# Load Chroma Cloud credentials from the .env file next to this script.
load_dotenv(Path(__file__).resolve().parent / ".env")

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise RuntimeError("HF_TOKEN is not set. Add it to your .env file.")

'''
This is a very basic simple example of RAG (Retrieval-Augmented Generation) using Streamlit, ChromaDB, Ollama and Langchain.
If you want to use ChatGPT make sure you need to stash you API key in your environment variable.

ChromaDB Cloud credentials are read from environment variables (or Streamlit secrets):
  CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE
'''

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=hf_token,
)


@st.cache_resource
def get_embeddings_model() -> OllamaEmbeddings:
    """Create and cache embedding model across Streamlit reruns."""
    return OllamaEmbeddings(model="nomic-embed-text:latest")


@st.cache_resource
def get_chroma_client() -> chromadb.api.ClientAPI:
    """Create and cache a ChromaDB Cloud client across Streamlit reruns."""
    return chromadb.CloudClient(
        api_key=os.environ["CHROMA_API_KEY"],
        tenant=os.environ["CHROMA_TENANT"],
        database=os.environ["CHROMA_DATABASE"],
    )

def pdf_to_text(uploaded_file) -> str:
    """Extract text from a Streamlit uploaded PDF."""
    file_bytes = uploaded_file.read()
    if not file_bytes:
        return ""

    with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
        return "".join(page.get_text() for page in doc)

def chunk_text(raw_text: str) -> list[str]:
    """Split text into semantically coherent chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,        # ~512 characters per chunk
        chunk_overlap=64,      # 64-char overlap to preserve context at boundaries
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(raw_text)

#embeddings_model = OllamaEmbeddings(model="nomic-embed-text:latest") store to chromadb and use persistent storage for embeddings and metadata. This allows for efficient retrieval of relevant chunks based on user queries.

def store_chunks_in_chromadb(
    chunks: list[str],
    document_id: str,
    collection_name: str = "pdf_chunks",
):
    """Store text chunks in a ChromaDB collection."""
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=collection_name)

    # Create embeddings for each chunk
    embeddings = get_embeddings_model()
    chunk_embeddings = embeddings.embed_documents(chunks)

    # Store chunks and their embeddings in the collection.
    # upsert (instead of add) so re-uploading the same document won't error on duplicate IDs.
    collection.upsert(
        documents=chunks,
        metadatas=[{"chunk_index": i, "document_id": document_id} for i in range(len(chunks))],
        ids=[f"{document_id}_chunk_{i}" for i in range(len(chunks))],
        embeddings=chunk_embeddings,
    )


def collection_has_data(
    collection_name: str = "pdf_chunks",
) -> bool:
    """Return True if the collection already exists and contains at least one chunk."""
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=collection_name)
    return collection.count() > 0

def query_chromadb(
    query: str,
    collection_name: str = "pdf_chunks",
    top_k: int = 5,
) -> list[str]:
    """Query the ChromaDB collection for relevant chunks."""
    client = get_chroma_client()
    collection = client.get_collection(name=collection_name)

    # Create embedding for the query
    embeddings = get_embeddings_model()
    query_embedding = embeddings.embed_query(query)

    # Perform similarity search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    return results["documents"][0] if results["documents"] else []



def main():
    st.title("RAG for OIL AND GAS PROFESSIONALS")
    st.subheader("This is a prototype of a Retrieval-Augmented Generation (RAG) system for oil and gas professionals. It allows users to input queries and retrieve relevant information from a knowledge base.")
    # Track which PDFs have already been ingested so we don't re-embed on every rerun.
    if "ingested_files" not in st.session_state:
        st.session_state.ingested_files = set()

    # Always show the uploader. Uploading a new PDF appends its chunks to the vector DB.
    uploaded_file = st.file_uploader("Upload a PDF file (optional if you already have documents)", type=["pdf"])
    if uploaded_file is not None and uploaded_file.name not in st.session_state.ingested_files:
        # Convert PDF to text
        try:
            pdf_text = pdf_to_text(uploaded_file)
        except Exception as exc:
            st.error(f"Failed to read PDF: {exc}")
            return

        if not pdf_text.strip():
            st.warning("No text could be extracted from this PDF.")
            return

        with st.spinner(f"Indexing {uploaded_file.name}..."):
            chunks = chunk_text(pdf_text)
            # Store chunks in ChromaDB, using the filename as the document identifier
            store_chunks_in_chromadb(chunks, document_id=uploaded_file.name)
        st.session_state.ingested_files.add(uploaded_file.name)
        st.success(f"Added {len(chunks)} chunks from {uploaded_file.name} to the knowledge base.")

    # Show the query box whenever the vector DB already has data (from this or a previous session).
    if not collection_has_data():
        st.info("Upload a PDF to get started.")
        return

    user_query = st.text_input("Enter your query:")
    if user_query:
        # Retrieve relevant chunks from ChromaDB
        relevant_chunks = query_chromadb(user_query)
        if relevant_chunks:
            context = "\n\n".join(relevant_chunks)
            prompt = (
                "You are a helpful assistant answering questions using only the provided context. "
                "If the context is insufficient, say so clearly.\n\n"
                f"Context:\n{context}\n\nQuestion: {user_query}"
            )
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b:cerebras",
                messages=[{"role": "user", "content": prompt}],
            )
            st.write("Generated response:")
            st.write(response.choices[0].message.content)
        else:
            st.warning("No relevant information found for your query.")


if __name__ == "__main__":
    main()