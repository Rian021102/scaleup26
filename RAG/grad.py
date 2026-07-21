import gradio as gr
import chromadb
import pymupdf
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama


PERSIST_DIRECTORY = Path(__file__).resolve().parent / "chroma_store.db"


def get_embeddings_model() -> OllamaEmbeddings:
    """Create embedding model used by ChromaDB indexing and retrieval."""
    return OllamaEmbeddings(model="nomic-embed-text:latest")


def resolve_persist_directory(persist_directory: str | Path | None = None) -> Path:
    """Resolve and create the Chroma persistence directory."""
    target_dir = Path(persist_directory) if persist_directory else PERSIST_DIRECTORY
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def pdf_to_text(uploaded_file) -> str:
    """Extract text from a Gradio-uploaded PDF (raw bytes or a file-like object)."""
    file_bytes = uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file
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


def store_chunks_in_chromadb(
    chunks: list[str],
    document_id: str,
    collection_name: str = "pdf_chunks",
    persist_directory: str | Path | None = None,
) -> None:
    """Store text chunks in a ChromaDB collection."""
    db_path = resolve_persist_directory(persist_directory)
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(name=collection_name)

    embeddings = get_embeddings_model()
    chunk_embeddings = embeddings.embed_documents(chunks)

    collection.upsert(
        documents=chunks,
        metadatas=[{"chunk_index": i, "document_id": document_id} for i in range(len(chunks))],
        ids=[f"{document_id}_chunk_{i}" for i in range(len(chunks))],
        embeddings=chunk_embeddings,
    )


def collection_has_data(
    collection_name: str = "pdf_chunks",
    persist_directory: str | Path | None = None,
) -> bool:
    """Return True when the collection has at least one indexed chunk."""
    db_path = resolve_persist_directory(persist_directory)
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(name=collection_name)
    return collection.count() > 0


def query_chromadb(
    query: str,
    collection_name: str = "pdf_chunks",
    persist_directory: str | Path | None = None,
    top_k: int = 5,
) -> list[str]:
    """Retrieve relevant chunks from ChromaDB for a query."""
    db_path = resolve_persist_directory(persist_directory)
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection(name=collection_name)

    embeddings = get_embeddings_model()
    query_embedding = embeddings.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return results["documents"][0] if results["documents"] else []


def ingest_pdf(uploaded_file: bytes):
    """Read uploaded PDF bytes, chunk text, and store embeddings."""
    if uploaded_file is None:
        return "Please upload a PDF file."

    try:
        pdf_text = pdf_to_text(uploaded_file)
    except Exception as exc:
        return f"Failed to read PDF: {exc}"

    if not pdf_text.strip():
        return "No text could be extracted from this PDF."

    chunks = chunk_text(pdf_text)
    if not chunks:
        return "No chunks were generated from this PDF."

    store_chunks_in_chromadb(chunks=chunks, document_id="uploaded_pdf")
    return f"Indexed {len(chunks)} chunks successfully. You can now ask questions."


def answer_question(user_query: str) -> str:
    """Answer a query using retrieved PDF chunks as context."""
    if not user_query or not user_query.strip():
        return "Enter a question first."

    if not collection_has_data():
        return "Upload and index a PDF first."

    relevant_chunks = query_chromadb(user_query)
    if not relevant_chunks:
        return "No relevant information found for your query."

    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are a helpful assistant. Answer only from the provided context. "
        "If context is insufficient, say so clearly.\n\n"
        f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"
    )

    model_ollama = ChatOllama(model="gemma3:4b")
    response = model_ollama.invoke(prompt)
    return response.content


with gr.Blocks(title="RAG PDF Loader") as demo:
    gr.Markdown("# RAG for Oil and Gas Professionals")
    gr.Markdown("Upload a PDF, index it into ChromaDB, then ask questions.")

    with gr.Row():
        pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"], type="binary")
        ingest_btn = gr.Button("Load PDF")

    ingest_status = gr.Textbox(label="Indexing Status", interactive=False)

    question = gr.Textbox(label="Ask a question", placeholder="What does the document say about...?")
    ask_btn = gr.Button("Ask")
    answer = gr.Textbox(label="Answer", lines=8, interactive=False)

    ingest_btn.click(fn=ingest_pdf, inputs=pdf_file, outputs=ingest_status)
    ask_btn.click(fn=answer_question, inputs=question, outputs=answer)


if __name__ == "__main__":
    demo.launch()
