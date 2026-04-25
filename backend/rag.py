import os
import shutil
import uuid
import time
import pickle
import pytesseract
from dotenv import load_dotenv
from pdf2image import convert_from_path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document

load_dotenv()

CHROMA_DIR = "./chroma_db"
CACHE_MARKER = f"{CHROMA_DIR}/.pdf_source"
BM25_PATH = f"{CHROMA_DIR}/bm25.pkl"
PARENT_STORE_PATH = f"{CHROMA_DIR}/parent_store.pkl"

# Global stores
parent_store = InMemoryStore()
bm25_retriever = None

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def load_pdf_with_ocr(pdf_path: str):
    """Convert PDF pages to images and extract text via OCR."""
    print("   🔍 Scanned PDF detected, using OCR...")
    images = convert_from_path(pdf_path, dpi=200)
    docs = []
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"page": i, "source": pdf_path}
            ))
    print(f"   Extracted text from {len(docs)} pages via OCR")
    return docs

def process_pdf(pdf_path: str):
    """Load, chunk with parent-child strategy, embed and store a PDF."""
    global parent_store, bm25_retriever
    print(f"📄 Processing: {pdf_path}")

    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    parent_store = InMemoryStore()
    bm25_retriever = None

    # Load PDF — fallback to OCR if needed
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    pages = [p for p in pages if p.page_content.strip()]
    if not pages:
        pages = load_pdf_with_ocr(pdf_path)
    if not pages:
        raise ValueError("Could not extract any text from this PDF.")

    print(f"   Loaded {len(pages)} pages with content")

    # Parent chunks — large, for LLM context
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=100
    )
    parent_chunks = parent_splitter.split_documents(pages)
    parent_chunks = [p for p in parent_chunks if p.page_content.strip()]
    print(f"   Created {len(parent_chunks)} parent chunks")

    # Child chunks — small, for precise search
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=30
    )

    child_chunks = []
    for parent in parent_chunks:
        parent_id = str(uuid.uuid4())
        parent_store.mset([(parent_id, parent)])
        children = child_splitter.split_documents([parent])
        for child in children:
            if not child.page_content.strip():
                continue
            child.metadata["parent_id"] = parent_id
            child_chunks.append(child)

    print(f"   Created {len(child_chunks)} child chunks")

    # Build BM25 retriever on child chunks
    bm25_retriever = BM25Retriever.from_documents(child_chunks)
    bm25_retriever.k = 5

    # Save BM25 and parent store to disk so they survive restarts
    os.makedirs(CHROMA_DIR, exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)
    with open(PARENT_STORE_PATH, "wb") as f:
        pickle.dump(dict(parent_store.store), f)

    # Embed child chunks in batches
    embeddings = get_embeddings()
    batch_size = 5
    batches = [child_chunks[i:i+batch_size] for i in range(0, len(child_chunks), batch_size)]
    vectorstore = None
    for i, batch in enumerate(batches):
        print(f"   Embedding batch {i+1}/{len(batches)}...")
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=CHROMA_DIR
            )
        else:
            vectorstore.add_documents(batch)
        time.sleep(35)  # wait 35 seconds between batches to respect rate limit

    print("   ✅ Vector store ready")
    return vectorstore

def load_retrievers():
    """Load ChromaDB, BM25 and parent store from disk."""
    global parent_store, bm25_retriever

    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    with open(BM25_PATH, "rb") as f:
        bm25_retriever = pickle.load(f)

    with open(PARENT_STORE_PATH, "rb") as f:
        store_dict = pickle.load(f)
        parent_store = InMemoryStore()
        parent_store.mset(list(store_dict.items()))

    return vectorstore

def is_pdf_processed():
    return os.path.exists(CHROMA_DIR) and os.path.exists(CACHE_MARKER)

def hybrid_retrieve(question: str, vectorstore) -> list:
    """Merge semantic + keyword results, then fetch parent chunks."""

    # Semantic search — finds meaning-similar chunks
    semantic_results = vectorstore.similarity_search(question, k=5)

    # Keyword search — finds exact word matches
    keyword_results = bm25_retriever.invoke(question)

    # Merge and deduplicate child chunks
    seen = set()
    merged_children = []
    for doc in semantic_results + keyword_results:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            merged_children.append(doc)

    # Fetch parent chunks for full context
    parent_ids = list({
        doc.metadata["parent_id"]
        for doc in merged_children
        if "parent_id" in doc.metadata
    })

    parent_docs = parent_store.mget(parent_ids)
    relevant_docs = [doc for doc in parent_docs if doc is not None]

    # Fallback to children if no parents found
    return relevant_docs if relevant_docs else merged_children


def ask_question(question: str) -> dict:
    """Ask a question using hybrid retrieval with answer grounding."""
    vectorstore = load_retrievers()
    relevant_docs = hybrid_retrieve(question, vectorstore)

    if not relevant_docs:
        return {
            "answer": "I don't have enough information in the document to answer that.",
            "chunks_used": 0,
            "grounded": False
        }

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Step 1 — Grounding check
    # Ask the LLM if the context actually contains the answer
    grounding_prompt = f"""You are a strict fact checker.

Given this context from a document:
{context}

Can this question be answered using ONLY the context above?
Question: {question}

Reply with ONLY one word: YES or NO."""

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    grounding_response = llm.invoke(grounding_prompt)
    is_grounded = "YES" in grounding_response.content.strip().upper()

    if not is_grounded:
        return {
            "answer": "I couldn't find information about that in the uploaded document. Try rephrasing or ask something else.",
            "chunks_used": len(relevant_docs),
            "grounded": False
        }

    # Step 2 — Answer the question
    answer_prompt = f"""Use the following context to answer the question.
Be precise and only use information from the context.
If exact figures or names are in the context, include them in your answer.

Context:
{context}

Question: {question}
"""
    answer_response = llm.invoke(answer_prompt)

    return {
        "answer": answer_response.content,
        "chunks_used": len(relevant_docs),
        "grounded": True
    }