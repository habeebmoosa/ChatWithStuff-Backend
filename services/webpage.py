from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import logging
import hashlib

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
VECTOR_STORE_PATH = "./vector_stores"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

CURRENT_URL = None
CURRENT_VECTORSTORE = None

class InitializeRequest(BaseModel):
    web_url: str
    chunk_size: int = 1000
    chunk_overlap: int = 200

class ChatRequest(BaseModel):
    question: str

def generate_url_hash(url: str) -> str:
    """Generate a unique hash for a given URL."""
    return hashlib.md5(url.encode()).hexdigest()

def load_or_create_vectorstore(web_url: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Load existing vector store or create a new one if it doesn't exist.
    
    Args:
        web_url (str): URL of the web content
        chunk_size (int): Size of text chunks
        chunk_overlap (int): Overlap between text chunks
    
    Returns:
        Chroma vectorstore
    """
    global CURRENT_URL, CURRENT_VECTORSTORE
    
    url_hash = generate_url_hash(web_url)
    vectorstore_path = os.path.join(VECTOR_STORE_PATH, url_hash)

    try:
        if CURRENT_URL == web_url and CURRENT_VECTORSTORE:
            return CURRENT_VECTORSTORE
        
        if os.path.exists(vectorstore_path):
            logger.info(f"Loading existing vectorstore for {web_url}")
            vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)
        else:
            logger.info(f"Creating new vectorstore for {web_url}")
            loader = WebBaseLoader(web_paths=(web_url,))
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            splits = text_splitter.split_documents(docs)

            vectorstore = Chroma.from_documents(
                documents=splits, 
                embedding=embeddings, 
                persist_directory=vectorstore_path
            )
        
        CURRENT_URL = web_url
        CURRENT_VECTORSTORE = vectorstore
        
        return vectorstore

    except Exception as e:
        logger.error(f"Error processing vectorstore for {web_url}: {e}")
        raise

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@router.post("/initialize")
async def initialize_content(request: InitializeRequest):
    """Initialize the vector store for a specific URL."""
    try:
        vectorstore = load_or_create_vectorstore(
            request.web_url, 
            request.chunk_size, 
            request.chunk_overlap
        )
        return {"message": f"Web content from {request.web_url} loaded successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
async def chat_with_model(request: ChatRequest):
    """Chat with the current loaded web content."""
    try:
        if not CURRENT_VECTORSTORE:
            raise HTTPException(status_code=400, detail="No web content initialized. Call /initialize first.")
        
        retriever = CURRENT_VECTORSTORE.as_retriever()
        
        prompt = hub.pull("rlm/rag-prompt")
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        response = rag_chain.invoke(request.question)
        
        return {"response": response}
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))