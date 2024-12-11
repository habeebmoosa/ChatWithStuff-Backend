from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
import logging
import hashlib
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

VECTOR_STORE_PATH = "./vector_stores"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

CURRENT_DOCUMENT = None
CURRENT_VECTORSTORE = None

class ChatRequest(BaseModel):
    question: str

def generate_document_hash(filename: str) -> str:
    """Generate a unique hash for a given document name."""
    return hashlib.md5(filename.encode()).hexdigest()

def load_or_create_vectorstore(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Load existing vector store or create a new one if it doesn't exist.
    
    Args:
        file_path (str): Path to the PDF document
        chunk_size (int): Size of text chunks
        chunk_overlap (int): Overlap between text chunks
    
    Returns:
        Chroma vectorstore
    """
    global CURRENT_DOCUMENT, CURRENT_VECTORSTORE
    
    document_hash = generate_document_hash(os.path.basename(file_path))
    vectorstore_path = os.path.join(VECTOR_STORE_PATH, document_hash)

    try:
        if CURRENT_DOCUMENT == file_path and CURRENT_VECTORSTORE:
            return CURRENT_VECTORSTORE
        
        if os.path.exists(vectorstore_path):
            logger.info(f"Loading existing vectorstore for {file_path}")
            vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)
        else:
            logger.info(f"Creating new vectorstore for {file_path}")
            
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            splits = text_splitter.split_documents(docs)

            
            vectorstore = Chroma.from_documents(
                documents=splits, 
                embedding=embeddings, 
                persist_directory=vectorstore_path
            )
        
        CURRENT_DOCUMENT = file_path
        CURRENT_VECTORSTORE = vectorstore
        
        return vectorstore

    except Exception as e:
        logger.error(f"Error processing vectorstore for {file_path}: {e}")
        raise

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@router.post("/initialize")
async def initialize_document(file: UploadFile = File(...)):
    """Initialize the vector store for a PDF document."""
    try:
        
        file_location = f"./temp_uploads/{file.filename}"
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        
        with open(file_location, "wb+") as file_object:
            file_object.write(await file.read())
        
        
        vectorstore = load_or_create_vectorstore(file_location)
        
        return {"message": f"Document {file.filename} loaded successfully!"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
async def chat_with_document(request: ChatRequest):
    """Chat with the current loaded document."""
    try:
        if not CURRENT_VECTORSTORE:
            raise HTTPException(status_code=400, detail="No document initialized. Call /initialize first.")
        
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