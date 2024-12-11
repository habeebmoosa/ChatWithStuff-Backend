from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import os
import logging
import hashlib
from dotenv import load_dotenv

import pandas as pd
import numpy as np

load_dotenv()

router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

VECTOR_STORE_PATH = "./vector_stores_excel"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

CURRENT_DOCUMENT = None
CURRENT_VECTORSTORE = None
CURRENT_DATAFRAME = None

class ChatRequest(BaseModel):
    question: str

def generate_document_hash(filename: str) -> str:
    """Generate a unique hash for a given document name."""
    return hashlib.md5(filename.encode()).hexdigest()

def preprocess_dataframe(df: pd.DataFrame) -> str:
    """
    Convert DataFrame to a text representation suitable for embedding.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        str: Textual representation of the DataFrame
    """
    # Convert numeric columns to string with limited precision
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
    
    # Convert DataFrame to string with column names and values
    text_representation = []
    
    # Add column names and data types
    column_info = "Columns: " + ", ".join([
        f"{col} ({df[col].dtype})" for col in df.columns
    ])
    text_representation.append(column_info)
    
    # Add first few rows as context
    for idx, row in df.head(10).iterrows():
        row_text = f"Row {idx}: " + ", ".join([
            f"{col}: {val}" for col, val in row.items() if pd.notnull(val)
        ])
        text_representation.append(row_text)
    
    # Add summary statistics as text
    summary_stats = df.describe().transpose()
    summary_text = "Summary Statistics: " + ", ".join([
        f"{col}: [min: {summary_stats.loc[col, 'min']}, "
        f"max: {summary_stats.loc[col, 'max']}, "
        f"mean: {summary_stats.loc[col, 'mean']}]" 
        for col in summary_stats.index if col in numeric_cols
    ])
    text_representation.append(summary_text)
    
    return "\n".join(text_representation)

def load_or_create_vectorstore(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Load existing vector store or create a new one for an Excel file.
    """
    global CURRENT_DOCUMENT, CURRENT_VECTORSTORE, CURRENT_DATAFRAME
    
    document_hash = generate_document_hash(os.path.basename(file_path))
    vectorstore_path = os.path.join(VECTOR_STORE_PATH, document_hash)

    try:
        # Load Excel file with error handling
        try:
            df = pd.read_excel(file_path)
            if df.empty:
                raise ValueError("The Excel file is empty")
        except Exception as read_error:
            logger.error(f"Error reading Excel file: {read_error}")
            raise HTTPException(status_code=400, detail=f"Could not read Excel file: {str(read_error)}")
        
        CURRENT_DATAFRAME = df
        
        # Convert DataFrame to text representation with error handling
        try:
            text_representation = preprocess_dataframe(df)
        except Exception as preprocess_error:
            logger.error(f"Error preprocessing DataFrame: {preprocess_error}")
            raise HTTPException(status_code=500, detail=f"Could not preprocess Excel data: {str(preprocess_error)}")
        
        # Rest of the existing vectorstore creation logic remains the same
        if os.path.exists(vectorstore_path):
            logger.info(f"Loading existing vectorstore for {file_path}")
            vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)
        else:
            logger.info(f"Creating new vectorstore for {file_path}")
            
            from langchain_core.documents import Document
            docs = [Document(page_content=text_representation)]
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
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
        logger.error(f"Comprehensive error processing vectorstore for {file_path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@router.post("/initialize")
async def initialize_document(file: UploadFile = File(...)):
    """Initialize the vector store for an Excel document."""
    try:
        file_location = f"./temp_uploads/{file.filename}"
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        
        with open(file_location, "wb+") as file_object:
            file_object.write(await file.read())
        
        vectorstore = load_or_create_vectorstore(file_location)
        
        return {"message": f"Excel document {file.filename} loaded successfully!"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
async def chat_with_document(request: ChatRequest):
    """Chat with the current loaded Excel document."""
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

@router.get("/summary")
async def get_dataframe_summary():
    """Retrieve summary of the current loaded DataFrame."""
    try:
        if CURRENT_DATAFRAME is None:
            raise HTTPException(status_code=400, detail="No Excel document loaded")
        
        # Provide a comprehensive summary
        summary = {
            "columns": list(CURRENT_DATAFRAME.columns),
            "row_count": len(CURRENT_DATAFRAME),
            "column_types": {col: str(CURRENT_DATAFRAME[col].dtype) for col in CURRENT_DATAFRAME.columns},
            "numeric_summary": CURRENT_DATAFRAME.describe().to_dict()
        }
        
        return summary
    
    except Exception as e:
        logger.error(f"Summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))