import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Check if required API keys are present
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
if not SERPAPI_API_KEY:
    raise ValueError("SERPAPI_API_KEY not found in environment variables")
if not HUGGINGFACE_API_KEY:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables")

# Directory configurations
PDF_DIRECTORY = "backend/data/pdfs"
CHROMA_PERSIST_DIR = "./chroma_db"

# Model configurations
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "BAAI/bge-reranker-base"