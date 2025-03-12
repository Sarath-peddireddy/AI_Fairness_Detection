from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from ..config import PDF_DIRECTORY, HUGGINGFACE_API_KEY, EMBEDDING_MODEL

def load_pdfs(pdf_directory=PDF_DIRECTORY):
    """Load PDFs from a specified folder."""
    loader = DirectoryLoader(pdf_directory, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from {pdf_directory}")
    return documents

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    print(f"Generated {len(chunks)} chunks from documents")
    return chunks

def get_embeddings():
    # Load the embeddings model correctly
    embeddings =  HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

def embed_chunks(chunks):
    """Generate embeddings for chunks."""
    embeddings = get_embeddings()
    embeddings_list = embeddings.embed_documents([chunk.page_content for chunk in chunks])
    print(f"Generated embeddings for {len(embeddings_list)} chunks")
    return embeddings_list
