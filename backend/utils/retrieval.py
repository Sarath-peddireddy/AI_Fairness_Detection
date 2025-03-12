from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import EnsembleRetriever
from ..config import CHROMA_PERSIST_DIR, RERANKER_MODEL
import os
import chromadb
from chromadb.utils.embedding_functions import HuggingFaceEmbeddingFunction

def initialize_retrievers(chunks, embeddings):
    """Initialize all retrievers."""
    try:
        print("Creating or loading Chroma vector store...")
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        # Collection name
        collection_name = "document_collection"
        existing_collections = [col.name for col in client.list_collections()]

        if collection_name not in existing_collections:
            # Create a new collection
            print("Creating a new Chroma collection...")
            collection = client.create_collection(
                name=collection_name,
                embedding_function=HuggingFaceEmbeddingFunction(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
            )
        else:
            # Load the existing collection
            print("Loading existing Chroma collection...")
            collection = client.get_collection(name=collection_name)

        # Check if documents are already in the collection to avoid duplication
        if collection.count() == 0:
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [{"source": str(i)} for i in range(len(chunks))]
            ids = [str(i) for i in range(len(chunks))]

            print("Adding documents to the collection...")
            collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            print("Documents added successfully.")
        else:
            print("Documents already exist in the collection.")

        # Create Chroma retriever from LangChain
        print("Initializing Chroma retriever...")
        chroma_db = Chroma(
            collection_name=collection_name,
            persist_directory=CHROMA_PERSIST_DIR
        ).as_retriever()

        # Initialize BM25 retriever
        print("Creating BM25 retriever...")
        bm25_retriever = BM25Retriever.from_documents(chunks)

        # Initialize Cross Encoder reranker
        print("Creating Cross Encoder reranker...")
        model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
        cross_encoder = CrossEncoderReranker(model=model, top_n=3)

        # Create ensemble retriever
        print("Creating ensemble retriever...")
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_db],
            reranker=cross_encoder,
        )

        print("Retrievers initialized successfully!")
        return ensemble_retriever

    except Exception as e:
        print(f"Error initializing retrievers: {str(e)}")
        print("Falling back to BM25 retriever only...")
        bm25_retriever = BM25Retriever.from_documents(chunks)
        return bm25_retriever
