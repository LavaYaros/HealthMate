"""
ChromaDB configuration and management for HealthMate knowledge base.
Handles vector storage, embeddings, and retrieval for first-aid content.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import List, Dict, Optional

from src.config.settings import get_settings
from src.logging.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()


class ChromaDBManager:
    """
    Manages ChromaDB collection for first-aid knowledge base.
    
    Features:
    - Configurable embedding model (OpenAI by default)
    - Persistent storage with stable chunk identifiers
    - Retrieval with configurable top-k and similarity threshold
    """
    
    def __init__(
        self,
        collection_name: str = "healthmate_firstaid",
        persist_directory: str = "./chroma_db"
    ):
        """
        Initialize ChromaDB manager.
        
        Args:
            collection_name: Name of the Chroma collection
            persist_directory: Directory to persist the vector database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize embedding function
        self.embedding_function = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.openai_api_key
        )
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize LangChain Chroma vectorstore
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embedding_function,
        )
        
        logger.info(f"ChromaDB initialized: collection='{collection_name}', dir='{persist_directory}'")
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            texts: List of text chunks to add
            metadatas: Optional metadata for each chunk (source, page, etc.)
            ids: Optional stable identifiers for chunks
            
        Returns:
            List of document IDs
        """
        try:
            doc_ids = self.vectorstore.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(doc_ids)} documents to ChromaDB")
            return doc_ids
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return (defaults to settings.TOP_K)
            filter: Optional metadata filter
            
        Returns:
            List of documents with content and metadata
        """
        k = k or settings.TOP_K
        
        try:
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter
            )
            
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in results
            ]
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise
    
    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict] = None
    ) -> List[tuple]:
        """
        Search with similarity scores.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of (document, score) tuples
        """
        k = k or settings.TOP_K
        
        try:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
            
            return [
                (
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                    },
                    score
                )
                for doc, score in results
            ]
        except Exception as e:
            logger.error(f"Similarity search with score failed: {e}")
            raise
    
    def get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        try:
            collection = self.client.get_collection(self.collection_name)
            return collection.count()
        except Exception as e:
            logger.error(f"Failed to get collection count: {e}")
            return 0
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise
    
    def reset_collection(self):
        """Reset collection by deleting and recreating it."""
        try:
            self.delete_collection()
            self.vectorstore = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
            )
            logger.info(f"Reset collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            raise


# Singleton instance
_db_manager: Optional[ChromaDBManager] = None


def get_db_manager() -> ChromaDBManager:
    """Get or create singleton ChromaDB manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = ChromaDBManager()
    return _db_manager


if __name__ == "__main__":
    # Test the ChromaDB setup
    print("Testing ChromaDB initialization...")
    db = get_db_manager()
    print(f"✓ ChromaDB initialized")
    print(f"Collection: {db.collection_name}")
    print(f"Document count: {db.get_collection_count()}")
