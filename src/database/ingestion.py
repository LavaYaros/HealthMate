"""
Document ingestion module for HealthMate.
Loads PDF documents from med_sources folder, chunks them, and stores in ChromaDB.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import List, Dict, Optional
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.database.chroma_db import get_db_manager
from src.logging.logger import setup_logger

logger = setup_logger(__name__)


class DocumentIngestion:
    """
    Handles PDF document ingestion and processing for ChromaDB.
    
    Features:
    - Load PDFs from specified directory
    - Extract and clean text content
    - Chunk documents with overlap for context preservation
    - Generate stable IDs for tracking
    - Store in ChromaDB with metadata
    """
    
    def __init__(
        self,
        source_dir: str = "med_sources",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize document ingestion.
        
        Args:
            source_dir: Directory containing PDF files
            chunk_size: Size of text chunks (characters)
            chunk_overlap: Overlap between chunks for context
        """
        self.source_dir = Path(ROOT) / source_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Get ChromaDB manager
        self.db_manager = get_db_manager()
        
        logger.info(f"Initialized ingestion from: {self.source_dir}")
    
    def load_pdf(self, pdf_path: Path) -> Dict:
        """
        Load and extract text from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dict with 'text' and 'metadata'
        """
        try:
            reader = PdfReader(str(pdf_path))
            
            # Extract text from all pages
            text_parts = []
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    text_parts.append(text)
            
            full_text = "\n\n".join(text_parts)
            
            metadata = {
                "source": pdf_path.name,
                "pages": len(reader.pages),
                "path": str(pdf_path.relative_to(ROOT))
            }
            
            logger.info(f"Loaded {pdf_path.name}: {len(reader.pages)} pages, {len(full_text)} chars")
            
            return {
                "text": full_text,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to load {pdf_path.name}: {e}")
            raise
    
    def chunk_document(self, document: Dict) -> List[Dict]:
        """
        Split document into chunks with metadata.
        
        Args:
            document: Dict with 'text' and 'metadata'
            
        Returns:
            List of chunk dicts with text and metadata
        """
        text = document["text"]
        base_metadata = document["metadata"]
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create chunk documents with metadata
        chunk_docs = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **base_metadata,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            
            chunk_docs.append({
                "text": chunk,
                "metadata": chunk_metadata,
                "id": f"{base_metadata['source']}_chunk_{i}"
            })
        
        logger.info(f"Split {base_metadata['source']} into {len(chunks)} chunks")
        return chunk_docs
    
    def ingest_pdfs(self, file_pattern: str = "*.pdf") -> int:
        """
        Ingest all PDFs matching pattern from source directory.
        
        Args:
            file_pattern: Glob pattern for PDF files
            
        Returns:
            Number of chunks added to database
        """
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")
        
        pdf_files = list(self.source_dir.glob(file_pattern))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.source_dir}")
            return 0
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        all_chunks = []
        
        # Process each PDF
        for pdf_path in pdf_files:
            try:
                # Load PDF
                document = self.load_pdf(pdf_path)
                
                # Chunk document
                chunks = self.chunk_document(document)
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"Skipping {pdf_path.name} due to error: {e}")
                continue
        
        if not all_chunks:
            logger.warning("No chunks to add to database")
            return 0
        
        # Add to ChromaDB
        texts = [chunk["text"] for chunk in all_chunks]
        metadatas = [chunk["metadata"] for chunk in all_chunks]
        ids = [chunk["id"] for chunk in all_chunks]
        
        self.db_manager.add_documents(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"✓ Ingested {len(all_chunks)} chunks from {len(pdf_files)} PDFs")
        return len(all_chunks)
    
    def get_ingestion_stats(self) -> Dict:
        """
        Get statistics about ingested documents.
        
        Returns:
            Dict with collection stats
        """
        count = self.db_manager.get_collection_count()
        
        return {
            "total_chunks": count,
            "source_directory": str(self.source_dir),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }


def ingest_all_documents():
    """Helper function to ingest all documents from med_sources."""
    ingestion = DocumentIngestion()
    chunks_added = ingestion.ingest_pdfs()
    stats = ingestion.get_ingestion_stats()
    
    print(f"\n✓ Ingestion complete!")
    print(f"  Total chunks in database: {stats['total_chunks']}")
    print(f"  Source directory: {stats['source_directory']}")
    print(f"  Chunk size: {stats['chunk_size']} chars")
    print(f"  Chunk overlap: {stats['chunk_overlap']} chars")
    
    return stats
