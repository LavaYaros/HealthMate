"""
RAG Retrieval Module for HealthMate.
Handles semantic search, deduplication, and filtering of retrieved passages from ChromaDB.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import List, Dict, Optional, Set
from difflib import SequenceMatcher

from src.database.chroma_db import get_db_manager
from src.config.settings import get_settings
from src.logging.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()


class RAGRetriever:
    """
    Handles retrieval and filtering of relevant passages for RAG pipeline.
    
    Features:
    - Semantic similarity search via ChromaDB
    - Deduplication of near-identical passages
    - Relevance filtering based on similarity scores
    - Citation metadata extraction
    """
    
    def __init__(
        self,
        similarity_threshold: Optional[float] = None,
        dedup_threshold: Optional[float] = None,
        top_k: Optional[int] = None
    ):
        """
        Initialize RAG retriever.
        
        Args:
            similarity_threshold: Minimum similarity score (0-1). Passages below this are filtered out
            dedup_threshold: Text similarity threshold for deduplication (0-1, default 0.85)
            top_k: Number of passages to retrieve (default from settings)
        """
        self.db_manager = get_db_manager()
        self.similarity_threshold = similarity_threshold or settings.SIMILARITY_THRESHOLD
        self.dedup_threshold = dedup_threshold or settings.DEDUP_THRESHOLD
        self.top_k = top_k or settings.TOP_K
        
        logger.info(
            f"RAGRetriever initialized: top_k={self.top_k}, "
            f"similarity_threshold={self.similarity_threshold}, "
            f"dedup_threshold={self.dedup_threshold}"
        )
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        apply_dedup: bool = True
    ) -> List[Dict]:
        """
        Retrieve relevant passages for a query.
        
        Args:
            query: User's medical query
            top_k: Number of passages to retrieve (overrides default)
            apply_dedup: Whether to deduplicate similar passages
            
        Returns:
            List of passage dicts with 'content', 'metadata', and 'score'
        """
        k = top_k or self.top_k
        
        # Retrieve passages with scores
        results = self.db_manager.similarity_search_with_score(
            query=query,
            k=k * 2  # Get extra results for filtering
        )
        
        if not results:
            logger.warning(f"No results found for query: {query[:100]}")
            return []
        
        # Convert to standard format with scores
        passages = []
        for doc, score in results:
            # ChromaDB returns distance (lower is better), convert to similarity
            # Distance is L2, so we invert: similarity = 1 / (1 + distance)
            similarity_score = 1 / (1 + score)
            
            passages.append({
                "content": doc["content"],
                "metadata": doc["metadata"],
                "score": similarity_score
            })
        
        logger.info(f"Retrieved {len(passages)} passages for query")
        
        # Filter by similarity threshold
        filtered_passages = self._filter_by_relevance(passages)
        
        # Deduplicate if requested
        if apply_dedup:
            filtered_passages = self._deduplicate_passages(filtered_passages)
        
        # Return top-k after filtering
        final_passages = filtered_passages[:k]
        
        logger.info(
            f"After filtering: {len(final_passages)} passages "
            f"(filtered {len(passages) - len(filtered_passages)}, kept top {k})"
        )
        
        return final_passages
    
    def _filter_by_relevance(self, passages: List[Dict]) -> List[Dict]:
        """
        Filter passages based on similarity threshold.
        
        Args:
            passages: List of passage dicts with scores
            
        Returns:
            Filtered list of passages
        """
        filtered = [
            p for p in passages
            if p["score"] >= self.similarity_threshold
        ]
        
        if len(filtered) < len(passages):
            logger.debug(
                f"Filtered {len(passages) - len(filtered)} passages "
                f"below threshold {self.similarity_threshold}"
            )
        
        return filtered
    
    def _deduplicate_passages(self, passages: List[Dict]) -> List[Dict]:
        """
        Remove near-duplicate passages based on text similarity.
        
        Uses sequence matching to find passages with high textual overlap.
        Keeps the passage with the highest score among duplicates.
        
        Args:
            passages: List of passage dicts with content and scores
            
        Returns:
            Deduplicated list of passages
        """
        if not passages:
            return passages
        
        # Sort by score descending (keep highest scoring duplicates)
        sorted_passages = sorted(passages, key=lambda x: x["score"], reverse=True)
        
        unique_passages = []
        seen_content: List[str] = []
        
        for passage in sorted_passages:
            content = passage["content"]
            
            # Check if this content is similar to any we've already kept
            is_duplicate = False
            for seen in seen_content:
                similarity = self._text_similarity(content, seen)
                if similarity >= self.dedup_threshold:
                    is_duplicate = True
                    logger.debug(
                        f"Deduplicating passage (similarity: {similarity:.2f}): "
                        f"{content[:100]}..."
                    )
                    break
            
            if not is_duplicate:
                unique_passages.append(passage)
                seen_content.append(content)
        
        if len(unique_passages) < len(passages):
            logger.info(f"Deduplicated: {len(passages)} → {len(unique_passages)} passages")
        
        return unique_passages
    
    @staticmethod
    def _text_similarity(text1: str, text2: str) -> float:
        """
        Calculate text similarity using sequence matching.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        return SequenceMatcher(None, text1, text2).ratio()
    
    def extract_citations(self, passages: List[Dict]) -> List[Dict]:
        """
        Extract citation information from passages.
        
        Args:
            passages: List of passage dicts with metadata
            
        Returns:
            List of unique citation dicts with source info
        """
        citations = []
        seen_sources: Set[str] = set()
        
        for passage in passages:
            metadata = passage.get("metadata", {})
            source = metadata.get("source", "Unknown")
            
            if source not in seen_sources:
                citation = {
                    "source": source,
                    "path": metadata.get("path", ""),
                    "pages": metadata.get("pages", "N/A")
                }
                citations.append(citation)
                seen_sources.add(source)
        
        return citations
    
    def format_citations(self, citations: List[Dict]) -> str:
        """
        Format citations for display in responses.
        
        Args:
            citations: List of citation dicts
            
        Returns:
            Formatted citation string
        """
        if not citations:
            return ""
        
        formatted = "\n\n**Sources:**\n"
        for i, citation in enumerate(citations, 1):
            formatted += f"{i}. {citation['source']}"
            if citation.get("pages"):
                formatted += f" ({citation['pages']} pages)"
            formatted += "\n"
        
        return formatted


def get_retriever() -> RAGRetriever:
    """Get a configured RAG retriever instance."""
    return RAGRetriever()
