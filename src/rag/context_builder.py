"""
Context Assembly Module for HealthMate RAG Pipeline.
Formats retrieved passages into structured context for LLM prompts with citations.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import List, Dict, Optional
import tiktoken

from src.config.settings import get_settings
from src.logging.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()


class ContextBuilder:
    """
    Assembles retrieved passages into formatted context strings for LLM prompts.
    
    Features:
    - Token-aware context assembly
    - Citation formatting and tracking
    - Metadata extraction and display
    - Context truncation when exceeding limits
    """
    
    def __init__(
        self,
        max_context_tokens: Optional[int] = None,
        encoding_name: str = "cl100k_base"  # GPT-4, GPT-3.5-turbo
    ):
        """
        Initialize context builder.
        
        Args:
            max_context_tokens: Maximum tokens for context (default from settings)
            encoding_name: Tokenizer encoding to use
        """
        self.max_context_tokens = max_context_tokens or settings.MAX_CONTEXT_TOKENS
        self.encoding = tiktoken.get_encoding(encoding_name)
        
        logger.info(
            f"ContextBuilder initialized: max_tokens={self.max_context_tokens}, "
            f"encoding={encoding_name}"
        )
    
    def build_context(
        self,
        passages: List[Dict],
        include_metadata: bool = True,
        separator: str = "\n\n---\n\n"
    ) -> Dict[str, str]:
        """
        Build formatted context from retrieved passages.
        
        Args:
            passages: List of passage dicts with 'content' and 'metadata'
            include_metadata: Whether to include source metadata in context
            separator: String to separate passages
            
        Returns:
            Dict with 'context' and 'citations' strings
        """
        if not passages:
            logger.warning("No passages provided for context building")
            return {
                "context": "No relevant information found in the knowledge base.",
                "citations": ""
            }
        
        # Build context parts with citations
        context_parts = []
        citation_map = {}  # Track source -> citation number
        citation_counter = 1
        
        current_tokens = 0
        included_passages = 0
        
        for passage in passages:
            content = passage["content"]
            metadata = passage.get("metadata", {})
            source = metadata.get("source", "Unknown")
            
            # Assign citation number
            if source not in citation_map:
                citation_map[source] = citation_counter
                citation_counter += 1
            
            citation_num = citation_map[source]
            
            # Format passage with citation
            if include_metadata:
                formatted_passage = f"[Source {citation_num}] {content}"
            else:
                formatted_passage = content
            
            # Check token count
            passage_tokens = self._count_tokens(formatted_passage + separator)
            
            if current_tokens + passage_tokens > self.max_context_tokens:
                logger.warning(
                    f"Context truncated: {included_passages}/{len(passages)} passages fit "
                    f"within {self.max_context_tokens} tokens"
                )
                break
            
            context_parts.append(formatted_passage)
            current_tokens += passage_tokens
            included_passages += 1
        
        # Assemble final context
        context = separator.join(context_parts)
        
        # Build citations string
        citations = self._build_citations_from_map(citation_map, passages)
        
        logger.info(
            f"Built context: {included_passages} passages, {current_tokens} tokens, "
            f"{len(citation_map)} unique sources"
        )
        
        return {
            "context": context,
            "citations": citations,
            "num_passages": included_passages,
            "num_sources": len(citation_map),
            "total_tokens": current_tokens
        }
    
    def _build_citations_from_map(
        self,
        citation_map: Dict[str, int],
        passages: List[Dict]
    ) -> str:
        """
        Build formatted citations list from citation map.
        
        Args:
            citation_map: Dict mapping source names to citation numbers
            passages: Original passages list for metadata
            
        Returns:
            Formatted citations string
        """
        # Gather unique source metadata
        source_metadata = {}
        for passage in passages:
            metadata = passage.get("metadata", {})
            source = metadata.get("source", "Unknown")
            
            if source not in source_metadata:
                source_metadata[source] = metadata
        
        # Build formatted citations
        citations_list = []
        for source, num in sorted(citation_map.items(), key=lambda x: x[1]):
            metadata = source_metadata.get(source, {})
            
            citation = f"{num}. {source}"
            
            # Add page count if available
            pages = metadata.get("pages")
            if pages:
                citation += f" ({pages} pages)"
            
            citations_list.append(citation)
        
        return "\n".join(citations_list)
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the configured encoding.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))
    
    def format_for_prompt(
        self,
        context_data: Dict[str, str],
        query: str
    ) -> Dict[str, str]:
        """
        Format context and query for insertion into prompt templates.
        
        Args:
            context_data: Dict from build_context() with 'context' and 'citations'
            query: User's original query
            
        Returns:
            Dict with 'query', 'context', and 'citations' ready for prompt template
        """
        return {
            "query": query,
            "context": context_data["context"],
            "citations": context_data["citations"]
        }
    
    def get_context_summary(self, context_data: Dict) -> str:
        """
        Get a summary of the built context.
        
        Args:
            context_data: Dict from build_context()
            
        Returns:
            Human-readable summary string
        """
        return (
            f"Context: {context_data.get('num_passages', 0)} passages, "
            f"{context_data.get('num_sources', 0)} sources, "
            f"{context_data.get('total_tokens', 0)} tokens"
        )


def get_context_builder() -> ContextBuilder:
    """Get a configured context builder instance."""
    return ContextBuilder()
