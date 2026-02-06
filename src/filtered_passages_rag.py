#!/usr/bin/env python3
"""
Filtered Passages RAG System
============================

RAG system for retrieving passages from filtered_guten/filtered_passages/corpus.jsonl.
Uses semantic_threads as embedding keys (not passage text) for retrieval.

Each passage has 1-3 semantic_threads - each thread is embedded separately
and mapped back to the passage for retrieval.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import faiss
    import numpy as np
except ImportError:
    raise ImportError("pip install faiss-cpu numpy")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("pip install sentence-transformers")


class FilteredPassagesRAG:
    """
    RAG system for filtered passages corpus.
    
    Uses semantic_threads as embedding keys for retrieval.
    Each passage may have multiple semantic_threads, each embedded separately.
    """
    
    def __init__(
        self,
        corpus_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize RAG system.
        
        Args:
            corpus_path: Path to corpus.jsonl file
            embedding_model: Sentence transformer model name
        """
        self.corpus_path = Path(corpus_path)
        self.embedding_model = embedding_model
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        
        # Load passages and build index
        self.passages = []
        self.index = None
        self.thread_to_passage = []  # Maps thread index to passage index
        
        if not self.corpus_path.exists():
            logger.warning(f"Corpus file not found: {corpus_path}")
            logger.warning("RAG system initialized but will return empty results")
            return
        
        self._load_corpus()
        if self.passages:
            self._build_index()
    
    def _load_corpus(self):
        """Load passages from corpus.jsonl."""
        logger.info(f"Loading corpus from {self.corpus_path}")
        
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    passage = json.loads(line)
                    # Validate required fields
                    if 'passage_id' not in passage or 'text' not in passage:
                        logger.warning(f"Skipping invalid passage at line {line_num}: missing required fields")
                        continue
                    
                    # Ensure semantic_threads is a list
                    semantic_threads = passage.get('semantic_threads', [])
                    if not isinstance(semantic_threads, list):
                        semantic_threads = []
                    
                    # Only add passages with at least one semantic_thread
                    if not semantic_threads:
                        logger.debug(f"Skipping passage {passage.get('passage_id')}: no semantic_threads")
                        continue
                    
                    self.passages.append(passage)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(self.passages)} passages from corpus")
    
    def _build_index(self):
        """Build FAISS index from semantic_threads."""
        if not self.passages:
            logger.warning("No passages loaded, cannot build index")
            return
        
        logger.info("Building FAISS index from semantic_threads...")
        
        # Collect all semantic_threads with their passage indices
        all_threads = []
        self.thread_to_passage = []
        
        for passage_idx, passage in enumerate(self.passages):
            semantic_threads = passage.get('semantic_threads', [])
            for thread in semantic_threads:
                if thread and thread.strip():  # Skip empty threads
                    all_threads.append(thread.strip())
                    self.thread_to_passage.append(passage_idx)
        
        if not all_threads:
            logger.warning("No semantic_threads found in passages")
            return
        
        logger.info(f"Embedding {len(all_threads)} semantic_threads...")
        
        # Generate embeddings
        embeddings = self.embedder.encode(all_threads, show_progress_bar=False)
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize embeddings for cosine similarity (using L2 normalization)
        faiss.normalize_L2(embeddings)
        
        # Build FAISS index (using inner product for cosine similarity)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors = cosine similarity
        self.index.add(embeddings)
        
        logger.info(f"FAISS index built: {self.index.ntotal} vectors from {len(self.passages)} passages")
    
    def retrieve(self, query_text: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve passages relevant to query text.
        
        Args:
            query_text: Query text to search for
            top_k: Number of passages to return
        
        Returns:
            List of passage dicts with metadata, sorted by relevance
        """
        logger.info(f"FilteredPassagesRAG.retrieve called with query_text length={len(query_text)}, top_k={top_k}")
        
        if self.index is None or not self.passages:
            logger.warning(f"Index not built or no passages available - index={self.index is not None}, passages={len(self.passages) if self.passages else 0}")
            return []
        
        logger.info(f"Index has {self.index.ntotal} vectors, {len(self.passages)} passages available")
        
        try:
            # Embed query
            logger.debug("Encoding query text...")
            query_embedding = self.embedder.encode([query_text], show_progress_bar=False)
            query_embedding = np.array(query_embedding).astype('float32')
            logger.debug(f"Query embedding shape: {query_embedding.shape}")
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search
            k = min(top_k * 3, self.index.ntotal)  # Get more results to deduplicate
            logger.info(f"Searching index with k={k}")
            similarities, indices = self.index.search(query_embedding, k)
            logger.info(f"Search returned {len(indices[0])} results")
            
            # Map thread indices to passages and deduplicate
            seen_passage_ids = set()
            results = []
            
            for similarity, thread_idx in zip(similarities[0], indices[0]):
                if thread_idx >= len(self.thread_to_passage):
                    logger.warning(f"Thread index {thread_idx} out of range (max: {len(self.thread_to_passage) - 1})")
                    continue
                
                passage_idx = self.thread_to_passage[thread_idx]
                if passage_idx >= len(self.passages):
                    logger.warning(f"Passage index {passage_idx} out of range (max: {len(self.passages) - 1})")
                    continue
                
                passage = self.passages[passage_idx]
                passage_id = passage.get('passage_id')
                
                # Deduplicate by passage_id
                if passage_id in seen_passage_ids:
                    continue
                
                seen_passage_ids.add(passage_id)
                
                # Create result with similarity score
                result = {
                    **passage,
                    'similarity': float(similarity),
                }
                results.append(result)
                
                # Stop when we have enough unique passages
                if len(results) >= top_k:
                    break
            
            logger.info(f"Found {len(results)} unique passages after deduplication")
            
            # Sort by similarity (descending)
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            return results
        
        except Exception as e:
            logger.error(f"Error in FilteredPassagesRAG.retrieve: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Test FilteredPassagesRAG")
    parser.add_argument(
        "--corpus",
        type=str,
        default="./filtered_guten/filtered_passages/corpus.jsonl",
        help="Path to corpus.jsonl file"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What is the nature of self?",
        help="Query text"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of passages to retrieve"
    )
    
    args = parser.parse_args()
    
    rag = FilteredPassagesRAG(corpus_path=args.corpus)
    
    results = rag.retrieve(args.query, top_k=args.top_k)
    
    print(f"\nQuery: {args.query}")
    print(f"\nRetrieved {len(results)} passages:\n")
    
    for i, passage in enumerate(results, 1):
        print(f"{i}. [{passage.get('author', 'Unknown')}] {passage.get('title', 'Untitled')}")
        print(f"   Tradition: {passage.get('tradition', 'unknown')}")
        print(f"   Similarity: {passage['similarity']:.4f}")
        print(f"   Text: {passage.get('text', '')[:200]}...")
        print()
