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
import os
import re
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import httpx
except ImportError:
    raise ImportError("pip install httpx")

try:
    import faiss
    import numpy as np
except ImportError:
    raise ImportError("pip install faiss-cpu numpy")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("pip install sentence-transformers")


# Prompt that mirrors the semantic_thread creation task in passage_identification.py
# but adapted for rewriting a dialogue (question + response) into embedding keys.
SEMANTIC_THREAD_REWRITE_PROMPT = """You are generating embedding keys for a retrieval system that finds contemplative passages from world traditions that resonate with a dialogue about Ramana Maharshi's teachings.

The retrieval corpus is indexed by "semantic threads": 1-3 brief descriptions (under 30 words each) of the contemplative or experiential meaning of each passage. These threads are expressed in clear, tradition-neutral language that captures the inner or spiritual territory the passage points to -- focusing on what it says about awareness, self, mind, realization, or lived contemplative experience.

Your task: given the dialogue below, produce 1-3 semantic thread descriptions that capture the contemplative territory being explored. These will be embedded and compared against the corpus threads, so they must be written in the same register.

GOOD thread: "Non-reliance on any fixed principle opens perception; clinging to method blinds, while letting go reveals clear seeing."
BAD thread: "The user asks about meditation and the response discusses self-inquiry."

Focus on the experiential and contemplative substance -- what territory of awareness, identity, mind, or realization is being pointed at? NOT the surface topic, rhetorical structure, or conversation mechanics.

{user_context_section}DIALOGUE:
Question: {user_input}

Response: {response}

{expanded_section}
Respond with ONLY a JSON array of 1-3 strings. No markdown, no explanation.
"""


class FilteredPassagesRAG:
    """
    RAG system for filtered passages corpus.
    
    Uses semantic_threads as embedding keys for retrieval.
    Each passage may have multiple semantic_threads, each embedded separately.
    
    Query rewriting: raw user dialogue is rewritten into semantic threads
    via an LLM call, so the query embedding lives in the same semantic
    space as the corpus embedding keys.
    """
    
    def __init__(
        self,
        corpus_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_url: str = "http://localhost:5000/v1",
        llm_model: Optional[str] = None,
    ):
        """
        Initialize RAG system.
        
        Args:
            corpus_path: Path to corpus.jsonl file
            embedding_model: Sentence transformer model name
            llm_url: URL for the LLM API (used for query rewriting)
            llm_model: Model name for LLM (auto-detected if None for local, required for OpenRouter)
        """
        self.corpus_path = Path(corpus_path)
        self.embedding_model = embedding_model
        self.llm_url = llm_url
        self.llm_model = llm_model
        
        # Detect backend from URL
        self.llm_backend = "openrouter" if "openrouter.ai" in llm_url else "local"
        
        # HTTP client for LLM calls
        self.http_client = httpx.Client(timeout=60.0)
        
        # Auto-detect LLM model if not provided (local only)
        if self.llm_model is None:
            if self.llm_backend == "openrouter":
                # Default for OpenRouter
                self.llm_model = "anthropic/claude-sonnet-4"
                logger.info(f"Using default OpenRouter model: {self.llm_model}")
            else:
                self.llm_model = self._detect_llm_model()
                logger.info(f"Auto-detected LLM model: {self.llm_model}")
        
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
    
    def _detect_llm_model(self) -> str:
        """Auto-detect model from local vLLM API."""
        try:
            resp = self.http_client.get(f"{self.llm_url}/models")
            resp.raise_for_status()
            data = resp.json()
            if "data" in data and len(data["data"]) > 0:
                return data["data"][0]["id"]
            else:
                logger.warning("Could not detect LLM model, using 'unknown'")
                return "unknown"
        except Exception as e:
            logger.warning(f"Could not detect LLM model: {e}")
            return "unknown"
    
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
    
    def rewrite_query_as_threads(
        self,
        user_input: str,
        response: str,
        expanded_response: str = "",
        user_context: str = "",
    ) -> list[str]:
        """
        Rewrite a user dialogue into semantic threads via LLM call.
        
        Produces 1-3 thread descriptions in the same register as the
        corpus semantic_threads, so both query and corpus embeddings
        occupy the same semantic space.
        
        Args:
            user_input: The user's question
            response: The generated response
            expanded_response: The expanded response (may be empty)
            user_context: User model threads + recent turns (may be empty)
        
        Returns:
            List of 1-3 semantic thread strings
        """
        expanded_section = ""
        if expanded_response:
            expanded_section = f"Expanded Response: {expanded_response}"
        
        user_context_section = ""
        if user_context:
            user_context_section = f"VISITOR CONTEXT:\n{user_context}\n\n"
        
        prompt = SEMANTIC_THREAD_REWRITE_PROMPT.format(
            user_input=user_input,
            response=response,
            expanded_section=expanded_section,
            user_context_section=user_context_section,
        )
        
        try:
            # Build request payload
            payload = {
                "model": self.llm_model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 512,
                "temperature": 0.3,
            }
            
            # Build headers (OpenRouter needs auth)
            headers = {}
            if self.llm_backend == "openrouter":
                api_key = os.environ.get("OPENROUTER_API_KEY")
                if not api_key:
                    raise ValueError("OPENROUTER_API_KEY environment variable required")
                headers["Authorization"] = f"Bearer {api_key}"
                headers["HTTP-Referer"] = "https://github.com/ramana-website"
                headers["X-Title"] = "Ramana Website"
            
            resp = self.http_client.post(
                f"{self.llm_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            
            logger.info(f"LLM query rewrite raw output: {raw[:300]}")
            
            # Parse JSON array from response -- tolerate markdown fences
            cleaned = raw
            if cleaned.startswith("```"):
                cleaned = re.sub(r'^```\w*\n?', '', cleaned)
                cleaned = re.sub(r'\n?```$', '', cleaned)
                cleaned = cleaned.strip()
            
            # LLM sometimes returns multiple JSON arrays on separate lines
            # e.g. ["thread1"]\n["thread2"]\n["thread3"]
            # Try direct parse first; on failure, merge separate arrays
            try:
                threads = json.loads(cleaned)
            except json.JSONDecodeError:
                merged = []
                for line in cleaned.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                        if isinstance(parsed, list):
                            merged.extend(parsed)
                        elif isinstance(parsed, str):
                            merged.append(parsed)
                    except json.JSONDecodeError:
                        # Bare string line — use as-is
                        if line.strip('"').strip("'"):
                            merged.append(line.strip('"').strip("'"))
                if merged:
                    logger.info(f"Merged {len(merged)} threads from multi-line LLM output")
                    threads = merged
                else:
                    raise
            
            if not isinstance(threads, list):
                logger.warning(f"LLM returned non-list: {type(threads)}")
                return [raw]  # Fallback: use raw text as single thread
            
            # Filter to non-empty strings, cap at 3
            threads = [t.strip() for t in threads if isinstance(t, str) and t.strip()][:3]
            
            if not threads:
                logger.warning("LLM returned empty threads list, falling back to raw text")
                return [f"{user_input}. {response}"]
            
            logger.info(f"Rewritten query into {len(threads)} semantic threads")
            for i, t in enumerate(threads):
                logger.info(f"  Thread {i+1}: {t}")
            
            return threads
        
        except Exception as e:
            logger.error(f"LLM query rewrite failed: {e}", exc_info=True)
            # Fallback: return raw concatenation so retrieval still works
            logger.info("Falling back to raw text concatenation for retrieval")
            parts = [p for p in [user_input, response, expanded_response] if p]
            return ['. '.join(parts)]
    
    def retrieve_with_rewrite(
        self,
        user_input: str,
        response: str,
        expanded_response: str = "",
        user_context: str = "",
        top_k: int = 5,
    ) -> list[dict]:
        """
        Rewrite the dialogue into semantic threads, then retrieve.
        
        This is the primary retrieval interface. It rewrites the raw
        dialogue into semantic threads via an LLM call, then searches
        the corpus using those threads as queries.
        
        Args:
            user_input: The user's question
            response: The generated response
            expanded_response: The expanded response (may be empty)
            user_context: User model threads + recent turns (may be empty)
            top_k: Number of passages to return
        
        Returns:
            List of passage dicts with metadata, sorted by relevance
        """
        threads = self.rewrite_query_as_threads(
            user_input, response, expanded_response, user_context,
        )
        return self.retrieve(threads, top_k=top_k)
    
    def retrieve(self, query_texts: str | list[str], top_k: int = 5) -> list[dict]:
        """
        Retrieve passages relevant to one or more query texts.
        
        When multiple query texts are provided (e.g. multiple semantic
        threads), each is searched independently and results are merged
        by best similarity per passage.
        
        Args:
            query_texts: Single query string or list of query strings
            top_k: Number of passages to return
        
        Returns:
            List of passage dicts with metadata, sorted by relevance
        """
        if isinstance(query_texts, str):
            query_texts = [query_texts]
        
        logger.info(f"FilteredPassagesRAG.retrieve called with {len(query_texts)} query texts, top_k={top_k}")
        
        if self.index is None or not self.passages:
            logger.warning(f"Index not built or no passages available - index={self.index is not None}, passages={len(self.passages) if self.passages else 0}")
            return []
        
        logger.info(f"Index has {self.index.ntotal} vectors, {len(self.passages)} passages available")
        
        try:
            # Merge results across all query texts, keeping best similarity per passage
            best_by_passage = {}  # passage_id -> (similarity, passage)
            
            for q_idx, query_text in enumerate(query_texts):
                logger.info(f"Searching with query {q_idx+1}/{len(query_texts)}: {query_text[:120]}...")
                
                query_embedding = self.embedder.encode([query_text], show_progress_bar=False)
                query_embedding = np.array(query_embedding).astype('float32')
                faiss.normalize_L2(query_embedding)
                
                k = min(top_k * 3, self.index.ntotal)
                similarities, indices = self.index.search(query_embedding, k)
                
                for similarity, thread_idx in zip(similarities[0], indices[0]):
                    if thread_idx >= len(self.thread_to_passage):
                        continue
                    
                    passage_idx = self.thread_to_passage[thread_idx]
                    if passage_idx >= len(self.passages):
                        continue
                    
                    passage = self.passages[passage_idx]
                    passage_id = passage.get('passage_id')
                    sim = float(similarity)
                    
                    # Keep best similarity for each passage
                    if passage_id not in best_by_passage or sim > best_by_passage[passage_id][0]:
                        best_by_passage[passage_id] = (sim, passage)
            
            # Sort by similarity and take top_k
            results = []
            for passage_id, (sim, passage) in best_by_passage.items():
                results.append({
                    **passage,
                    'similarity': sim,
                })
            
            results.sort(key=lambda x: x['similarity'], reverse=True)
            results = results[:top_k]
            
            logger.info(f"Found {len(results)} passages after merging across {len(query_texts)} queries")
            
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
