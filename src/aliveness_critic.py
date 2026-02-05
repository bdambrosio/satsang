#!/usr/bin/env python3
"""
Aliveness Critic for Contemplative Dialogue
============================================

A runtime scaffolding system that uses a large general-purpose LLM to evaluate
candidate responses from a smaller specialized generator model. The critic
identifies and filters "stale" responses that exhibit closure-seeking,
helpfulness reflexes, or prescriptive patterns.

Architecture:
    [User Input] → [Small Generator] → [Candidate Response]
                                              ↓
                                    [Large Critic Model]
                                              ↓
                        [Score < threshold?] → Regenerate / Return silence
                                              ↓
                                    [Return Response]

Usage:
    # Using HTTP API (default)
    from aliveness_critic import LocalCritic, ContemplativeGenerator
    
    critic = LocalCritic(backend="openrouter", model="anthropic/claude-sonnet-4")
    generator = ContemplativeGenerator(
        generator_url="http://localhost:5000/v1",
        critic=critic,
    )
    
    # Using local inference provider
    from phi2_contemplative_inference_lora import Phi2InferenceProvider
    
    provider = Phi2InferenceProvider(
        base_model_name="microsoft/phi-2",
        checkpoint_dir=Path("./phi2-contemplative-lora/checkpoint-4000")
    )
    generator = ContemplativeGenerator(
        inference_provider=provider,
        critic=critic,
    )
    
    response = generator.generate("I feel lost in my practice.")

Requirements:
    pip install httpx torch transformers peft
"""

import argparse
import json
import logging
import os
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import numpy as np
except ImportError:
    np = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress httpx INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)

# Optional import for local inference providers
try:
    import importlib.util
    
    # Try Phi2InferenceProvider
    inference_module_path = Path(__file__).parent / "phi2-contemplative-inference-lora.py"
    if inference_module_path.exists():
        spec = importlib.util.spec_from_file_location("phi2_inference", inference_module_path)
        phi2_inference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(phi2_inference)
        Phi2InferenceProvider = phi2_inference.Phi2InferenceProvider
    else:
        Phi2InferenceProvider = None
except Exception as e:
    Phi2InferenceProvider = None
    logger.debug(f"Phi2InferenceProvider not available: {e}")

# Try ContemplativeRAGProvider
try:
    import contemplative_rag
    ContemplativeRAGProvider = contemplative_rag.ContemplativeRAGProvider
    NAN_YAR = contemplative_rag.NAN_YAR
except ImportError as e:
    ContemplativeRAGProvider = None
    logger.debug(f"ContemplativeRAGProvider not available (ImportError): {e}")
except Exception as e:
    ContemplativeRAGProvider = None
    logger.debug(f"ContemplativeRAGProvider not available: {e}")

# =============================================================================
# CRITIC PROMPTS
# =============================================================================

ALIVENESS_CRITIC_SYSTEM = """You are evaluating responses in a contemplative dialogue context.

Your task is to assess whether a response is ALIVE or STALE.

ALIVE responses:
- Leave space for the questioner's own discovery
- Turn energy back as genuine inquiry (not rhetorical deflection)  
- Refuse to close what wants to stay open
- Are comfortable with not-knowing
- Respond to what's beneath the surface question
- Use minimal words—no more than the moment requires
- Point rather than explain

STALE responses:
- Offer resolution without being asked
- Use prescriptive language ("you should", "try this", "consider...")
- Mirror the questioner's framing rather than refracting it
- Provide excessive agreement or validation
- Inflate with unnecessary words
- Explain concepts rather than pointing at them
- Perform wisdom rather than embodying it
- Sound like generic spiritual advice

You are detecting the CLOSURE REFLEX—the habitual move toward resolution,
helpfulness, and answer-giving that kills contemplative space.

First identify the specific qualities (ALIVE or STALE) that are present in the response.
Then score the response based on the specific qualities present, using this scale:

- 9-10: Genuinely alive, creates space, masterful
- 7-8: Good, mostly alive with minor staleness
- 5-6: Mixed, some alive qualities but closure reflex present
- 3-4: Mostly stale, performing rather than being
- 1-2: Generic spiritual chatbot output
- 0: Actively harmful to contemplative space

Respond with a JSON object:
{
    "score": <float 0-10>,
    "reasoning": "<terse (8-10 word) explanation>",
    "suggestion": "<optional: terse (6-12 word) statement of how it might be improved, or 'none'>"
}

"""

ALIVENESS_CRITIC_USER = """Evaluate this response for aliveness.

CONTEXT (what the person said):
{user_input}

RESPONSE TO EVALUATE:
{response}

Respond with JSON only."""


# =============================================================================
# STALENESS MARKERS (for fast pre-filtering)
# =============================================================================

STALE_PHRASE_PATTERNS = [
    # Prescriptive patterns
    r'\byou should\b',
    r'\byou could try\b',
    r'\btry to\b',
    r'\bconsider\b(?!ing)',  # "consider" as command, not "considering"
    r'\bi recommend\b',
    r'\bi suggest\b',
    r'\bhere are some\b',
    r'\bhere\'s what\b',
    r'\bsteps to\b',
    r'\btips for\b',
    r'\bways to\b',
    
    # Validation/agreement excess
    r'\bthat\'s (a )?(great|wonderful|excellent|good) (question|point|observation)\b',
    r'\bi (completely |totally )?understand\b',
    r'\bi hear you\b',
    r'\bthat makes sense\b',
    r'\byou\'re (absolutely |totally )?right\b',
    
    # Explanation/teaching mode
    r'\blet me explain\b',
    r'\bto clarify\b',
    r'\bin other words\b',
    r'\bwhat this means is\b',
    r'\bthe key (thing |point |idea )?is\b',
    r'\bessentially\b',
    r'\bfundamentally\b',
    
    # Generic spiritual phrases
    r'\bon your journey\b',
    r'\byour path\b',
    r'\binner wisdom\b',
    r'\bhigher self\b',
    r'\bthe universe\b',
    r'\bmanifest\b',
    r'\benergy\b(?! of)',  # standalone "energy" 
    r'\bvibration\b',
    r'\balignment\b',
]

STALE_STRUCTURAL_PATTERNS = [
    # Lists and bullets (almost always stale in this context)
    r'^\s*[-•*]\s',  # bullet points
    r'^\s*\d+\.\s',  # numbered lists
    r'\n\s*[-•*]\s',
    r'\n\s*\d+\.\s',
    
    # Headers/structure (over-formatting)
    r'^#+\s',  # markdown headers
    r'\*\*[^*]+\*\*:',  # bold labels
]


@dataclass
class CriticResult:
    """Result from the aliveness critic."""
    score: float
    alive_signals: list[str] = field(default_factory=list)
    stale_signals: list[str] = field(default_factory=list)
    reasoning: str = ""
    suggestion: str = ""
    raw_response: str = ""
    
    @property
    def is_alive(self) -> bool:
        return self.score >= 6.0
    
    @property
    def needs_silence(self) -> bool:
        """Score so low that silence is better than any response."""
        return self.score < 3.0
    
    def display(self) -> str:
        """Pretty-print the critic result."""
        lines = []
        
        # Score and status
        status = "✓ ALIVE" if self.is_alive else "✗ STALE"
        if self.needs_silence:
            status = "⚠ SILENCE RECOMMENDED"
        lines.append(f"Score: {self.score:.1f}/10.0  [{status}]")
        lines.append("")
        
        # Alive signals
        #if self.alive_signals:
        #    lines.append("Alive Signals:")
        #    for signal in self.alive_signals:
        #        lines.append(f"  ✓ {signal}")
        #    lines.append("")
        
        # Stale signals
        #if self.stale_signals:
        #    lines.append("Stale Signals:")
        #    for signal in self.stale_signals:
        #        lines.append(f"  ✗ {signal}")
        #    lines.append("")
        
        # Reasoning
        #if self.reasoning:
        #    lines.append("Reasoning:")
        #    lines.append(f"  {self.reasoning}")
        #    lines.append("")
        
        # Suggestion
        #if self.suggestion and self.suggestion.lower() != "none":
        #    lines.append("Suggestion:")
        #    lines.append(f"  {self.suggestion}")
        #    lines.append("")
        
        #lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# CRITIC IMPLEMENTATIONS
# =============================================================================

class BaseCritic(ABC):
    """Base class for aliveness critics."""
    
    @abstractmethod
    def evaluate(self, user_input: str, response: str) -> CriticResult:
        """Evaluate a response for aliveness."""
        pass
    
    def quick_stale_check(self, response: str) -> tuple[bool, list[str]]:
        """
        Fast pattern-based check for obvious staleness.
        Returns (is_obviously_stale, matched_patterns).
        """
        matched = []
        response_lower = response.lower()
        
        for pattern in STALE_PHRASE_PATTERNS:
            if re.search(pattern, response_lower):
                matched.append(pattern)
        
        for pattern in STALE_STRUCTURAL_PATTERNS:
            if re.search(pattern, response, re.MULTILINE):
                matched.append(pattern)
        
        # If many stale markers, it's obviously stale
        is_stale = len(matched) >= 3
        return is_stale, matched


class LocalCritic(BaseCritic):
    """
    Uses a local model (via vLLM) or OpenRouter as critic.
    
    Supports both local vLLM (port 5000) and OpenRouter API.
    """
    
    def __init__(
        self,
        backend: str = "local",  # "local" or "openrouter"
        base_url: str = "http://localhost:5000/v1",
        model: Optional[str] = None,
        threshold: float = 6.0,
    ):
        self.backend = backend
        self.base_url = base_url
        self.threshold = threshold
        
        try:
            import httpx
            self.http_client = httpx.Client(timeout=60.0)
        except ImportError:
            raise ImportError("pip install httpx")
        
        # Set model
        if backend == "openrouter":
            if model is None:
                self.model = "anthropic/claude-sonnet-4"  # Default for openrouter
            else:
                self.model = model
            logger.info(f"Using OpenRouter critic model: {self.model}")
        else:  # local
            if model is None:
                self.model = self._detect_model()
                logger.info(f"Auto-detected local critic model: {self.model}")
            else:
                self.model = model
                logger.info(f"Using specified critic model: {self.model}")
    
    def _detect_model(self) -> str:
        """Auto-detect model from local API."""
        try:
            resp = self.http_client.get(f"{self.base_url}/models")
            resp.raise_for_status()
            data = resp.json()
            
            # Handle OpenAI-compatible format
            if "data" in data and len(data["data"]) > 0:
                return data["data"][0]["id"]
            elif isinstance(data, list) and len(data) > 0:
                return data[0] if isinstance(data[0], str) else data[0].get("id", "local-model")
            else:
                logger.warning("Could not detect model, using 'local-model'")
                return "local-model"
        except Exception as e:
            logger.warning(f"Could not detect local model: {e}, using 'local-model'")
            return "local-model"
    
    def evaluate(self, user_input: str, response: str) -> CriticResult:
        """Evaluate using local model API."""
        
        # Quick check
        is_obviously_stale, stale_markers = self.quick_stale_check(response)
        if is_obviously_stale:
            return CriticResult(
                score=2.0,
                stale_signals=stale_markers[:5],
                reasoning="Failed quick staleness check",
            )
        
        prompt = ALIVENESS_CRITIC_USER.format(
            user_input=user_input,
            response=response
        )
        
        try:
            if self.backend == "openrouter":
                return self._call_openrouter(prompt)
            else:
                return self._call_local(prompt)
        except Exception as e:
            logger.warning(f"Critic evaluation failed: {e}")
            return CriticResult(score=5.0, reasoning=f"Error: {e}")
    
    def _call_local(self, prompt: str) -> CriticResult:
        """Call local vLLM API."""
        resp = self.http_client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": ALIVENESS_CRITIC_SYSTEM},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.3,
            }
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        return self._parse_result(raw)
    
    def _call_openrouter(self, prompt: str) -> CriticResult:
        """Call OpenRouter API."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable required")
        
        resp = self.http_client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/your-repo",
                "X-Title": "Contemplative Critic",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": ALIVENESS_CRITIC_SYSTEM},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.3,
            }
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        return self._parse_result(raw)
    
    def _parse_result(self, raw: str) -> CriticResult:
        """Parse JSON response from critic."""
        try:
            json_match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(raw)
            
            return CriticResult(
                score=float(data.get('score', 5.0)),
                alive_signals=data.get('alive_signals', []),
                stale_signals=data.get('stale_signals', []),
                reasoning=data.get('reasoning', ''),
                suggestion=data.get('suggestion', ''),
                raw_response=raw,
            )
        except Exception as e:
            return CriticResult(score=5.0, reasoning=f"Parse error: {e}", raw_response=raw)


# =============================================================================
# CONTEMPLATIVE GENERATOR WITH CRITIC
# =============================================================================

@dataclass
class GenerationConfig:
    """Configuration for the contemplative generator."""
    max_attempts: int = 3
    temperature_base: float = 0.7
    temperature_bump: float = 0.15  # Increase per retry
    threshold: float = 6.0
    allow_silence: bool = True
    silence_threshold: float = 3.0  # Below this, return silence


class ContemplativeGenerator:
    """
    Generator that uses critic feedback to filter stale responses.
    
    Architecture:
        1. Generate candidate response
        2. Evaluate with critic
        3. If below threshold, regenerate with higher temperature
        4. If all attempts fail, optionally return silence
    """
    
    def __init__(
        self,
        generator_url: str = "http://localhost:5000/v1",
        generator_model: Optional[str] = None,
        critic: Optional[BaseCritic] = None,
        config: Optional[GenerationConfig] = None,
        auto_select_model: bool = True,
        inference_provider=None,  # Optional Phi2InferenceProvider instance
    ):
        self.inference_provider = inference_provider
        self.generator_url = generator_url
        self.critic = critic
        self.config = config or GenerationConfig()
        
        if inference_provider is not None:
            # Check if it's a valid inference provider (Phi2InferenceProvider or ContemplativeRAGProvider)
            is_valid = False
            if Phi2InferenceProvider and isinstance(inference_provider, Phi2InferenceProvider):
                is_valid = True
            elif ContemplativeRAGProvider and isinstance(inference_provider, ContemplativeRAGProvider):
                is_valid = True
            
            if not is_valid:
                available = []
                if Phi2InferenceProvider:
                    available.append("Phi2InferenceProvider")
                if ContemplativeRAGProvider:
                    available.append("ContemplativeRAGProvider")
                raise ValueError(
                    f"inference_provider must be one of: {available}. "
                    f"Got: {type(inference_provider).__name__}"
                )
        
        if inference_provider is None:
            # Use HTTP API (backward compatible)
            try:
                import httpx
                self.http_client = httpx.Client(timeout=120.0)
            except ImportError:
                raise ImportError("pip install httpx")
            
            # Auto-discover model if not provided or if auto_select_model is True
            if generator_model is None or auto_select_model:
                available_models = self.get_available_models()
                if available_models:
                    self.generator_model = generator_model or available_models[0]
                    logger.info(f"Using model: {self.generator_model}")
                elif generator_model is None:
                    raise ValueError(
                        f"No models available at {generator_url} and no model specified. "
                        f"Available models: {available_models}"
                    )
                else:
                    self.generator_model = generator_model
            else:
                self.generator_model = generator_model
        else:
            # Using local inference provider
            self.http_client = None
            self.generator_model = None
            logger.info("Using local inference provider")
        
        # Load nan-yar.txt
        script_dir = Path(__file__).parent.parent
        nan_yar_path = script_dir / "ramana/nan-yar.txt"
        if nan_yar_path.exists():
            with open(nan_yar_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.nan_yar = [line for line in content.split('\n') if line.strip()]
        else:
            logger.warning(f"nan-yar.txt not found at {nan_yar_path}")
            self.nan_yar = []
    
    def get_available_models(self) -> list[str]:
        """
        Query the API for available models.
        
        Returns:
            List of model IDs available at the generator_url.
        """
        try:
            resp = self.http_client.get(f"{self.generator_url}/models")
            resp.raise_for_status()
            data = resp.json()
            
            # Handle OpenAI-compatible format: {"data": [{"id": "model1"}, ...]}
            if "data" in data:
                return [model["id"] for model in data["data"]]
            # Handle alternative format: {"models": ["model1", "model2", ...]}
            elif "models" in data:
                return data["models"]
            # Handle direct list: ["model1", "model2", ...]
            elif isinstance(data, list):
                return data
            else:
                logger.warning(f"Unexpected models endpoint format: {data}")
                return []
        except Exception as e:
            logger.warning(f"Failed to fetch available models: {e}")
            return []
    
    def generate(self, user_input: str, conversation_history: Optional[list[dict]] = None) -> dict:
        """
        Generate a response, using critic to filter staleness.
        
        Returns:
            {
                "response": str,  # The response text (or silence marker)
                "is_silence": bool,
                "attempts": int,
                "final_score": float,
                "critic_results": list[CriticResult],
            }
        """
        if conversation_history is None:
            if self.nan_yar:
                conversation_history = [{"role": "system", "content": random.choice(self.nan_yar)}]
            else:
                conversation_history = []
        
        # Check if using ContemplativeRAGProvider (returns multiple responses)
        is_rag_provider = (
            ContemplativeRAGProvider is not None
            and self.inference_provider is not None
            and isinstance(self.inference_provider, ContemplativeRAGProvider)
        )
        
        if is_rag_provider:
            # RAG provider returns multiple responses in one call
            return self._generate_with_rag_provider(user_input, conversation_history)
        else:
            # Standard provider: make multiple attempts
            return self._generate_with_standard_provider(user_input, conversation_history)
    
    def _generate_with_rag_provider(self, user_input: str, conversation_history: list[dict]) -> dict:
        """Generate using RAG provider that returns multiple responses."""
        critic_results = []
        
        # Generate all candidates in one call
        temperature = self.config.temperature_base
        messages = conversation_history + [{"role": "user", "content": user_input}]
        
        try:
            candidates = self.inference_provider.generate(
                query=user_input,
                max_new_tokens=450,
                temperature=temperature,
                do_sample=True,
            )
        except Exception as e:
            logger.warning(f"RAG generation failed: {e}")
            return {
                "response": "[silence]",
                "is_silence": True,
                "attempts": 0,
                "final_score": 0.0,
                "critic_results": [],
            }
        
        if not candidates:
            logger.warning("No candidates returned from RAG provider")
            return {
                "response": "[silence]",
                "is_silence": True,
                "attempts": 0,
                "final_score": 0.0,
                "critic_results": [],
            }
        
        # Evaluate all candidates
        scored_candidates = []
        for candidate in candidates:
            if not candidate:
                continue
            
            if self.critic:
                result = self.critic.evaluate(user_input, candidate)
                critic_results.append(result)
                
                logger.debug(f"Response: {candidate[:80]}")
                logger.debug(f"   score={result.score:.1f}")
                
                scored_candidates.append({"response": candidate, "score": result.score, "result": result})
            else:
                # No critic, use all candidates equally
                scored_candidates.append({"response": candidate, "score": 5.0, "result": None})
        
        if not scored_candidates:
            return {
                "response": "[silence]",
                "is_silence": True,
                "attempts": len(candidates),
                "final_score": 0.0,
                "critic_results": critic_results,
            }
        
        # Filter by threshold
        acceptable = [c for c in scored_candidates if c["score"] >= self.config.threshold]
        
        if acceptable:
            # Select using probabilities weighted by score (normalized)
            if np is None:
                raise ImportError("numpy is required for score-weighted selection")
            
            scores = np.array([c["score"] for c in acceptable])
            # Normalize scores to probabilities (softmax-like, but simpler: just normalize)
            # Shift scores to be positive (add min to ensure all positive)
            min_score = scores.min()
            if min_score < 0:
                scores = scores - min_score + 1.0
            probabilities = scores / scores.sum()
            
            # Sample one response
            selected_idx = np.random.choice(len(acceptable), p=probabilities)
            selected = acceptable[selected_idx]
            
            logger.debug(f"Selected response (score={selected['score']:.1f}) from {len(acceptable)} acceptable candidates")
            
            return {
                "response": selected["response"],
                "is_silence": False,
                "attempts": len(candidates),
                "final_score": selected["score"],
                "critic_results": critic_results,
            }
        else:
            # No acceptable responses
            if self.config.allow_silence:
                best_score = max(c["score"] for c in scored_candidates)
                if best_score < self.config.silence_threshold:
                    return {
                        "response": "[silence]",
                        "is_silence": True,
                        "attempts": len(candidates),
                        "final_score": best_score,
                        "critic_results": critic_results,
                    }
            
            # Return best even if below threshold
            best = max(scored_candidates, key=lambda c: c["score"])
            return {
                "response": best["response"],
                "is_silence": False,
                "attempts": len(candidates),
                "final_score": best["score"],
                "critic_results": critic_results,
            }
    
    def _generate_with_standard_provider(self, user_input: str, conversation_history: list[dict]) -> dict:
        """Generate using standard provider (multiple attempts)."""
        critic_results = []
        best_response = None
        best_score = 0.0
        
        for attempt in range(self.config.max_attempts):
            # Increase temperature on retries
            temperature = self.config.temperature_base + (attempt * self.config.temperature_bump)
            
            # Generate candidate
            candidate = self._generate_candidate(user_input, conversation_history, temperature)
            
            if not candidate:
                continue
            
            # Evaluate with critic
            if self.critic:
                result = self.critic.evaluate(user_input, candidate)
                critic_results.append(result)
                
                logger.debug(f"Attempt {attempt + 1}: response={candidate}")
                logger.debug(f"   score={result.score:.1f}")
                if result.stale_signals:
                    logger.debug(f"  Stale signals: {result.stale_signals[:3]}")
                
                if result.score >= self.config.threshold:
                    # Good enough!
                    return {
                        "response": candidate,
                        "is_silence": False,
                        "attempts": attempt + 1,
                        "final_score": result.score,
                        "critic_results": critic_results,
                    }
                
                # Track best so far
                if result.score > best_score:
                    best_score = result.score
                    best_response = candidate
            else:
                # No critic, return first generation
                return {
                    "response": candidate,
                    "is_silence": False,
                    "attempts": 1,
                    "final_score": -1,
                    "critic_results": [],
                }
        
        # All attempts exhausted
        if self.config.allow_silence and best_score < self.config.silence_threshold:
            # Silence is better than a stale response
            return {
                "response": "[silence]",
                "is_silence": True,
                "attempts": self.config.max_attempts,
                "final_score": best_score,
                "critic_results": critic_results,
            }
        else:
            # Return best attempt even though it's not great
            return {
                "response": best_response or "[silence]",
                "is_silence": best_response is None,
                "attempts": self.config.max_attempts,
                "final_score": best_score,
                "critic_results": critic_results,
            }
    
    def expand(self, query: str, response: str) -> Optional[str]:
        """
        Expand a generated response into a longer, less abstract response 
        in the form of passages from Commentaries.
        
        Args:
            query: The original user query
            response: The generated response to expand
            
        Returns:
            Expanded response string, or None if generation fails
        """
        # Check if RAG provider is available and has passages access
        is_rag_provider = (
            ContemplativeRAGProvider is not None
            and self.inference_provider is not None
            and isinstance(self.inference_provider, ContemplativeRAGProvider)
        )
        
        if is_rag_provider:
            # Use RAG provider with Commentaries passages
            commentaries_passages = self.inference_provider.query_passages(query+'. '+response, top_n=4 )
            
            # Build expansion prompt - add generated_response and ask for longer, less abstract response
            prompt_parts = ["You are a helpful assistant expanding on a response to a user query."]
            prompt_parts.append("Original query:")
            prompt_parts.append(query)
            prompt_parts.append("")
            prompt_parts.append("Your initial response:")
            prompt_parts.append(response)
            prompt_parts.append("")
            
            prompt_parts.append("Nam Yar is a core text of Bhagavan's teachings:")
            prompt_parts.append(NAN_YAR)
            prompt_parts.append("")
            if commentaries_passages:
                prompt_parts.append("Possibly relevant passages from Commentaries:\n")
                for passage in commentaries_passages:
                    prompt_parts.append(passage)
                    prompt_parts.append("")
            
            prompt_parts.append("Expand on the original response into a longer, less abstract form, in the style of the Commentaries passages included above")
            prompt_parts.append("")
            prompt_parts.append("Original query:")
            prompt_parts.append(query)
            prompt_parts.append("")
            prompt_parts.append("Your initial response:")
            prompt_parts.append(response)
            prompt_parts.append("")            
            prompt_parts.append("Do NOT repeat the original response, the goal it to make it more accessible to those not familiar with all of Bhagavan's teachings. Write 2 - 3 sentences, in the more informal style of Bhagavan's statements in the Commentaries passages included above.")
            prompt_parts.append("End your response with: </end>")
            
            expand_prompt = "\n".join(prompt_parts)
            
            try:
                results = self.inference_provider.generate_from_prompt(
                    prompt=expand_prompt,
                    max_new_tokens=500,
                    temperature=0.5,
                    stop_sequence="</end>",
                )
                result_array = results.split('\n')
                for item in result_array:
                    if item.strip():
                        return item.strip()
                return None
            except Exception as e:
                logger.warning(f"Expansion failed: {e}")
                return None
        else:
            # Standard provider - build prompt with generated response
            expand_prompt = f"""Original response:
{response}

Expand this into a longer, less abstract response in the style of Ramana Maharshi's teachings from Face_to_Face. 
Maintain the direct, pointing style. Make it more concrete and less abstract."""
            
            messages = [{"role": "user", "content": expand_prompt}]
            
            if self.inference_provider is not None:
                try:
                    return self.inference_provider.generate_from_messages(
                        messages=messages,
                        max_new_tokens=500,
                        temperature=0.7,
                    )
                except Exception as e:
                    logger.warning(f"Expansion failed: {e}")
                    return None
            
            # Fallback to HTTP API
            try:
                resp = self.http_client.post(
                    f"{self.generator_url}/chat/completions",
                    json={
                        "model": self.generator_model,
                        "messages": messages,
                        "max_tokens": 500,
                        "temperature": 0.7,
                    }
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(f"Expansion failed: {e}")
                return None
    
    def explain(
        self,
        query: str,
        response: str,
        expansion: Optional[str] = None,
    ) -> Optional[str]:
        """
        Explain a generated response.
        
        Args:
            query: The original user query
            response: The generated response to explain
            expansion: Optional expansion of the response
            
        Returns:
            Explanation string, or None if generation fails
        """
        # TODO: Implement explanation logic
        return None
    
    def _generate_candidate(
        self,
        user_input: str,
        history: list[dict],
        temperature: float,
    ) -> Optional[str]:
        """Generate a single candidate response."""
        
        messages = history + [{"role": "user", "content": user_input}]
        
        # Use local inference provider if available
        if self.inference_provider is not None:
            try:
                return self.inference_provider.generate_from_messages(
                    messages=messages,
                    max_new_tokens=450,
                    temperature=temperature,
                )
            except Exception as e:
                logger.warning(f"Local generation failed: {e}")
                return None
        
        # Fallback to HTTP API
        try:
            resp = self.http_client.post(
                f"{self.generator_url}/chat/completions",
                json={
                    "model": self.generator_model,
                    "messages": messages,
                    "max_tokens": 300,  # Keep responses short
                    "temperature": temperature,
                }
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            return None


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_usage(
    checkpoint_dir: Optional[str] = None,
    base_model: str = "microsoft/phi-2",
    critic_provider: str = "local",
    critic_model: Optional[str] = None,
    critic_url: Optional[str] = None,
    test_prompts: Optional[list[str]] = None,
    use_rag: bool = False,
    rag_jsonl: Optional[str] = None,
    rag_backend: str = "local",
    rag_prompt_prefix: Optional[str] = None,
):
    """
    Run the aliveness critic system with a local SFT model or RAG provider.
    
    Args:
        checkpoint_dir: Path to SFT output checkpoint (e.g., "./sft_output/final")
        base_model: Base model name (only used if checkpoint doesn't specify it)
        critic_provider: Critic provider ("local" or "openrouter") - default: "local"
        critic_model: Critic model name (auto-detected for local if None)
        critic_url: Critic API URL for local provider (default: http://localhost:5000/v1)
        test_prompts: List of prompts to test (defaults to built-in examples)
        use_rag: If True, use RAG provider instead of checkpoint
        rag_jsonl: Path to JSONL file for RAG (default: ./ramana/Talks-with-Sri-Ramana-Maharshi-parsed-reviewed-merged.jsonl)
        rag_backend: RAG backend ("local" or "openrouter")
        rag_prompt_prefix: Optional custom prompt prefix for RAG
    """
    import argparse
    
    # Example responses to evaluate
    if test_prompts is None:
        test_prompts = [
            "I feel stuck in my meditation practice.",
            "What should I do with my life?",
            "I'm afraid of death.",
        ]
    
    # Setup critic (local or openrouter only)
    if critic_provider == "local":
        # Use local vLLM critic (auto-detects model from server)
        critic_url = critic_url or "http://localhost:5000/v1"
        logger.info(f"Using local critic at {critic_url} (will auto-detect model)")
        critic = LocalCritic(backend="local", base_url=critic_url, model=critic_model)
    elif critic_provider == "openrouter":
        # Use OpenRouter critic
        logger.info(f"Using OpenRouter critic")
        critic = LocalCritic(backend="openrouter", model=critic_model)
    else:
        raise ValueError(f"Invalid critic_provider: {critic_provider}. Must be 'local' or 'openrouter'")
    
    # Setup generator with inference provider
    if use_rag:
        # Use RAG provider
        if ContemplativeRAGProvider is None:
            raise ValueError(
                "ContemplativeRAGProvider not available. "
                "Ensure contemplative_rag.py is in the same directory and install dependencies: "
                "pip install sentence-transformers faiss-cpu numpy httpx"
            )
        
        rag_jsonl_path = rag_jsonl or "./ramana/Talks-with-Sri-Ramana-Maharshi-parsed-reviewed-merged.jsonl"
        logger.info(f"Using RAG provider with backend: {rag_backend}")
        logger.info(f"Loading Q&A pairs from: {rag_jsonl_path}")
        
        provider = ContemplativeRAGProvider(
            jsonl_path=rag_jsonl_path,
            backend=rag_backend,
            model=None,  # Will use hardcoded default for openrouter
            custom_prompt_prefix=rag_prompt_prefix,
        )
        generator = ContemplativeGenerator(
            inference_provider=provider,
            critic=critic,
        )
    elif checkpoint_dir:
        # Use SFT checkpoint provider
        if Phi2InferenceProvider is None:
            raise ValueError(
                "Phi2InferenceProvider not available. "
                "Ensure phi2-contemplative-inference-lora.py is in the same directory."
            )
        
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        logger.info(f"Loading inference provider from checkpoint: {checkpoint_dir}")
        provider = Phi2InferenceProvider(
            base_model_name=base_model,
            checkpoint_dir=checkpoint_path,
        )
        generator = ContemplativeGenerator(
            inference_provider=provider,
            critic=critic,
        )
    else:
        # Fallback to HTTP API
        logger.warning("No checkpoint_dir or RAG provided, using HTTP API (http://localhost:5000/v1)")
        generator = ContemplativeGenerator(
            generator_url="http://localhost:5000/v1",
            critic=critic,
        )
    
    # Run evaluation
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}/{len(test_prompts)}] User: {prompt}")
        print("-" * 60)
        
        result = generator.generate(prompt)
        generated_response = result['response']
        attempts = result['attempts']
        final_score = result['final_score']
        if final_score is not None and final_score <7.0:
            print(f"Final score: {final_score:.1f} - STALE")
        
        print(f"Generated response: {generated_response}")
        print(f"Attempts: {attempts}, Final score: {final_score:.1f}")

        expansion = generator.expand(query=prompt, response=generated_response)
        print(f"Expansion: {expansion}")
        
        explanation = generator.explain(query=prompt, response=generated_response, expansion=expansion)
        print(f"Explanation: {explanation}\n")
        

def main():
    """Command-line interface for aliveness critic."""
    parser = argparse.ArgumentParser(
        description="Evaluate contemplative model responses using aliveness critic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use SFT output model
  python aliveness_critic.py --checkpoint ./sft_output/final
  
  # Use specific checkpoint
  python aliveness_critic.py --checkpoint ./sft_output/checkpoint-27
  
  # Use custom prompts file
  python aliveness_critic.py --checkpoint ./sft_output/final --prompts eval_prompts.txt
  
  # Use OpenRouter critic instead of local
  python aliveness_critic.py --checkpoint ./sft_output/final --critic-provider openrouter --critic-model anthropic/claude-sonnet-4
  
  # Use RAG provider with local vLLM backend
  python aliveness_critic.py --use-rag --rag-backend local --prompts eval_prompts.txt
  
  # Use RAG provider with OpenRouter backend
  python aliveness_critic.py --use-rag --rag-backend openrouter --prompts eval_prompts.txt
  
  # Use RAG with custom JSONL and prompt prefix
  python aliveness_critic.py --use-rag --rag-backend openrouter --rag-jsonl ./custom.jsonl --rag-prompt-prefix "You are a contemplative teacher..."
        """
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to SFT output checkpoint directory (e.g., ./sft_output/final). "
             "If not provided, uses HTTP API at http://localhost:5000/v1"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="microsoft/phi-2",
        help="Base model name (only used if checkpoint doesn't specify it in adapter_config.json)"
    )
    parser.add_argument(
        "--critic-provider",
        type=str,
        default="local",
        choices=["local", "openrouter"],
        help="Critic provider: 'local' (vLLM port 5000) or 'openrouter' (default: local)"
    )
    parser.add_argument(
        "--critic-model",
        type=str,
        default=None,
        help="Critic model name (default: auto-detect for local, anthropic/claude-sonnet-4 for openrouter)"
    )
    parser.add_argument(
        "--critic-url",
        type=str,
        default=None,
        help="Critic API URL for local provider (default: http://localhost:5000/v1)"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Path to file with test prompts (one per line). If not provided, uses built-in examples."
    )
    parser.add_argument(
        "--use-rag",
        action="store_true",
        help="Use RAG provider instead of checkpoint"
    )
    parser.add_argument(
        "--rag-jsonl",
        type=str,
            default=None,
            help="Path to JSONL file for RAG (default: ./ramana/Talks-parsed_reviewed.jsonl)"
    )
    parser.add_argument(
        "--rag-backend",
        type=str,
        default="local",
        choices=["local", "openrouter"],
        help="RAG backend: 'local' (vLLM) or 'openrouter' (default: local)"
    )
    parser.add_argument(
        "--rag-prompt-prefix",
        type=str,
        default=None,
        help="Optional custom prompt prefix text for RAG (user-editable)"
    )
    
    args = parser.parse_args()
    
    # Load prompts if provided
    test_prompts = None
    if args.prompts:
        prompts_path = Path(args.prompts)
        if prompts_path.exists():
            with open(prompts_path, 'r') as f:
                test_prompts = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.lstrip().startswith("#")
                ]
        else:
            logger.warning(f"Prompts file not found: {args.prompts}, using built-in examples")
    
    example_usage(
        checkpoint_dir=args.checkpoint,
        base_model=args.base_model,
        critic_provider=args.critic_provider,
        critic_model=args.critic_model,
        critic_url=args.critic_url,
        test_prompts=test_prompts,
        use_rag=args.use_rag,
        rag_jsonl=args.rag_jsonl,
        rag_backend=args.rag_backend,
        rag_prompt_prefix=args.rag_prompt_prefix,
    )


if __name__ == "__main__":
    main()
