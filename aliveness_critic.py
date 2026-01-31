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
    from aliveness_critic import AlivenessCritic, ContemplativeGenerator
    
    critic = AlivenessCritic(model="openai/gpt-4", threshold=6.0)
    generator = ContemplativeGenerator(
        model="local/your-finetuned-model",
        critic=critic,
        max_attempts=3
    )
    
    response = generator.generate("I feel lost in my practice.")

Requirements:
    pip install openai anthropic httpx
"""

import json
import logging
import os
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nan_yar=""""""

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

Respond with a JSON object:
{
    "score": <float 0-10>,
    "alive_signals": [<list of specific alive qualities observed>],
    "stale_signals": [<list of specific stale qualities observed>],
    "reasoning": "<brief explanation>",
    "suggestion": "<optional: how it might be improved, or 'none'>"
}

Score guide:
- 9-10: Genuinely alive, creates space, masterful
- 7-8: Good, mostly alive with minor staleness
- 5-6: Mixed, some alive qualities but closure reflex present
- 3-4: Mostly stale, performing rather than being
- 1-2: Generic spiritual chatbot output
- 0: Actively harmful to contemplative space
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
        lines.append("=" * 60)
        lines.append("ALIVENESS CRITIC RESULT")
        lines.append("=" * 60)
        
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


class LLMCritic(BaseCritic):
    """
    Uses a large LLM (GPT-4, Claude, etc.) as the critic.
    
    The insight: recognition is easier than generation.
    A large model can identify staleness even if it can't avoid generating it.
    """
    
    def __init__(
        self,
        provider: str = "anthropic",  # or "openai"
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        threshold: float = 6.0,
    ):
        self.provider = provider
        self.model = model
        self.threshold = threshold
        
        # Resolve API key from parameter or environment
        if provider == "anthropic":
            if api_key is None:
                # Try CLAUDE_API_KEY first (user's convention), then ANTHROPIC_API_KEY (standard)
                api_key = os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
            self.api_key = api_key
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("pip install anthropic")
        elif provider == "openai":
            if api_key is None:
                api_key = os.environ.get("OPENAI_API_KEY")
            self.api_key = api_key
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("pip install openai")
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def evaluate(self, user_input: str, response: str) -> CriticResult:
        """Evaluate response using LLM critic."""
        
        # Quick check first
        is_obviously_stale, stale_markers = self.quick_stale_check(response)
        if is_obviously_stale:
            return CriticResult(
                score=2.0,
                stale_signals=stale_markers[:5],
                reasoning="Failed quick staleness check (multiple stale patterns detected)",
            )
        
        # Full LLM evaluation
        prompt = ALIVENESS_CRITIC_USER.format(
            user_input=user_input,
            response=response
        )
        
        try:
            if self.provider == "anthropic":
                raw = self._call_anthropic(prompt)
            else:
                raw = self._call_openai(prompt)
            
            return self._parse_result(raw)
            
        except Exception as e:
            logger.warning(f"Critic evaluation failed: {e}")
            # On failure, assume it's borderline
            return CriticResult(score=5.0, reasoning=f"Evaluation error: {e}")
    
    def _call_anthropic(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            system=ALIVENESS_CRITIC_SYSTEM,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def _call_openai(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=500,
            messages=[
                {"role": "system", "content": ALIVENESS_CRITIC_SYSTEM},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    
    def _parse_result(self, raw: str) -> CriticResult:
        """Parse JSON response from critic."""
        try:
            # Extract JSON from response (handle markdown code blocks)
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
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse critic response: {e}\nRaw: {raw}")
            return CriticResult(score=5.0, reasoning=f"Parse error: {e}", raw_response=raw)


class LocalCritic(BaseCritic):
    """
    Uses a local model (via vLLM, SGLang, llama.cpp, etc.) as critic.
    
    For when you want to avoid API costs or need offline operation.
    Requires a capable local model (70B+ recommended for good discrimination).
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "local-model",
        threshold: float = 6.0,
    ):
        self.base_url = base_url
        self.model = model
        self.threshold = threshold
        
        try:
            import httpx
            self.http_client = httpx.Client(timeout=60.0)
        except ImportError:
            raise ImportError("pip install httpx")
    
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
            
        except Exception as e:
            logger.warning(f"Local critic failed: {e}")
            return CriticResult(score=5.0, reasoning=f"Error: {e}")
    
    def _parse_result(self, raw: str) -> CriticResult:
        """Same parsing as LLMCritic."""
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
    ):
        self.generator_url = generator_url
        self.critic = critic
        self.config = config or GenerationConfig()
        
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
        
        # Load nan-yar.txt
        script_dir = Path(__file__).parent
        nan_yar_path = script_dir / "nan-yar.txt"
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
                
                logger.info(f"Attempt {attempt + 1}: response={candidate[:80]}")
                logger.info(f"   score={result.score:.1f}")
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
    
    def _generate_candidate(
        self,
        user_input: str,
        history: list[dict],
        temperature: float,
    ) -> Optional[str]:
        """Generate a single candidate response."""
        
        messages = history + [{"role": "user", "content": user_input}]
        
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

def example_usage():
    """Demonstrate the critic system."""
    
    # Example responses to evaluate
    test_cases = [
        {
            "user": "I feel stuck in my meditation practice.",
            "stale": """I understand how frustrating that can be. Here are some suggestions:
1. Try varying your technique
2. Consider joining a meditation group  
3. Be patient with yourself - progress isn't always linear
Remember, the journey is just as important as the destination.""",
            "alive": "What does 'stuck' feel like?"
        },
        {
            "user": "What should I do with my life?",
            "stale": """That's a profound question that many people struggle with. I'd recommend:
- Reflecting on your core values
- Exploring what brings you joy
- Consider talking to a career counselor or life coach
The key is to align your actions with your authentic self.""",
            "alive": "Who is asking?"
        },
        {
            "user": "I'm afraid of death.",
            "stale": """Fear of death is completely natural and something humans have grappled with 
throughout history. You might find comfort in exploring philosophical perspectives on mortality,
or consider speaking with a therapist who specializes in existential concerns.""",
            "alive": "Yes."
        },
    ]
    
    # Create a mock critic for demonstration
    print("=" * 60)
    print("ALIVENESS CRITIC DEMONSTRATION")
    print("=" * 60)
    
    critic = LLMCritic('anthropic', 'claude-sonnet-4-20250514', api_key=os.environ.get("CLAUDE_API_KEY"))
    # Auto-discover available models from the generator API
    generator = ContemplativeGenerator('http://localhost:5000/v1', critic=critic)
    for case in test_cases:
        print(f"\nUser: {case['user']}")
        print("-" * 40)
        
        # Check stale response
        #analysis = critic.evaluate(case['user'], case['stale'])
        #print(f"\nSTALE response:")
        #print(f" {analysis.display()} \n    \n")      
        
        # Check alive response  
        #analysis = critic.evaluate(case['user'], case['alive'])
        #print(f"\nALIVE response:")
        #print(f" {analysis.display()} \n    \n")      
        
        # Check alive response  
        generated_response = generator.generate(case['user'])['response']
        print(f"Generated response: {generated_response}")
        analysis = critic.evaluate(case['user'], generated_response)
        print(f" {analysis.display()} \n    \n")      
        
        print()


if __name__ == "__main__":
    example_usage()
