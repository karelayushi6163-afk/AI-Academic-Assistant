"""
utils/llm_client.py
Unified LLM client that wraps both the Anthropic Claude API and the Groq API
behind a single `.chat(...)` interface, with automatic provider fallback.

Design goals:
- All 4 agents call one simple method: `llm.chat(system, user)`
- If the primary provider errors out (bad key, rate limit, network issue) and
  a secondary provider's key is available, we automatically retry with it.
- No API key is ever printed or logged.
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from config import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_GROQ_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROVIDER_ORDER,
    DEFAULT_TEMPERATURE,
)


class LLMError(Exception):
    """Raised when no configured provider could complete the request."""
    pass


@dataclass
class LLMClient:
    anthropic_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL
    groq_model: str = DEFAULT_GROQ_MODEL
    provider_order: List[str] = field(default_factory=lambda: list(DEFAULT_PROVIDER_ORDER))

    def available_providers(self) -> List[str]:
        providers = []
        if self.anthropic_api_key:
            providers.append("anthropic")
        if self.groq_api_key:
            providers.append("groq")
        # Respect configured order, but only include ones that actually have a key.
        return [p for p in self.provider_order if p in providers] or providers

    # ------------------------------------------------------------------
    # Provider-specific calls
    # ------------------------------------------------------------------
    def _call_anthropic(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise LLMError(f"anthropic package not installed: {e}")

        client = Anthropic(api_key=self.anthropic_api_key)
        response = client.messages.create(
            model=self.anthropic_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = [block.text for block in response.content if getattr(block, "type", "") == "text"]
        return "".join(parts).strip()

    def _call_groq(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        try:
            from groq import Groq
        except ImportError as e:
            raise LLMError(f"groq package not installed: {e}")

        client = Groq(api_key=self.groq_api_key)
        response = client.chat.completions.create(
            model=self.groq_model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        """
        Send a chat request, trying each available provider in order until
        one succeeds. Raises LLMError if all providers fail (or none are
        configured).
        """
        providers = self.available_providers()
        if not providers:
            raise LLMError(
                "No LLM API key configured. Please enter an Anthropic or Groq "
                "API key in the sidebar."
            )

        last_error = None
        for provider in providers:
            try:
                if provider == "anthropic":
                    return self._call_anthropic(system, user, max_tokens, temperature)
                elif provider == "groq":
                    return self._call_groq(system, user, max_tokens, temperature)
            except Exception as e:  # noqa: BLE001 - deliberately broad, we fall back
                last_error = e
                continue

        raise LLMError(
            f"All configured LLM providers failed. Last error: {last_error}"
        )

    def chat_json(
        self,
        system: str,
        user: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        """
        Convenience wrapper that asks the model for JSON and parses it,
        stripping markdown code fences and any stray leading/trailing prose.
        """
        raw = self.chat(system, user, max_tokens=max_tokens, temperature=temperature)
        return extract_json(raw)


def extract_json(text: str):
    """
    Extract and parse the first valid JSON object/array found in a string.
    Handles common LLM quirks: ```json fences, leading/trailing commentary.
    """
    if not text:
        raise LLMError("Empty response from LLM; could not extract JSON.")

    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned.strip(), flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned.strip()).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: find the widest {...} or [...] span in the text.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    raise LLMError("Could not parse JSON from LLM response.")
