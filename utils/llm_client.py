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

    # ------------------------------------------------------------------
    # Tool-calling (function calling) agentic loop
    # ------------------------------------------------------------------
    def chat_with_tools(
        self,
        system: str,
        chat_history: List[dict],
        tools_handler,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tool_iterations: int = 5,
    ):
        """
        Run a full agentic tool-calling turn: the LLM sees the conversation
        so far plus a set of bound tools (from `tools_handler`, a
        utils.tools.CareerAgentTools instance) and decides for itself
        whether to call one or more tools before producing a final answer.

        Args:
            system: system/persona prompt.
            chat_history: list of {"role": "user"|"assistant", "content": str}
                          representing the conversation so far (this IS the
                          session memory -- pass the full history every call).
            tools_handler: object exposing `.specs` (Anthropic format),
                          `.specs_openai_format()`, and `.execute(name, input)`.
            max_tool_iterations: safety cap on tool-call round-trips.

        Returns:
            (final_answer_text: str, tool_call_log: list[dict])
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
                    return self._chat_with_tools_anthropic(
                        system, chat_history, tools_handler, max_tokens, temperature, max_tool_iterations
                    )
                elif provider == "groq":
                    return self._chat_with_tools_groq(
                        system, chat_history, tools_handler, max_tokens, temperature, max_tool_iterations
                    )
            except Exception as e:  # noqa: BLE001
                last_error = e
                continue

        raise LLMError(f"All configured LLM providers failed during tool-calling. Last error: {last_error}")

    def _chat_with_tools_anthropic(self, system, chat_history, tools_handler,
                                    max_tokens, temperature, max_tool_iterations):
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise LLMError(f"anthropic package not installed: {e}")

        client = Anthropic(api_key=self.anthropic_api_key)
        messages = [{"role": m["role"], "content": m["content"]} for m in chat_history]
        tool_call_log = []

        for _ in range(max_tool_iterations):
            response = client.messages.create(
                model=self.anthropic_model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=messages,
                tools=tools_handler.specs,
            )

            if response.stop_reason != "tool_use":
                text_parts = [b.text for b in response.content if getattr(b, "type", "") == "text"]
                return "".join(text_parts).strip(), tool_call_log

            # Model wants to call one or more tools -- execute them and
            # feed the results back in, then let it continue.
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if getattr(block, "type", "") == "tool_use":
                    result_text = tools_handler.execute(block.name, block.input)
                    tool_call_log.append({"tool": block.name, "input": block.input, "result": result_text})
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })
            messages.append({"role": "user", "content": tool_results})

        # Safety cap hit -- return whatever text we can extract, if any.
        return "I wasn't able to fully complete that request within the tool-call limit.", tool_call_log

    def _chat_with_tools_groq(self, system, chat_history, tools_handler,
                               max_tokens, temperature, max_tool_iterations):
        try:
            from groq import Groq
        except ImportError as e:
            raise LLMError(f"groq package not installed: {e}")

        client = Groq(api_key=self.groq_api_key)
        messages = [{"role": "system", "content": system}]
        messages.extend([{"role": m["role"], "content": m["content"]} for m in chat_history])
        tool_call_log = []

        for _ in range(max_tool_iterations):
            response = client.chat.completions.create(
                model=self.groq_model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages,
                tools=tools_handler.specs_openai_format(),
                tool_choice="auto",
            )
            choice = response.choices[0]
            tool_calls = getattr(choice.message, "tool_calls", None)

            if not tool_calls:
                return (choice.message.content or "").strip(), tool_call_log

            messages.append({
                "role": "assistant",
                "content": choice.message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                tool_input = json.loads(tc.function.arguments or "{}")
                result_text = tools_handler.execute(tc.function.name, tool_input)
                tool_call_log.append({"tool": tc.function.name, "input": tool_input, "result": result_text})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text,
                })

        return "I wasn't able to fully complete that request within the tool-call limit.", tool_call_log


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
