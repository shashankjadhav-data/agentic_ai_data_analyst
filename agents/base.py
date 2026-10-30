# agents/base.py
import time
from dataclasses import dataclass
from typing import Tuple

from ollama import Client  # pip install ollama


@dataclass
class OllamaConfig:
    model: str = "llama3.1:8b"
    temperature: float = 0.1
    max_tokens: int = 256          # speed: fewer tokens
    timeout: float = 90.0          # avoid hangs


class LLMClient:
    """
    Wrapper around Ollama Client with timeout.
    Using Client(timeout=...) is a common way to set request timeouts. [web:130]
    """

    def __init__(self, config: OllamaConfig | None = None):
        self.config = config or OllamaConfig()
        self.client = Client(timeout=self.config.timeout)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Tuple[str, float]:
        cfg = self.config
        max_tokens = max_tokens or cfg.max_tokens
        temperature = temperature if temperature is not None else cfg.temperature

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        start = time.time()
        resp = self.client.chat(
            model=cfg.model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            stream=False,
        )
        duration = time.time() - start

        return resp["message"]["content"].strip(), duration


class Agent:
    def __init__(self, name: str, llm_client: LLMClient, bus):
        self.name = name
        self.llm = llm_client
        self.bus = bus

    def run(self, **kwargs):
        raise NotImplementedError
