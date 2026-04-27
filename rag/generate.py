import os
from typing import Optional
from dotenv import load_dotenv

import requests


class LLMClient:
    def __init__(self, api_base: str, api_key: str, model_name: str):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name

    @classmethod
    def from_env(cls) -> "LLMClient":
        load_dotenv()
        # Example config for a free-tier OpenAI-compatible endpoint.
        # Replace with concrete provider details in .env.
        api_base = os.getenv("LLM_API_BASE", "")
        api_key = os.getenv("LLM_API_KEY", "")
        model_name = os.getenv("LLM_MODEL_NAME", "llama-3.1-8b-instant")

        if not api_base:
            raise RuntimeError(
                "LLM_API_BASE is not set. Configure a free-tier provider (e.g., Groq, Together) in .env."
            )
        if not api_key:
            raise RuntimeError("LLM_API_KEY is not set. Place your free-tier API key in .env (do NOT commit it).")

        return cls(api_base=api_base, api_key=api_key, model_name=model_name)

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        url = f"{self.api_base}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        # NOTE: This follows the OpenAI-compatible schema used by many free-tier providers.
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
