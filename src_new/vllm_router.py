#!/usr/bin/env python3
import logging
from openai import OpenAI

class RawTemplateBuilder:
    def build(self, prompt_messages, tokenizer=None):
        return prompt_messages

class VLLMChatCompletion:
    def __init__(self, model_configs: dict):
        self.model_configs = model_configs
        self.clients = {}
        self._initialize_clients()

    def _initialize_clients(self):
        for cfg in self.model_configs.values():
            url = cfg["base_url"]
            if url not in self.clients:
                self.clients[url] = OpenAI(api_key="EMPTY", base_url=url)

    def inference_call(
        self,
        model_alias: str,
        prompt_messages: list,
        max_tokens: int = 100,
        temperature: float = 0.0
    ) -> str:
        cfg = self.model_configs.get(model_alias)
        if cfg is None:
            raise ValueError(f"Unknown model alias: {model_alias}")

        client = self.clients[cfg["base_url"]]
        builder = cfg.get("template_builder")
        messages = builder.build(prompt_messages) if builder else prompt_messages

        if cfg.get("is_chat", False):
            resp = client.chat.completions.create(
                model=cfg["model_id"],
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        else:
            prompt_str = "\n".join(m.get("content", "") for m in messages)
            resp = client.completions.create(
                model=cfg["model_id"],
                prompt=prompt_str,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return getattr(resp.choices[0], "text", "").strip()
