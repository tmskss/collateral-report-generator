from openai import OpenAI
from dotenv import load_dotenv

import os

class LLM:
    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(
            base_url="http://mobydick.elte-dh.hu:24642/v1",
            api_key=os.getenv("NLP_API_KEY")
        )

    def invoke(self, messages: list[dict], stream: bool = False, model: str = "Qwen/Qwen3-32B-AWQ"):
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream
        )
