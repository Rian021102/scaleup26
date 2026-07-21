import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv



project_root = Path(__file__).resolve().parents[1]
load_dotenv(project_root / ".env")
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise RuntimeError("HF_TOKEN is not set. Add it to your .env file.")

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=hf_token)

response = client.chat.completions.create(
    model="moonshotai/Kimi-K2.6:novita",
    messages=[{"role": "user", "content": "What is quantum computing?"}],
)

print(response.choices[0].message.content)