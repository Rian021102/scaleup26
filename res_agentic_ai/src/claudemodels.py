import anthropic
import os
from dotenv import load_dotenv
load_dotenv()

api_key=os.getenv("api_key")
client = anthropic.Anthropic(api_key=api_key)

print(client.models.list(limit=20))