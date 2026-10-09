import os

from dotenv import load_dotenv
from google import genai


# Force .env values to override existing environment variables
load_dotenv(override=True)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

# Safe check — does NOT print the complete key
print(f"API key loaded: Yes")
print(f"Key starts with: {api_key[:4]}")
print(f"Key length: {len(api_key)}")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say hello and explain RAG in one sentence.",
)

print("\nGemini Response:")
print("-" * 50)
print(response.text)