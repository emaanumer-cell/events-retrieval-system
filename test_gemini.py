"""Quick test to check if Gemini API key works."""
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY", "")
model = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")

print(f"API key: {api_key[:8]}...{api_key[-4:]}")
print(f"Model: {model}")

from google import genai
from google.genai import types

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model=model,
    contents="Say hello in one sentence.",
    config=types.GenerateContentConfig(temperature=0.5, max_output_tokens=50),
)

print(f"\nResponse: {response.text}")
