"""
Quick test to debug API response format
"""
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

prompt = """Classify: "Hello"

Respond with ONLY: CASUAL or DOCUMENT_QUERY"""

response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents=prompt,
    config=types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=10
    )
)

print(f"Response type: {type(response)}")
print(f"Response: {response}")
print(f"Response.text: {response.text}")
print(f"Response.text type: {type(response.text)}")
