"""
Test with thinking_config
"""
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

prompt = "Classify:  'Hello'. Answer: CASUAL or DOCUMENT_QUERY"

try:
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=100,
            thinking_config={"thinking_mode": "NONE"}
        )
    )
    
    print(f"Success! Response: {response}")
    if response.candidates and len(response.candidates) > 0:
        candidate = response.candidates[0]
        print(f"\nfinish_reason: {candidate.finish_reason}")
        if candidate.content and candidate.content.parts:
            print(f"text: {candidate.content.parts[0].text}")
except Exception as e:
    print(f"Error: {e}")
