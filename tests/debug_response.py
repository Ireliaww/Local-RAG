"""
Debug API response structure
"""
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

prompt = "Classify: 'Hello'. Respond ONLY: CASUAL or DOCUMENT_QUERY"

response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents=prompt,
    config=types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=20
    )
)

print("=== Response Structure ===")
print(f"response: {response}")
print(f"\nresponse.candidates: {response.candidates}")
print(f"\nlen(candidates): {len(response.candidates) if response.candidates else 0}")

if response.candidates and len(response.candidates) > 0:
    candidate = response.candidates[0]
    print(f"\ncandidate: {candidate}")
    print(f"\ncandidate.content: {candidate.content}")
    
    if candidate.content:
        print(f"\ncandidate.content.parts: {candidate.content.parts}")
        print(f"\nlen(parts): {len(candidate.content.parts) if candidate.content.parts else 0}")
        
        if candidate.content.parts and len(candidate.content.parts) > 0:
            part = candidate.content.parts[0]
            print(f"\npart: {part}")
            print(f"\npart.text: {part.text if hasattr(part, 'text') else 'NO TEXT ATTR'}")
            print(f"\ndir(part): {dir(part)}")
