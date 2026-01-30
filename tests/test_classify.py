"""
Simple test for question classification
"""
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def classify_question(question: str) -> str:
    prompt = f"""Classify: "{question}"

Respond ONLY: CASUAL or DOCUMENT_QUERY"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=20
            )
        )
        
        if not response or not response.candidates or len(response.candidates) == 0:
            return "DOCUMENT_QUERY (no response)"
        
        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts or len(candidate.content.parts) == 0:
            return "DOCUMENT_QUERY (no content)"
        
        text = candidate.content.parts[0].text
        if not text:
            return "DOCUMENT_QUERY (no text)"
            
        return text.strip().upper()
        
    except Exception as e:
        return f"ERROR: {e}"

# Test
questions = [
    "Hello",
    "Who are you?",
    "What was NVIDIA's revenue?"
]

for q in questions:
    result = classify_question(q)
    print(f"{q:40} -> {result}")
