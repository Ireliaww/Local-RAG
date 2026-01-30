"""
Test simplified prompt
"""
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def test_classify(question):
    prompt = f"""Is this casual chat or a document query?

"{question}"

Reply ONE WORD: CASUAL or DOCUMENT_QUERY"""

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
           max_output_tokens=100
        )
    )
    
    print(f"\nQ: {question}")
    print(f"finish_reason: {response.candidates[0].finish_reason}")
    print(f"usage: {response.usage_metadata}")
    
    if response.candidates[0].content and response.candidates[0].content.parts:
        text = response.candidates[0].content.parts[0].text
        print(f"Answer: {text}")
    else:
        print("NO CONTENT!")

# Test
test_classify("Hello")
test_classify("What was NVIDIA's revenue?")
test_classify("Who are you?")
