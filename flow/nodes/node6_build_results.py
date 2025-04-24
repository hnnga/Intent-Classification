from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()  

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_results(intent, question):
    prompt = f"""
Given the intent '{intent}', generate an appropriate query or action string to handle this question:
'{question}'
Return a short action string. Do not include prefixes like 'Action:'.
"""
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content.strip()
