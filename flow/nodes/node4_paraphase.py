from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()  

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_paraphrases(question, n=30):
    prompt = f"Generate {n} diverse paraphrases of the question: '{question}'. Respond in JSON list format."
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    content = completion.choices[0].message.content

    try:
        # if output covered in ```json
        if content.startswith("```"):
            content = content.strip("`").strip()
            if content.lower().startswith("json"):
                content = content[4:].strip()

        paraphrases = json.loads(content)
        return paraphrases
    except json.JSONDecodeError:
        print("Error parsing JSON:", content)
        return []