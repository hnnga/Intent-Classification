from openai import OpenAI
import json
import os
from dotenv import load_dotenv

load_dotenv()  

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("data/intents.json") as f:
    INTENTS = json.load(f)

def classify_intent(user_question):
    system_prompt = "You are an intents classification. Please give me the correct intent. Only return the intent name. Do not include any explanation. Ensure it should be in the intent options list."
    options = [item["intent"] for item in INTENTS]
    prompt = f"""
            Question: {user_question}
            Intent options: {options}
            Intent:
            """

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": prompt},])

    return completion.choices[0].message.content.strip()
