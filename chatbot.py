import os

from dotenv import load_dotenv
from google import genai

from prompt import SYSTEM_PROMPT

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def ask_gemini(user_message):

    prompt = f"""
{SYSTEM_PROMPT}

User Question:
{user_message}
"""

    try:

        response = client.models.generate_content(
             model="gemini-3.5-flash",
            contents=prompt
        )

        if (
            response.text
            and response.text.strip()
        ):
            return response.text.strip()

        return (
            "⚠️ Sorry, I couldn't generate a response."
        )

    except Exception as e:

        print("Gemini Error:", e)

        return (
            "⚠️ Unable to connect to Gemini."
        )


def is_hospital_query(message):

    keywords = [

        "hospital",
        "hospitals",
        "near",
        "nearby",
        "nearest",
        "doctor",
        "clinic",
        "emergency",
        "specialist",
        "pulmonologist"

    ]

    message = message.lower()

    return any(
        word in message
        for word in keywords
    )