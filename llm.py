# llm.py

import os
import base64
import openai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_response(prompt, image_path=None, label="Abnormal"):
    """
    Generates a medical explanation using OpenAI GPT-4o-mini Vision.
    Falls back safely if OpenAI API is unavailable.
    """

    try:
        messages = [
            {
                "role": "system",
                "content": "You are a careful medical AI assistant capable of analyzing X-rays."
            }
        ]

        user_content = [{"type": "text", "text": prompt}]

        if image_path and os.path.exists(image_path):
            base64_image = encode_image(image_path)
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "auto"
                }
            })

        messages.append({
            "role": "user",
            "content": user_content
        })

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=500
        )

        return response.choices[0].message.content

    except Exception as e:
        # Catch ALL OpenAI-related failures safely
        print("LLM error occurred:", str(e))
        return fallback_response(label)


def fallback_response(label):
    """
    Safe fallback explanation when OpenAI API fails.
    """
    if label.lower() == "normal":
        return (
            "The chest X-ray has been identified as **Normal**.\n\n"
            "No significant areas of increased opacity, fluid accumulation, "
            "or major signs of inflammation were detected by the local AI.\n\n"
            "It is important to understand that a chest X-ray alone cannot provide a definitive diagnosis. "
            "Clinical symptoms, medical history, and additional investigations are usually required for accurate interpretation."
        )

    return (
        f"The chest X-ray has been identified as **{label}**.\n\n"
        "An abnormal chest X-ray may show changes such as areas of increased opacity, fluid accumulation, "
        "or signs of inflammation in the lungs. These findings can be associated with conditions such as "
        "infections, inflammatory processes, or other lung-related issues.\n\n"
        "It is important to understand that a chest X-ray alone cannot provide a definitive diagnosis. "
        "Clinical symptoms, medical history, and additional investigations are usually required for accurate interpretation.\n\n"
        "Please consult a qualified medical professional for further evaluation and guidance."
    )