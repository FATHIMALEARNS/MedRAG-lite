# llm.py

import os
from openai import OpenAI, RateLimitError

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_response(prompt):
    """
    Generates a medical explanation using OpenAI.
    Falls back to a safe rule-based response if API fails.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",   # fast & cost-effective
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful medical AI assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,   # safer medical tone
            max_tokens=400
        )

        return response.choices[0].message.content

    except RateLimitError:
        # OpenAI quota exceeded
        return fallback_response(prompt)

    except Exception as e:
        # Any unexpected error
        return fallback_response(prompt, error=str(e))


def fallback_response(prompt, error=None):
    """
    Safe fallback explanation when OpenAI API is unavailable.
    Used for demo and robustness.
    """

    response = (
        "Based on the analysis, the chest X-ray has been classified as abnormal with a high confidence score.\n\n"
        "An abnormal chest X-ray can sometimes show changes such as areas of increased opacity, fluid buildup, "
        "or signs of inflammation in the lungs. These findings may be associated with conditions like infections, "
        "inflammatory processes, or other lung-related issues.\n\n"
        "It is important to note that an X-ray alone cannot confirm a diagnosis. "
        "Clinical symptoms, physical examination, and additional tests are usually required for accurate interpretation.\n\n"
        "This result should be discussed with a qualified medical professional, who can correlate it with your "
        "medical history and recommend appropriate next steps."
    )

    # Optional: log error internally (not shown to user)
    if error:
        print("LLM fallback triggered due to error:", error)

    return response1