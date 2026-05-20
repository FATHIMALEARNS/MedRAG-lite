# rag.py

PROMPT_TEMPLATE = """
You are a medical AI assistant designed to support clinical understanding.

A chest X-ray image is attached to this message. It was pre-screened by our local ResNet vision model with the following result:
- Predicted label: {label}
- Confidence score: {probability}
- Confidence level: {confidence}

Patient Symptoms & Clinical Context:
{patient_symptoms}

Relevant medical context retrieved from our trusted knowledge base:
{retrieved_context}

User question:
{user_query}

Instructions for analysis:
1. Carefully examine the attached X-ray image.
2. If the local model flagged it as abnormal or if you spot issues, point out specific visual findings (e.g., opacities, effusions, consolidation).
3. Actively cross-reference the patient's symptoms (if provided) with your visual findings to build a holistic clinical picture.
4. Connect your visual findings to the retrieved medical context and the user query.
5. Explain the findings in simple, clear language.
6. Emphasize uncertainty and recommend professional medical consultation.
7. Do NOT give definitive medical medical diagnosis.

Final response:
"""

def build_prompt(ml_result, retrieved_context, user_query, patient_symptoms=""):
    symptoms_text = patient_symptoms if patient_symptoms else "No specific patient symptoms or history were provided."
    return PROMPT_TEMPLATE.format(
        label=ml_result["label"],
        probability=ml_result["probability"],
        confidence=ml_result["confidence"],
        patient_symptoms=symptoms_text,
        retrieved_context=retrieved_context,
        user_query=user_query
    )