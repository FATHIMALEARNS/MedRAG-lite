# rag_pipeline.py

from ml.predict import predict
from retriever import SimpleRetriever
from rag import build_prompt
from llm import generate_response

# Instantiate globally to prevent disk-read bottlenecks
global_retriever = SimpleRetriever()


def run_rag_pipeline(image_path, user_query, patient_symptoms=""):
    from ml.validator import is_valid_xray
    import os
    """
    Runs the complete MedRAG pipeline:
    ML prediction → Retrieval → Prompt building → LLM response
    """
    # Convert web URL path (/static/uploads/x.jpg) → absolute disk path
    # so that cv2.imread() and torch can actually read the file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if image_path.startswith("/"):
        disk_path = os.path.join(base_dir, image_path.lstrip("/"))
    else:
        disk_path = image_path

    if not is_valid_xray(disk_path):
        return {
            "error": "This system is designed to analyze chest X-ray images only. Please upload a valid X-ray image."
        }
    # ---- ML Prediction ----
    ml_result = predict(image_path)

    # ---- Retrieval ----
    retriever_query = f"{user_query} {patient_symptoms}".strip()
    retrieved_context = global_retriever.retrieve(retriever_query)

    # ---- Prompt Construction ----
    prompt = build_prompt(
        ml_result=ml_result,
        retrieved_context=retrieved_context,
        user_query=user_query,
        patient_symptoms=patient_symptoms
    )

    # ---- LLM / Fallback ----
    label = ml_result.get("label", "Abnormal")
    final_answer = generate_response(prompt, image_path=image_path, label=label)

    return {
        "ml_result": ml_result,
        "retrieved_context": retrieved_context,
        "prompt": prompt,
        "final_answer": final_answer
    }