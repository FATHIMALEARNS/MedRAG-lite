# test_rag.py

from rag_pipeline import run_rag_pipeline

# -------- INPUTS --------
image_path = "ml/data/val/abnormal/7b7d7bf3-0684-465d-9a74-f51887685387.png"
user_query = "What does this chest X-ray result mean?"

# -------- RUN PIPELINE --------
result = run_rag_pipeline(image_path, user_query)

print("\n===== PROMPT SENT TO LLM =====\n")
print(result["prompt"])

print("\n===== FINAL LLM RESPONSE =====\n")
print(result["final_answer"])