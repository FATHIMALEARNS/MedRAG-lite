# MedRAG-lite 

MedRAG-lite is a lightweight, local **Medical Retrieval-Augmented Generation (RAG)** system
designed to analyze **Chest X-ray images** using image captioning, grounding, and
LLM-based explanation.

This project is intended as an academic and learning-oriented medical AI demo.

---

## Features

- Upload Chest X-ray images via a simple UI
- Automatic image captioning using BLIP
- Grounding based on medical prompts (e.g., opacity)
- RAG-style explanation fallback
- Runs completely **locally**
- No cloud or external APIs required

---

## Tech Stack

- Python
- Streamlit
- PyTorch
- Transformers
- Computer Vision models (captioning & grounding)

---

## Project Structure
MedRAG-lite/
│
├── scripts/ # Core ML & RAG logic
│ ├── caption_blip.py
│ ├── grounding_infer.py
│ ├── query_rag.py
│ ├── build_faiss_index.py
│ └── extract_features.py
│
├── ui/ # Streamlit UI
│ ├── app.py # Main application entry point
│ └── test_app.py
│
├── utils/ # Helper utilities
│
├── requirements.txt
└── README.md

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
streamlit run ui/app.py



