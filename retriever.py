# retriever.py

import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")

class SimpleRetriever:
    def __init__(self):
        self.documents = []
        self.doc_names = []

        for file in os.listdir(KB_DIR):
            if file.endswith(".txt"):
                path = os.path.join(KB_DIR, file)
                with open(path, "r", encoding="utf-8") as f:
                    self.documents.append(f.read())
                    self.doc_names.append(file)

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_vectors = self.vectorizer.fit_transform(self.documents)

    def retrieve(self, query, top_k=2):
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_vectors)[0]

        top_indices = similarities.argsort()[::-1][:top_k]

        retrieved_text = []
        for idx in top_indices:
            retrieved_text.append(self.documents[idx])

        return "\n\n".join(retrieved_text)