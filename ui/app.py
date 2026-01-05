# ui/app.py
import streamlit as st
import sys, os
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.caption_blip import caption
from scripts.query_rag import query

st.set_page_config(page_title="MedRAG-lite", layout="centered")
st.title("🩺 MedRAG-lite — Local demo ")

uploaded = st.file_uploader("Upload chest X-ray", type=["png","jpg","jpeg"])

if uploaded:
    with open("tmp_upload.jpg", "wb") as f:
        f.write(uploaded.getbuffer())

    st.image("tmp_upload.jpg", caption="Uploaded X-ray", use_column_width=True)

    # 🔹 Caption
    if st.button("🧠 Generate Caption"):
        with st.spinner("Generating caption..."):
            cap = caption("tmp_upload.jpg")
        st.subheader("Auto-caption")
        st.write(cap)

    # 🔹 Heatmap grounding (NEW)
    if st.button("🔥 Show Heatmap Grounding"):
        from scripts.heatmap_grounding import generate_heatmap
        import matplotlib.pyplot as plt

        with st.spinner("Generating heatmap..."):
            heatmap = generate_heatmap("tmp_upload.jpg")
            img = Image.open("tmp_upload.jpg").convert("RGB")

            plt.figure(figsize=(6,6))
            plt.imshow(img)
            plt.imshow(heatmap, cmap="jet", alpha=0.5)
            plt.axis("off")
            st.pyplot(plt)

    # 🔹 Explanation
    if st.button("💬 Explain (RAG + LLM)"):
        with st.spinner("Generating explanation..."):
            out = query("tmp_upload.jpg")
        st.subheader("Plain-language summary")
        st.write(out["answer"])
