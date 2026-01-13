# ui/app.py
import streamlit as st
import sys, os
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.db import create_tables
from scripts.auth import register_user, login_user
from scripts.caption_blip import caption
from scripts.query_rag import query

create_tables()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = None

st.set_page_config(
    page_title="MedRAG-lite",
    page_icon="🩺",
    layout="centered"
)

# =====================================================
# 🔐 AUTH PAGE
# =====================================================
if not st.session_state.logged_in:
    st.title("🩺 MedRAG-lite")
    st.subheader("AI-based Medical Report Explanation System")

    with st.container():
        st.markdown("### 🔐 Login / Sign Up")

        choice = st.radio(
            "Select Action",
            ["Login", "Sign Up"],
            horizontal=True
        )

        email = st.text_input("📧 Email")
        password = st.text_input("🔑 Password", type="password")

        if choice == "Sign Up":
            if st.button("📝 Create Account"):
                if register_user(email, password):
                    st.success("✅ Account created! Please login.")
                else:
                    st.error("❌ Email already exists.")

        if choice == "Login":
            if st.button("🔓 Login"):
                user = login_user(email, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.user_email = email
                    st.experimental_rerun()
                else:
                    st.error("❌ Invalid credentials")

# =====================================================
# 🧠 MAIN APP (AFTER LOGIN)
# =====================================================
else:
    # ---------- SIDEBAR ----------
    st.sidebar.markdown("## 👤 User Panel")
    st.sidebar.success(st.session_state.user_email)

    page = st.sidebar.radio(
        "📂 Navigation",
        ["🏠 Home", "🧠 Analyze X-ray"]
    )

    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.user_email = None
        st.experimental_rerun()

    # ---------- HOME ----------
    if page == "🏠 Home":
        st.title("🏠 Welcome to MedRAG-lite")
        st.markdown("""
        **What this system can do:**
        - 🖼️ Analyze chest X-ray images
        - 🧠 Generate AI-based explanations
        - 🔍 Highlight important regions (Explainable AI)
        - 🔐 Secure user access
        """)

    # ---------- ANALYZE XRAY ----------
    if page == "🧠 Analyze X-ray":
        st.title("🧠 Chest X-ray Analysis")

        uploaded = st.file_uploader(
            "📤 Upload Chest X-ray Image",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded:
            with open("tmp_upload.jpg", "wb") as f:
                f.write(uploaded.getbuffer())

            st.image(
                "tmp_upload.jpg",
                caption="🖼️ Uploaded X-ray",
                use_column_width=True
            )

            col1, col2 = st.columns(2)

            with col1:
                if st.button("🧠 Generate Caption"):
                    with st.spinner("Analyzing image..."):
                        cap = caption("tmp_upload.jpg")
                    st.markdown("### 📝 Image Caption")
                    st.write(cap)

            with col2:
                if st.button("💬 Explain Report"):
                    with st.spinner("Generating explanation..."):
                        out = query("tmp_upload.jpg")
                    st.markdown("### 📄 Plain-language Explanation")
                    st.write(out["answer"])
