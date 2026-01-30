import streamlit as st
import sys, os
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import auth functions
from ui.auth import add_userdata, login_user

# Import app logic
from scripts.caption_blip import caption
from scripts.query_rag import query

st.set_page_config(page_title="MedRAG-lite", layout="centered")

# Custom CSS for Fantastic UI
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f8ff;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Helvetica', sans-serif;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border-radius: 12px;
        padding: 10px 24px;
        font-size: 16px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    div[data-testid="stFileUploadDropzone"] {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# Session State
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""

def login_page():
    st.title("Welcome to MedRAG-lite 🩺")
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        st.subheader("Login to your Account")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type='password', key="login_pass")
        if st.button("Login"):
            result = login_user(username, password)
            if result:
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.success(f"Welcome back {username}")
                st.rerun()
            else:
                st.warning("Incorrect Username/Password")

    with tab2:
        st.subheader("Create New Account")
        new_user = st.text_input("Username", key="signup_user")
        new_password = st.text_input("Password", type='password', key="signup_pass")
        if st.button("Sign Up"):
            if add_userdata(new_user, new_password):
                st.success("You have successfully created an account")
                st.info("Go to Login Menu to login")
            else:
                st.warning("Username already exists")

def main_app():
    st.sidebar.write(f"Logged in as: **{st.session_state['username']}**")
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = ""
        st.rerun()

    st.title("🩺 MedRAG-lite — Analysis Dashboard")

    uploaded = st.file_uploader("Upload chest X-ray", type=["png","jpg","jpeg"])

    if uploaded:
        with open("tmp_upload.jpg", "wb") as f:
            f.write(uploaded.getbuffer())

        st.image("tmp_upload.jpg", caption="Uploaded X-ray", use_column_width=True)

        # 🔹 Caption
        if st.button("🧠 Generate Caption"):
            with st.spinner("Generating caption..."):
                try:
                    cap = caption("tmp_upload.jpg")
                    st.subheader("Auto-caption")
                    st.write(cap)
                except Exception as e:
                    st.error(f"Error generating caption: {e}")

        # 🔹 Heatmap grounding
        if st.button("🔥 Show Heatmap Grounding"):
            # Lazy import to avoid errors if module missing
            try:
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
            except ImportError:
                 st.warning("Heatmap module not found or failed to load.")
            except Exception as e:
                 st.error(f"Error: {e}")

        # 🔹 Explanation
        if st.button("💬 Explain (RAG + LLM)"):
            with st.spinner("Generating explanation..."):
                try:
                    out = query("tmp_upload.jpg")
                    st.subheader("Plain-language summary")
                    st.write(out["answer"])
                except Exception as e:
                    st.error(f"Error generating explanation: {e}")

if st.session_state['logged_in']:
    main_app()
else:
    login_page()
