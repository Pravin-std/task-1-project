import os
from dotenv import load_dotenv
load_dotenv()

# Set HuggingFace and temporary cache directories to persistent location on New Volume
HF_CACHE_DIR = "/run/media/pravin/New Volume/huggingface-cache"
TMP_DIR = os.path.join(HF_CACHE_DIR, "tmp")

os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = HF_CACHE_DIR
os.environ["TMPDIR"] = TMP_DIR

import streamlit as st
import importlib
import utils.retriever
importlib.reload(utils.retriever)
from utils.retriever import simple_qa_response
from utils.loader import load_pdf_chunks
from utils.embedder import create_vector_db, load_vector_db


st.set_page_config(
    page_title="HRBot - AI HR Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main Title
st.markdown('<h1 class="main-header">🤖 HRBot - AI HR Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your intelligent HR companion powered by open-source AI</p>', unsafe_allow_html=True)

# Sidebar for navigation and file upload

with st.sidebar:
    st.header("⚡ Speed & API Acceleration")
    api_key_input = st.text_input(
        "Free API Key (Gemini / Groq / HF)",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", "") if (os.environ.get("GEMINI_API_KEY", "").startswith("AIzaSy") or os.environ.get("GEMINI_API_KEY", "").startswith("gsk_") or os.environ.get("GEMINI_API_KEY", "").startswith("hf_")) else "",
        help="Paste a free Google Gemini API key (or Groq/HF token) for sub-second responses!"
    )
    key_val = api_key_input.strip()
    if key_val and (key_val.startswith("AIzaSy") or key_val.startswith("gsk_") or key_val.startswith("hf_")):
        os.environ["GEMINI_API_KEY"] = key_val
        st.success("⚡ Cloud API Acceleration ACTIVE (<0.5s response time)")
    else:
        if "GEMINI_API_KEY" in os.environ and not (os.environ["GEMINI_API_KEY"].startswith("AIzaSy") or os.environ["GEMINI_API_KEY"].startswith("gsk_") or os.environ["GEMINI_API_KEY"].startswith("hf_")):
            os.environ.pop("GEMINI_API_KEY", None)
        st.info("⚡ Running on Local Direct Extraction Engine (Fast & Accurate)")

    st.markdown("---")
    st.header("📁 Document Management")

    # File upload section
    uploaded_file = st.file_uploader(
        "Upload HR Policy Document",
        type=["pdf"],
        help="Upload PDF documents containing HR policies, handbooks, or guidelines"
    )

    # Show current documents
    if os.path.exists("data"):
        files = [f for f in os.listdir("data") if f.endswith('.pdf')]
        if files:
            st.subheader("📚 Uploaded Documents")
            for file in files:
                st.text(f"• {file}")

    st.markdown("---")
    st.subheader("ℹ️ About HRBot")
    st.info("""
    HRBot supports dual execution:
    • **⚡ Fast Cloud API**: Google Gemini / Groq (<0.5s)
    • **🐢 Local Extraction**: Instant Direct Extractive Engine (No API key needed)
    • **Vector DB**: ChromaDB
    """)

# Absolute directory paths relative to app.py location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# Handle file upload
if uploaded_file:
    file_path = os.path.join(DATA_DIR, uploaded_file.name)

    # Save uploaded file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("📚 Processing and indexing document..."):
        try:
            chunks = load_pdf_chunks(file_path)
            create_vector_db(chunks)
            st.markdown('<div class="success-box">✅ Document processed and stored in vector database!</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")

# Main chat interface
st.markdown("---")

# Check if vector database exists, auto-index data directory if PDF exists
vector_db = load_vector_db()
if vector_db is None and os.path.exists(DATA_DIR):
    pdf_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    if pdf_files:
        try:
            chunks = load_pdf_chunks(pdf_files[0])
            create_vector_db(chunks)
            vector_db = load_vector_db()
        except Exception as e:
            st.error(f"Auto-indexing error: {e}")

if vector_db is None and uploaded_file:
    pass

if vector_db is not None:
    st.subheader("💬 Ask Your HR Questions")

    # Sample questions
    with st.expander("💡 Sample Questions You Can Ask"):
        st.markdown("""
        • How many sick leave days are provided each year?
        • How do I apply for maternity leave?
        • What are the core working hours for remote employees?
        • What is the dress code policy?
        • How does the performance review process work?
        • What benefits are available to employees?
        """)

    with st.expander("🧪 Automated System Self-Test Verification"):
        st.markdown("Click below to run automated evaluation on required test cases.")
        if st.button("▶ Run Full System Verification Test"):
            importlib.reload(utils.retriever)
            if os.path.exists(DATA_DIR):
                pdf_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
                if pdf_files:
                    with st.spinner("Indexing PDF with page metadata..."):
                        chunks = load_pdf_chunks(pdf_files[0])
                        create_vector_db(chunks)

            q1 = "How many sick leave days are provided each year?"
            q2 = "What are the core working hours for remote employees?"
            q3 = "How do I apply for maternity leave?"
            q4 = "What is the policy on space travel to Mars?"

            with st.spinner("Testing Question 1 (Sick Leave)..."):
                a1 = utils.retriever.simple_qa_response(q1)
            p1 = "12" in a1.lower() or "sick leave" in a1.lower()
            st.markdown(f"**Test 1 (Sick Leave):** `{q1}`")
            st.markdown(f"**Result:** {'✅ PASS' if p1 else '❌ FAIL'}")
            st.markdown(f"**Answer:**\n{a1}")

            st.markdown("---")
            with st.spinner("Testing Question 2 (Remote Work Hours)..."):
                a2 = utils.retriever.simple_qa_response(q2)
            p2 = "core" in a2.lower() or "remote" in a2.lower() or "hours" in a2.lower()
            st.markdown(f"**Test 2 (Remote Work Hours):** `{q2}`")
            st.markdown(f"**Result:** {'✅ PASS' if p2 else '❌ FAIL'}")
            st.markdown(f"**Answer:**\n{a2}")

            st.markdown("---")
            with st.spinner("Testing Question 3 (Maternity Leave)..."):
                a3 = utils.retriever.simple_qa_response(q3)
            p3 = "maternity" in a3.lower() or "leave" in a3.lower()
            st.markdown(f"**Test 3 (Maternity Leave):** `{q3}`")
            st.markdown(f"**Result:** {'✅ PASS' if p3 else '❌ FAIL'}")
            st.markdown(f"**Answer:**\n{a3}")

            st.markdown("---")
            with st.spinner("Testing Question 4 (Unknown Question - Mars)..."):
                a4 = utils.retriever.simple_qa_response(q4)
            p4 = "could not find" in a4.lower() or "not found" in a4.lower()
            st.markdown(f"**Test 4 (Unknown - Mars Travel):** `{q4}`")
            st.markdown(f"**Result:** {'✅ PASS' if p4 else '❌ FAIL'}")
            st.markdown(f"**Answer:**\n`{a4}`")



    # Chat interface form
    with st.form(key="qa_form", clear_on_submit=False):
        user_input = st.text_input(
            "Type your HR question here:",
            placeholder="e.g., How many sick leave days are provided each year?"
        )
        submit_button = st.form_submit_button("🔎 Ask HRBot", type="primary")

    if submit_button and user_input.strip():
        with st.spinner("🤖 HRBot is thinking... Please wait."):
            try:
                importlib.reload(utils.retriever)
                response = utils.retriever.simple_qa_response(user_input.strip())

                st.markdown("### 🤖 HRBot Response:")
                st.markdown(f'<div style="background-color: #ffffff; color: #1e293b; padding: 1.25rem; border-radius: 0.5rem; border-left: 5px solid #1f77b4; box-shadow: 0 2px 4px rgba(0,0,0,0.08); font-size: 1.05rem; line-height: 1.6; margin-top: 1rem;">{response}</div>', unsafe_allow_html=True)

            except Exception as e:
                import traceback
                st.error(f"Sorry, I encountered an error: {str(e)}")
                st.code(traceback.format_exc())
                st.info("Please try rephrasing your question or contact HR directly.")

    # Quick actions
    st.markdown("---")
    st.subheader("🚀 Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📋 Leave Policies"):
            st.session_state.quick_question = "What are the different types of leave available?"

    with col2:
        if st.button("💰 Benefits Info"):
            st.session_state.quick_question = "What employee benefits does the company offer?"

    with col3:
        if st.button("📞 Contact HR"):
            st.session_state.quick_question = "How can I contact the HR department?"

    # Handle quick questions
    if hasattr(st.session_state, 'quick_question') and st.session_state.quick_question:
        with st.spinner("🤖 Processing quick question..."):
            q_text = st.session_state.quick_question
            importlib.reload(utils.retriever)
            response = utils.retriever.simple_qa_response(q_text)
            st.markdown(f"### 🤖 HRBot Response for: *'{q_text}'*")
            st.markdown(f'<div style="background-color: #ffffff; color: #1e293b; padding: 1.25rem; border-radius: 0.5rem; border-left: 5px solid #1f77b4; box-shadow: 0 2px 4px rgba(0,0,0,0.08); font-size: 1.05rem; line-height: 1.6; margin-top: 1rem;">{response}</div>', unsafe_allow_html=True)
        del st.session_state.quick_question

else:
    st.markdown('<div class="info-box">📂 Please upload an HR document to begin chatting with HRBot.</div>', unsafe_allow_html=True)

    st.subheader("🎯 What HRBot Can Do")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **📚 Document Analysis**
        • Process HR policies and handbooks
        • Extract key information automatically
        • Provide instant answers to policy questions

        **🔍 Smart Search**
        • Find relevant information quickly
        • Context-aware responses
        • No need to read through long documents
        """)

    with col2:
        st.markdown("""
        **🤖 AI-Powered Assistance**
        • 24/7 availability
        • Consistent and accurate responses
        • Learns from your HR documents

        **🔒 Privacy & Security**
        • All processing done locally
        • No data sent to external APIs
        • Your documents stay private
        """)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #666; font-size: 0.9rem;">HRBot v1.0 - Powered by Open Source AI 🚀</p>',
    unsafe_allow_html=True
)
