import os

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
from utils.loader import load_pdf_chunks
from utils.embedder import create_vector_db, load_vector_db
from utils.retriever import simple_qa_response


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

# System & Cache Diagnostic Expander
with st.expander("📊 Disk Space & HuggingFace Cache Diagnostic Info"):
    import shutil
    root_stat = shutil.disk_usage("/")
    nv_stat = shutil.disk_usage("/run/media/pravin/New Volume")
    
    st.write(f"**Root (/) Free Space:** {root_stat.free / (1024**3):.2f} GB / {root_stat.total / (1024**3):.2f} GB")
    st.write(f"**New Volume Free Space:** {nv_stat.free / (1024**3):.2f} GB / {nv_stat.total / (1024**3):.2f} GB")
    st.write(f"**HuggingFace Cache Location:** `{os.environ.get('HF_HOME')}`")
    st.write(f"**Temporary Directory (TMPDIR):** `{os.environ.get('TMPDIR')}`")
    
    if st.button("🧪 Test Loading google/flan-t5-small Model"):
        try:
            with st.spinner("Loading google/flan-t5-small into New Volume cache..."):
                from utils.retriever import get_local_llm
                llm = get_local_llm()
                st.success("✅ google/flan-t5-small loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load LLM model: {e}")

# Sidebar for navigation and file upload

with st.sidebar:
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
    HRBot uses completely open-source AI models:
    • **LLM**: HuggingFace Transformers
    • **Embeddings**: Sentence Transformers
    • **Vector DB**: ChromaDB
    • **No API keys required!**
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

    with st.spinner("📚 Processing and indexing document... This may take a few minutes for the first run."):
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
    # Upload process will populate vector_db
    pass

if vector_db is not None:
    st.subheader("💬 Ask Your HR Questions")

    # Sample questions
    with st.expander("💡 Sample Questions You Can Ask"):
        st.markdown("""
        • What is the company's leave policy?
        • How do I apply for maternity leave?
        • What are the working hours?
        • What is the dress code policy?
        • How does the performance review process work?
        • What benefits are available to employees?
        """)

    with st.expander("🧪 Automated System Self-Test Verification"):
        st.markdown("Click below to test the Q&A chain on the required evaluation test cases.")
        if st.button("▶ Run Full System Verification Test"):
            if os.path.exists(DATA_DIR):
                pdf_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
                if pdf_files:
                    with st.spinner("Indexing PDF with page metadata..."):
                        chunks = load_pdf_chunks(pdf_files[0])
                        create_vector_db(chunks)

            q1 = "How many sick leave days are provided each year?"
            q2 = "What are the core working hours for remote employees?"
            q3 = "What is the policy on space travel to Mars?"
            
            with st.spinner("Testing Question 1 (Sick Leave)..."):
                a1 = simple_qa_response(q1)
            st.markdown(f"**Question 1:** `{q1}`")
            st.markdown(f"**Answer 1:**\n{a1}")
            
            st.markdown("---")
            with st.spinner("Testing Question 2 (Remote Core Hours)..."):
                a2 = simple_qa_response(q2)
            st.markdown(f"**Question 2:** `{q2}`")
            st.markdown(f"**Answer 2:**\n{a2}")

            st.markdown("---")
            with st.spinner("Testing Question 3 (Out-of-Document)..."):
                a3 = simple_qa_response(q3)
            st.markdown(f"**Question 3:** `{q3}`")
            st.markdown(f"**Answer 3:**\n`{a3}`")



    # Chat interface
    user_input = st.text_input(
        "Type your HR question here:",
        placeholder="e.g., What is the company's remote work policy?"
    )

    col1, col2 = st.columns([1, 4])

    with col1:
        ask_button = st.button("🔎 Ask HRBot", type="primary")

    if ask_button and user_input:
        with st.spinner("🤖 HRBot is thinking... Please wait."):
            try:
                response = simple_qa_response(user_input)

                st.markdown("### 🤖 HRBot Response:")
                st.markdown(f'<div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4;">{response}</div>', unsafe_allow_html=True)

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
    if hasattr(st.session_state, 'quick_question'):
        with st.spinner("🤖 Processing quick question..."):
            response = simple_qa_response(st.session_state.quick_question)
            st.markdown("### 🤖 HRBot Response:")
            st.markdown(f'<div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4;">{response}</div>', unsafe_allow_html=True)
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
