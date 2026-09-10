import os
import traceback

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

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain.vectorstores import Chroma

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

# Absolute path resolving to current project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")

# Initialize open-source embedding model
def get_embedding_model():
    """Get the open-source embedding model"""
    model_name = "all-MiniLM-L6-v2"  # Lightweight and efficient
    return HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=HF_CACHE_DIR,
        model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
        encode_kwargs={'normalize_embeddings': True}
    )


def create_vector_db(chunks, collection_name="hr_documents"):
    """
    Create a ChromaDB vector database from text chunks or Document objects
    """
    if chunks and hasattr(chunks[0], "page_content"):
        docs = chunks
    else:
        docs = [Document(page_content=chunk, metadata={"chunk_id": i})
                for i, chunk in enumerate(chunks)]

    embeddings = get_embedding_model()
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR,
        collection_name=collection_name
    )

    if hasattr(vectordb, "persist"):
        try:
            vectordb.persist()
        except Exception:
            pass

    return vectordb

def load_vector_db(collection_name="hr_documents"):
    """
    Load existing ChromaDB vector database from absolute path
    """
    embeddings = get_embedding_model()

    if os.path.exists(VECTORSTORE_DIR):
        try:
            vectordb = Chroma(
                persist_directory=VECTORSTORE_DIR,
                embedding_function=embeddings,
                collection_name=collection_name
            )

            # Safely check if database contains stored documents
            doc_count = 0
            if hasattr(vectordb, "_collection") and vectordb._collection is not None:
                try:
                    doc_count = vectordb._collection.count()
                except Exception:
                    try:
                        res = vectordb.get(limit=1)
                        doc_count = len(res.get("ids", []))
                    except Exception:
                        doc_count = 1  # Assume populated if Chroma initialized cleanly

            if doc_count == 0:
                return None

            return vectordb
        except Exception as e:
            print(f"Error loading Chroma vector database: {e}\n{traceback.format_exc()}")
            return None
    else:
        return None


