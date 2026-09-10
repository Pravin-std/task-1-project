import fitz  # PyMuPDF
import os

try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain_community.docstore.document import Document
    except ImportError:
        from langchain.schema import Document

def load_pdf_chunks(file_path, chunk_size=500):
    """
    Load a PDF file and split it into text chunks with page and source metadata.
    Args:
        file_path (str): Path to the PDF file.
        chunk_size (int): Size of each text chunk.
    Returns:
        List[Document]: List of LangChain Document objects with metadata.
    """
    doc = fitz.open(file_path)
    source_name = os.path.basename(file_path)
    documents = []

    # Get text splitter
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=50,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        except ImportError:
            splitter = None

    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1
        page_text = page.get_text()
        if not page_text.strip():
            continue

        if splitter:
            chunks = splitter.split_text(page_text)
        else:
            step = max(1, chunk_size - 50)
            chunks = [page_text[i:i+chunk_size] for i in range(0, len(page_text), step)]

        for chunk in chunks:
            chunk_str = chunk.strip()
            if chunk_str:
                documents.append(
                    Document(
                        page_content=chunk_str,
                        metadata={
                            "source": source_name,
                            "page": page_num
                        }
                    )
                )

    return documents


