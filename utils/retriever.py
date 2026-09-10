import os
import traceback
import logging
from typing import Optional, List, Any
import torch

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

from langchain.chains import RetrievalQA

try:
    from langchain_core.language_models.llms import LLM
except ImportError:
    from langchain.llms.base import LLM

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.embedder import load_vector_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlanT5LLM(LLM):
    """Custom LangChain LLM wrapper for google/flan-t5-small"""
    model: Any = None
    tokenizer: Any = None

    @property
    def _llm_type(self) -> str:
        return "flan-t5-small"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        if self.tokenizer is None or self.model is None:
            raise ValueError("FlanT5LLM model or tokenizer is not initialized.")
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=256
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_local_llm():
    """
    Initialize local google/flan-t5-small stored on New Volume cache
    """
    model_name = "google/flan-t5-small"

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=HF_CACHE_DIR
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=HF_CACHE_DIR,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False
        )
        model.eval()

        llm = FlanT5LLM(model=model, tokenizer=tokenizer)
        return llm

    except Exception as e:
        err_msg = f"Error loading LLM model '{model_name}': {str(e)}"
        logger.error(f"{err_msg}\n{traceback.format_exc()}")
        raise RuntimeError(err_msg)



def get_hr_prompt_template():
    """
    Create an HR-specific prompt template compatible with Flan-T5
    """
    template = """Answer the question based only on the following HR document context. If the answer cannot be found in the context, say "I could not find information about that in the HR documents."

Context:
{context}

Question: {question}

Answer:"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

def get_qa_chain():
    """
    Create a QA chain using open-source components
    """
    vector_db = load_vector_db()

    if vector_db is None:
        raise ValueError("Vector database is empty or not loaded. Please ensure an HR PDF document is uploaded and indexed.")

    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    llm = get_local_llm()

    if llm is None:
        raise ValueError("LLM model failed to load.")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": get_hr_prompt_template()
        }
    )

    return qa_chain

def simple_qa_response(question, max_retries=1):
    """
    Simple Q&A function with transparent error logging, source citations, and fallback logic.
    """
    try:
        qa_chain = get_qa_chain()

        if hasattr(qa_chain, "invoke"):
            result = qa_chain.invoke({"query": question})
        else:
            result = qa_chain({"query": question})

        answer = result.get("result", "").strip()
        source_docs = result.get("source_documents", [])

        # Check if answer indicates missing info or fallback
        lower_ans = answer.lower()
        not_found_triggers = [
            "could not find", "not found", "cannot find", "not available",
            "no information", "don't know", "does not mention", "not mentioned"
        ]

        if any(trigger in lower_ans for trigger in not_found_triggers):
            return "I could not find information about that in the uploaded HR documents."

        # Format source document citations
        citations = []
        seen = set()
        for doc in source_docs:
            src = doc.metadata.get("source", "HR_Policy_Test_Dataset.pdf")
            page = doc.metadata.get("page")
            key = (src, page)
            if key not in seen:
                seen.add(key)
                if page:
                    citations.append(f"📄 **Source:** `{src}` (Page {page})")
                else:
                    citations.append(f"📄 **Source:** `{src}`")

        if citations:
            return f"{answer}\n\n" + "\n".join(citations)
        else:
            return answer

    except Exception as e:
        err_trace = traceback.format_exc()
        logger.error(f"QA Error: {e}\n{err_trace}")
        return f"System Error: {str(e)}"



