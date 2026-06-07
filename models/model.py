from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader

def embed_model():
    return SentenceTransformer("microsoft/harrier-oss-v1-270m", model_kwargs={"dtype": "auto"}, device = 'cuda')

def generate_model():
    llm = pipeline(
        "text-generation",
        model="dphn/Dolphin3.0-Qwen2.5-0.5B",
        device_map= 'auto'
    )
    return llm

def reranker_model():
    return CrossEncoder(
        "cross-encoder/ms-marco-TinyBERT-L4"
    )

def pdfloader():
    return PyPDFLoader