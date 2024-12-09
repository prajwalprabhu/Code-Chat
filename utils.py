from curses.ascii import EM
from datetime import datetime
import os
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_community.vectorstores import FAISS

from config import EMBEDDINGS_MODEL


def get_loader_for_file(file_path: str):
    file_extension = file_path.split(".")[-1].lower()

    loaders = {
        "txt": TextLoader,
        "pdf": PyPDFLoader,
        "csv": CSVLoader,
        "docx": Docx2txtLoader,
        "xlsx": UnstructuredExcelLoader,
        "xls": UnstructuredExcelLoader,
        "md": UnstructuredMarkdownLoader,
    }

    return loaders.get(file_extension)


def process_file(file_path: str, user_id: int, file_name: str, embeddings_dir: str):
    embeddings_model = HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL, show_progress=True
    )

    """Processes multiple file types, generates embeddings with metadata, and stores them."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    # Get appropriate loader
    LoaderClass = get_loader_for_file(file_path)
    if not LoaderClass:
        raise ValueError(f"Unsupported file type: {file_path}")

    loader = LoaderClass(file_path)
    raw_documents = loader.load()
    documents = text_splitter.split_documents(raw_documents)
    logger.info(f"File Name {file_name}")
    logger.info(f"Documents {len(documents)}")
    logger.info(f"Embeddings model {EMBEDDINGS_MODEL}")
    for i, doc in enumerate(documents):
        doc.metadata = {
            "user_id": user_id,
            "file_name": file_name,
            "chunk_id": i,
            "source": file_path,
            "created_at": datetime.now().isoformat(),
        }

    user_embeddings_dir = os.path.join(embeddings_dir, str(user_id))
    os.makedirs(user_embeddings_dir, exist_ok=True)

    embeddings_path = os.path.join(user_embeddings_dir, "vectorstore.faiss")
    if os.path.exists(embeddings_path):
        vectorstore = FAISS.load_local(
            embeddings_path, embeddings_model, allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(documents)

    else:
        vectorstore = FAISS.from_documents(documents, embeddings_model)

    vectorstore.save_local(embeddings_path)
    return embeddings_path


from loguru import logger
from markdown import markdown
import bleach
from markdown.extensions.codehilite import CodeHiliteExtension


def render_markdown_safely(content: str) -> str:

    # Convert markdown to HTML and sanitize
    html = markdown(
        content,
        extensions=[
            "fenced_code",
        ],
    )
    return html
