import os
import hashlib
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Set
from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS

from embeddings import CachedOpenAIEmbeddings


class RecursiveVectorStoreHandler(FileSystemEventHandler):
    # Consolidated and expanded supported file extensions
    SUPPORTED_EXTENSIONS = {
        ".txt",
        ".md",
        ".py",
        ".js",
        ".json",
        ".html",
        ".css",
        ".rs",
        ".cpp",
        ".c",
        ".java",
        ".go",
        ".ts",
        ".php",
        ".rb",
        ".swift",
    }

    # Excluded paths for faster processing
    EXCLUDED_PATHS = {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".cache",
        "dist",
        "build",
    }

    def __init__(
        self,
        folder_path: str,
        embedding_model_name: str = "all-mpnet-base-v2",
        vector_store_path: str = "./faiss_index",
        max_depth: Optional[int] = 4,
        chunk_size: int = 2000,
        chunk_overlap: int = 1000,
        max_workers: int = None,  # Defaults to number of processors
    ):
        """
        Enhanced initialization with more configurable parameters
        """
        self.folder_path = Path(folder_path).resolve()
        self.embedding_model_name = embedding_model_name
        self.vector_store_path = Path(vector_store_path).resolve()
        self.max_depth = max_depth
        self.max_workers = max_workers or (os.cpu_count() or 1)

        # Create vector store path if it doesn't exist
        self.vector_store_path.mkdir(parents=True, exist_ok=True)

        # More robust embeddings and text splitting
        self.embeddings = CachedOpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            show_progress_bar=True,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Thread-safe file tracking
        self.file_hashes = {}
        self._file_hash_lock = threading.Lock()

    def should_ignore_path(self, file_path: Path) -> bool:
        """
        Comprehensive path filtering
        """
        # Convert to string for easier checking
        path_str = str(file_path)

        # Check against excluded paths and patterns
        if any(
            excluded in path_str.split(os.path.sep) for excluded in self.EXCLUDED_PATHS
        ):
            return True

        # Ignore hidden files and specific file types
        if (
            file_path.name.startswith(".")
            or file_path.name.startswith("_")
            or file_path.suffix in {".pyc", ".lock", ".log"}
        ):
            return True

        # Depth check
        try:
            relative_path = file_path.relative_to(self.folder_path)
            depth = len(relative_path.parts) - 1
            return self.max_depth is not None and depth > self.max_depth
        except ValueError:
            return True

    def is_valid_file(self, file_path: Path) -> bool:
        """
        More comprehensive file validation
        """
        # Ignore paths first
        if self.should_ignore_path(file_path):
            return False

        # Check file is a file and has supported extension
        return (
            file_path.is_file()
            and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )

    def get_supported_files(self) -> List[Path]:
        """
        Parallel file discovery using thread pool
        """
        supported_files = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Collect all files first, then filter
            all_files = list(self.folder_path.rglob("*"))

            # Use thread pool to validate files
            futures = {
                executor.submit(self.is_valid_file, file_path): file_path
                for file_path in all_files
            }

            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    if future.result():
                        supported_files.append(file_path)
                except Exception as e:
                    print(f"Error checking file {file_path}: {e}")

        return supported_files

    def generate_file_hash(self, file_path: Path) -> str:
        """
        Faster hash generation with buffered reading
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def update_vector_store(self, changed_file_path: Path):
        """
        More robust vector store update
        """
        # Immediately ignore invalid or excluded paths
        if self.should_ignore_path(changed_file_path):
            return

        try:
            # Validate file extension
            if changed_file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                return

            current_hash = self.generate_file_hash(changed_file_path)
            file_name = changed_file_path.name

            # Thread-safe hash check
            with self._file_hash_lock:
                if (
                    file_name in self.file_hashes
                    and current_hash == self.file_hashes[file_name]
                ):
                    return

            # Load and process document
            loader = TextLoader(str(changed_file_path))
            docs = loader.load()
            splits = self.text_splitter.split_documents(docs)

            # Atomic vector store update
            vectorstore = self._get_or_create_vectorstore()
            self._update_vectorstore_atomically(vectorstore, splits, changed_file_path)

            # Update file hash
            with self._file_hash_lock:
                self.file_hashes[file_name] = current_hash

            print(f"Updated vector store for: {changed_file_path}")
            print(f"Cache hits {self.embeddings.cache_hits}/{len(splits)}")

        except Exception as e:
            print(f"Error processing {changed_file_path}: {e}")

    def initial_vector_store_creation(self):
        """
        Create initial vector store for all supported files
        """
        print("Creating initial vector store...")

        # Get all supported files
        supported_files = self.get_supported_files()

        if not supported_files:
            print("No supported files found.")
            return

        # Collect all document splits
        all_splits = []
        for file_path in supported_files:
            print(f"Loading {file_path}")
            try:
                loader = TextLoader(str(file_path))
                docs = loader.load()
                splits = self.text_splitter.split_documents(docs)
                all_splits.extend(splits)

                # Track file hash
                self.file_hashes[file_path.name] = self.generate_file_hash(file_path)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        # Create vector store
        if all_splits:
            vectorstore = FAISS.from_documents(all_splits, self.embeddings)
            vectorstore.save_local(str(self.vector_store_path))
            print(f"Cache Hits {self.embeddings.cache_hits}/{len(all_splits)}")

    def _get_or_create_vectorstore(self):
        """
        Safely retrieve or create vector store
        """
        try:
            return FAISS.load_local(
                str(self.vector_store_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            return FAISS.from_documents([], self.embeddings)

    def _update_vectorstore_atomically(self, vectorstore, splits, file_path):
        """
        Atomic vector store update with cleanup
        """
        # Remove existing embeddings for this file
        old_doc_ids = [
            doc_id
            for doc_id, doc in vectorstore.docstore._dict.items()
            if doc.metadata.get("source", "").endswith(file_path.name)
        ]
        vectorstore.delete(old_doc_ids)
        self.embeddings.cache_hits = 0
        # Add new document chunks
        for split in splits:
            split.metadata["source"] = str(file_path)

            vectorstore.add_documents([split])

        vectorstore.save_local(str(self.vector_store_path))

    def on_modified(self, event):
        """
        Efficient file modification handler
        """
        if not event.is_directory:
            file_path = Path(event.src_path)
            try:
                file_path.relative_to(self.folder_path)
                self.update_vector_store(file_path)
            except ValueError:
                pass


def watch_folder_and_update_vector_store(
    folder_path: str,
    embedding_model_name: str = "all-mpnet-base-v2",
    vector_store_path: str = "./faiss_index",
    max_depth: Optional[int] = 4,
):
    """
    Simplified folder watching function
    """
    event_handler = RecursiveVectorStoreHandler(
        folder_path, embedding_model_name, vector_store_path, max_depth
    )

    # Parallel initial vector store creation
    event_handler.initial_vector_store_creation()

    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=True)
    observer.start()

    try:
        print(f"Watching folder: {folder_path}")
        print(
            f"Max folder depth: {max_depth if max_depth is not None else 'Unlimited'}"
        )
        print("Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    finally:
        observer.join()


if __name__ == "__main__":
    load_dotenv()
    # Replace with your document folder path
    FOLDER_PATH = os.getenv("FOLDER_PATH")
    if not FOLDER_PATH:
        raise ValueError("FOLDER_PATH environment variable is not set.")
    EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")
    if not EMBEDDINGS_MODEL:
        raise ValueError("EMBEDDINGS_MODEL environment variable is not set.")
    EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH")
    if not EMBEDDINGS_PATH:
        raise ValueError("EMBEDDINGS_PATH environment variable is not set.")
    watch_folder_and_update_vector_store(
        folder_path=FOLDER_PATH,
        embedding_model_name=EMBEDDINGS_MODEL,
        vector_store_path=EMBEDDINGS_PATH,
        max_depth=5,  # Limit to 5 folder levels deep
    )
