import os
import sqlite3
import json
import hashlib
from typing import List, Optional
from langchain_community.embeddings import OpenAIEmbeddings
from pydantic import BaseModel, Field, ConfigDict


class SQLiteEmbeddingsCache:
    """
    Provides a SQLite-based caching mechanism for OpenAI embeddings.

    Features:
    - Persistent caching of embeddings
    - Efficient storage and retrieval
    - Tracks embedding metadata
    """

    def __init__(
        self,
        cache_db_path: str = ".embeddings_cache.sqlite",
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize SQLite embeddings cache.

        Args:
            cache_db_path (str): Path to SQLite database
            embedding_model (str, optional): Name of embedding model
        """
        self.cache_db_path = os.path.abspath(cache_db_path)
        self.embedding_model = embedding_model or "default"

        # Create cache database and tables
        self._create_cache_table()

    def _create_cache_table(self):
        """Create SQLite table for embedding cache if not exists."""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS embeddings_cache (
            text_hash TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            model TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
        )

        # Create index for faster lookups
        cursor.execute(
            """
        CREATE INDEX IF NOT EXISTS idx_text_hash 
        ON embeddings_cache(text_hash, model)
        """
        )

        conn.commit()
        conn.close()

    def _hash_text(self, text: str) -> str:
        """
        Generate a consistent hash for the input text.

        Args:
            text (str): Input text to hash

        Returns:
            str: SHA-256 hash of the text
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def store_embedding(
        self, text: str, embedding: List[float], model: Optional[str] = None
    ):
        """
        Store an embedding in the SQLite cache.

        Args:
            text (str): Original text
            embedding (List[float]): Embedding vector
            model (str, optional): Embedding model name
        """
        # Use default model if not specified
        model = model or self.embedding_model

        # Generate hash
        text_hash = self._hash_text(text)

        # Convert embedding to binary
        embedding_blob = json.dumps(embedding).encode("utf-8")

        # Connect and insert
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
            INSERT OR REPLACE INTO embeddings_cache 
            (text_hash, text, embedding, model) 
            VALUES (?, ?, ?, ?)
            """,
                (text_hash, text, embedding_blob, model),
            )

            conn.commit()
        except sqlite3.Error as e:
            print(f"Error storing embedding: {e}")
        finally:
            conn.close()

    def retrieve_embedding(
        self, text: str, model: Optional[str] = None
    ) -> Optional[List[float]]:
        """
        Retrieve an embedding from the cache.

        Args:
            text (str): Original text
            model (str, optional): Embedding model name

        Returns:
            Optional[List[float]]: Cached embedding or None
        """
        # Use default model if not specified
        model = model or self.embedding_model

        # Generate hash
        text_hash = self._hash_text(text)

        # Connect and retrieve
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
            SELECT embedding 
            FROM embeddings_cache 
            WHERE text_hash = ? AND model = ?
            """,
                (text_hash, model),
            )

            result = cursor.fetchone()

            if result:
                # Decode and parse embedding
                return json.loads(result[0].decode("utf-8"))

            return None
        except sqlite3.Error as e:
            print(f"Error retrieving embedding: {e}")
            return None
        finally:
            conn.close()

    def clear_cache(self, older_than_days: int = 30):
        """
        Clear old cache entries.

        Args:
            older_than_days (int): Remove entries older than specified days
        """
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
            DELETE FROM embeddings_cache 
            WHERE created_at < datetime('now', ?)
            """,
                (f"-{older_than_days} days",),
            )

            conn.commit()
            print(f"Cleared cache entries older than {older_than_days} days")
        except sqlite3.Error as e:
            print(f"Error clearing cache: {e}")
        finally:
            conn.close()


class CachedOpenAIEmbeddings(OpenAIEmbeddings):
    """
    Extended OpenAIEmbeddings with SQLite-based caching.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    cache: SQLiteEmbeddingsCache = Field(
        default_factory=lambda: SQLiteEmbeddingsCache()
    )
    cache_hits: int = Field(default=0)

    def __init__(
        self, cache_db_path: str = ".embeddings_cache.sqlite", *args, **kwargs
    ):
        """
        Initialize cached OpenAI embeddings.

        Args:
            cache_db_path (str): Path to SQLite cache database
            *args, **kwargs: Passed to OpenAIEmbeddings
        """
        super().__init__(*args, **kwargs)

        # Initialize SQLite cache
        self.cache = SQLiteEmbeddingsCache(cache_db_path, embedding_model="openai")

    def embed_documents(
        self, texts: List[str], use_cache: bool = True
    ) -> List[List[float]]:
        """
        Embed documents with optional SQLite caching.

        Args:
            texts (List[str]): Texts to embed
            use_cache (bool): Whether to use caching

        Returns:
            List[List[float]]: List of embeddings
        """
        embeddings = []

        for text in texts:
            # Try to retrieve from cache
            if use_cache:
                cached_embedding = self.cache.retrieve_embedding(text)
                if cached_embedding:
                    self.cache_hits += 1
                    embeddings.append(cached_embedding)
                    continue

            # Generate new embedding
            embedding = super().embed_documents([text])[0]

            # Store in cache if enabled
            if use_cache:
                self.cache.store_embedding(text, embedding)

            embeddings.append(embedding)

        return embeddings

    def embed_query(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Embed a query with optional SQLite caching.

        Args:
            text (str): Text to embed
            use_cache (bool): Whether to use caching

        Returns:
            List[float]: Embedding
        """
        # Try to retrieve from cache
        if use_cache:
            cached_embedding = self.cache.retrieve_embedding(text)
            if cached_embedding:
                self.cache_hits += 1
                return cached_embedding

        # Generate new embedding
        embedding = super().embed_query(text)

        # Store in cache if enabled
        if use_cache:
            self.cache.store_embedding(text, embedding)

        return embedding


# Example usage
def example_usage():
    # Initialize with custom cache database
    embeddings = CachedOpenAIEmbeddings(
        cache_db_path="./my_embeddings_cache.sqlite",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # First call - will make API request and cache
    doc_embeddings = embeddings.embed_documents(
        ["Hello world", "This is a test document"]
    )

    # Subsequent calls will use cache
    cached_embeddings = embeddings.embed_documents(
        [
            "Hello world",  # Will use cache
            "Another document",  # Will make a new API call
        ]
    )

    # Optional: Clear old cache entries
    embeddings.cache.clear_cache(older_than_days=30)


if __name__ == "__main__":
    example_usage()
