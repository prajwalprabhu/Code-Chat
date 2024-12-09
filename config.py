# Get config values
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# os.environ["GROQ_API_KEY"] = GROQ_API_KEY
SECRET_KEY = os.getenv(
    "SECRET_KEY", "12df3cv45sge5wer7teu7ew73uj47463672yh7e6y3hdjjdkdjdhduw8w7eh"
)
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "embeddings")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
