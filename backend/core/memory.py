import os

import lancedb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


class KnowledgeManager:
    """
    AEGIS Perception: Permanent Memory (LanceDB).
    Handles document ingestion, vector embedding, and RAG retrieval.
    """

    def __init__(self):
        # 1. Initialize the ultra-lightweight embedding model
        print("Loading Embedding Model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

        # 2. Connect to the embedded local database
        db_path = os.path.join(os.getcwd(), "aegis_data")
        self.db = lancedb.connect(db_path)
        self.table_name = "documents"

    def _chunk_text(self, text: str, chunk_size: int = 500) -> list[str]:
        """Splits large text into smaller chunks to fit Gemma's context window."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i : i + chunk_size]))  # type: ignore
        return chunks  # type: ignore

    def ingest_pdf(self, file_path: str, filename: str) -> str:
        """Extracts text from a PDF, embeds it, and saves it to LanceDB."""
        try:
            # Extract Text
            reader = PdfReader(file_path)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() + "\n"

            if not full_text.strip():
                return "Error: PDF is empty or purely image-based (Needs OCR)."

            # Chunk and Embed
            chunks = self._chunk_text(full_text)
            data = []
            for i, chunk in enumerate(chunks):
                vector = self.embedder.encode(chunk).tolist()  # type: ignore
                data.append(  # type: ignore
                    {"vector": vector, "text": chunk, "source": filename, "chunk_id": i}
                )

            # Save to Database
            if self.table_name in self.db.table_names():
                tbl = self.db.open_table(self.table_name)
                tbl.add(data)  # type: ignore
            else:
                self.db.create_table(self.table_name, data=data)  # type: ignore

            return f"Successfully memorized {len(chunks)} chunks from {filename}."

        except Exception as e:
            return f"Memory Ingestion Failed: {str(e)}"

    def search(self, query: str, limit: int = 3) -> str:
        """Embeds the query and retrieves the top matching document chunks."""
        try:
            if self.table_name not in self.db.table_names():
                return ""  # Database is empty

            # 1. Convert the user's text question into a math vector
            query_vector = self.embedder.encode(query).tolist()  # type: ignore

            # 2. Perform a Lightning-Fast Vector Search
            tbl = self.db.open_table(self.table_name)
            results = tbl.search(query_vector).limit(limit).to_list()  # type: ignore

            if not results:
                return ""

            # 3. Format the results so the LLM can read them easily
            context = "\n\n".join(
                [f"[Source: {res['source']}]\n{res['text']}" for res in results]  # type: ignore
            )
            return context

        except Exception as e:
            print(f"Memory Search Error: {e}")
            return ""


# Singleton instance
aegis_memory = KnowledgeManager()
