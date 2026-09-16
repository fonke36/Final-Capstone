from pathlib import Path
import re

import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

KB_FOLDER = Path("knowledge_base")

# Fixed-size chunk settings
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# ---------------------------------------------------------
# Embedding and ChromaDB configuration
# ---------------------------------------------------------

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHROMA_PATH = Path("chroma_db")

FIXED_COLLECTION_NAME = "ola_support_fixed_chunks"
SENTENCE_COLLECTION_NAME = "ola_support_sentence_chunks"


# ---------------------------------------------------------
# 1. Load all knowledge-base documents
# ---------------------------------------------------------

def load_documents():
    """
    Read all .txt files from the knowledge_base folder.

    Returns:
        A list of dictionaries containing document ID,
        filename, and text.
    """

    documents = []

    for file_path in sorted(KB_FOLDER.glob("*.txt")):

        text = file_path.read_text(
            encoding="utf-8"
        ).strip()

        documents.append({
            "doc_id": file_path.stem,
            "filename": file_path.name,
            "text": text,
        })

    return documents


# ---------------------------------------------------------
# 2. Fixed-size chunking with overlap
# ---------------------------------------------------------

def fixed_size_chunks(text, chunk_size=CHUNK_SIZE,
                       overlap=CHUNK_OVERLAP):
    """
    Split text into fixed-size character chunks.

    Example:

        Chunk 1: characters 0–300
        Chunk 2: characters 250–550
        Chunk 3: characters 500–800

    The 50-character overlap helps preserve context
    between neighboring chunks.
    """

    if overlap >= chunk_size:
        raise ValueError(
            "Overlap must be smaller than chunk size."
        )

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ---------------------------------------------------------
# 3. Sentence-based chunking
# ---------------------------------------------------------

def sentence_based_chunks(text, sentences_per_chunk=2):
    """
    Split the document into groups of complete sentences.

    Each chunk contains up to the specified number
    of sentences.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    chunks = []

    for i in range(
        0,
        len(sentences),
        sentences_per_chunk
    ):

        chunk = " ".join(
            sentences[i:i + sentences_per_chunk]
        )

        if chunk:
            chunks.append(chunk)

    return chunks


# ---------------------------------------------------------
# 4. Create both chunking strategies
# ---------------------------------------------------------

def create_chunks():

    documents = load_documents()

    fixed_chunks = []
    sentence_chunks = []

    for document in documents:

        # Fixed-size strategy
        chunks = fixed_size_chunks(
            document["text"]
        )

        for index, chunk in enumerate(chunks):

            fixed_chunks.append({
                "chunk_id": (
                    f"{document['doc_id']}"
                    f"_fixed_{index}"
                ),
                "doc_id": document["doc_id"],
                "text": chunk,
            })

        # Sentence-based strategy
        chunks = sentence_based_chunks(
            document["text"]
        )

        for index, chunk in enumerate(chunks):

            sentence_chunks.append({
                "chunk_id": (
                    f"{document['doc_id']}"
                    f"_sentence_{index}"
                ),
                "doc_id": document["doc_id"],
                "text": chunk,
            })

    return fixed_chunks, sentence_chunks


# ---------------------------------------------------------
# 5. Create embeddings and store chunks in ChromaDB
# ---------------------------------------------------------

def index_chunks(fixed_chunks, sentence_chunks):
    """
    Embed all chunks locally and store them in two
    separate ChromaDB collections.
    """

    print("\nLoading local embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print("Embedding model loaded.")

    # Create a persistent local ChromaDB client.
    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    # Separate collection for fixed-size chunks.
    fixed_collection = chroma_client.get_or_create_collection(
        name=FIXED_COLLECTION_NAME
    )

    # Separate collection for sentence-based chunks.
    sentence_collection = chroma_client.get_or_create_collection(
        name=SENTENCE_COLLECTION_NAME
    )

    # -----------------------------------------------------
    # Fixed-size chunks
    # -----------------------------------------------------

    fixed_texts = [
        chunk["text"]
        for chunk in fixed_chunks
    ]

    fixed_embeddings = model.encode(
        fixed_texts,
        normalize_embeddings=True
    ).tolist()

    fixed_collection.upsert(
        ids=[
            chunk["chunk_id"]
            for chunk in fixed_chunks
        ],
        embeddings=fixed_embeddings,
        documents=fixed_texts,
        metadatas=[
            {
                "doc_id": chunk["doc_id"],
                "chunking_strategy": "fixed_size",
            }
            for chunk in fixed_chunks
        ],
    )

    # -----------------------------------------------------
    # Sentence-based chunks
    # -----------------------------------------------------

    sentence_texts = [
        chunk["text"]
        for chunk in sentence_chunks
    ]

    sentence_embeddings = model.encode(
        sentence_texts,
        normalize_embeddings=True
    ).tolist()

    sentence_collection.upsert(
        ids=[
            chunk["chunk_id"]
            for chunk in sentence_chunks
        ],
        embeddings=sentence_embeddings,
        documents=sentence_texts,
        metadatas=[
            {
                "doc_id": chunk["doc_id"],
                "chunking_strategy": "sentence_based",
            }
            for chunk in sentence_chunks
        ],
    )

    print("\nChromaDB indexing complete.")

    print(
        f"Fixed-size collection: "
        f"{fixed_collection.count()} chunks"
    )

    print(
        f"Sentence-based collection: "
        f"{sentence_collection.count()} chunks"
    )

    return (
        model,
        fixed_collection,
        sentence_collection,
    )

# ---------------------------------------------------------
# 6. Run the complete indexing test
# ---------------------------------------------------------

if __name__ == "__main__":

    documents = load_documents()

    fixed_chunks, sentence_chunks = create_chunks()

    print("=" * 60)
    print("OLA SUPPORT RAG INDEXING")
    print("=" * 60)

    print(f"\nDocuments loaded: {len(documents)}")

    print(
        f"Fixed-size chunks: "
        f"{len(fixed_chunks)}"
    )

    print(
        f"Sentence-based chunks: "
        f"{len(sentence_chunks)}"
    )

    model, fixed_collection, sentence_collection = (
        index_chunks(
            fixed_chunks,
            sentence_chunks
        )
    )

    print("\n" + "=" * 60)
    print("INDEXING TEST PASSED")
    print("=" * 60)