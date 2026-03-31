import chromadb
import numpy as np
from typing import List, Any, Dict
import os
from pathlib import Path
import uuid

class VectorStore:
    def __init__(self, collection_name: str="pdf_documents", persist_directory: str=None):
        if persist_directory is None:
            # Keep persistence anchored to project root regardless of cwd.
            persist_directory = str(Path(__file__).resolve().parent.parent / "data" / "vector_store")
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name = collection_name,
            metadata = {"hnsw:space":"cosine"}
        )
        print(f"[INFO] Vector store ready at {persist_directory} with collection '{collection_name}'")

    def clear_collection(self):
        """Delete and recreate the collection to avoid stale retrieval results."""
        collection_name = self.collection.name
        try:
            self.client.delete_collection(name=collection_name)
        except Exception as e:
            print(f"[WARNING] Could not delete collection '{collection_name}': {e}")

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"[INFO] Collection '{collection_name}' reset.")

    def build_from_documents(self, chunks: List[Any], embeddings: np.ndarray):
        """
        Saves documents and embeddings to ChromaDB
        """
        # Prepare Data
        documents = [chunk.page_content for chunk in chunks]
        
        # Clean Metadata (Chroma only accepts strings, ints, floats, bools)
        metadatas = []
        for chunk in chunks:
            # Simple list comprehension to ensure metadata is safe
            clean_meta = {k: str(v) for k, v in chunk.metadata.items()}
            metadatas.append(clean_meta)

        # Convert numpy array to list
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        ids = [f"chunk_{uuid.uuid4().hex}_{i}" for i in range(len(documents))]


        # Add to DB
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids = ids
        )

    def query(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict]:
        """
        Search for relevant documents.
        """
        # Ensure query is a list
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Clean up results for the app
        parsed_results = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                parsed_results.append({
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "score": results['distances'][0][i]
                })
        
        return parsed_results
    
