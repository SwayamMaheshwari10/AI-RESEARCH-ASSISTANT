from src.embedding import EmbeddingPipeline
from src.vectorstore import VectorStore
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
import re
import json
from pathlib import Path

load_dotenv() 


class RAGSearch:
    def __init__(
        self,
        persist_directory: str = None,
        collection_name: str = "pdf_documents",
        llm_model: str = "llama-3.1-8b-instant"
    ):
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError(
        "GROQ_API_KEY not found. Check your .env file."
        )
        
        # Initialize embedding pipeline
        self.embedding_pipeline = EmbeddingPipeline()

        # Initialize vector store
        self.vectorstore = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory
        )

        # Initialize Groq via LangChain
        self.llm = ChatGroq(
            model=llm_model,
            groq_api_key=groq_api_key,
            temperature=0.1,
        )

        print("[INFO] RAGSearch initialized (LangChain + Groq)")

    def search_and_summarize(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve relevant chunks and generate an answer using RAG.
        """

        # 1. Embed query
        query_embedding = self.embedding_pipeline.model.encode(query)

        # 2. Retrieve relevant chunks
        results = self.vectorstore.query(
            query_embedding=query_embedding,
            top_k=top_k
        )

        if not results:
            return "No relevant information found in the uploaded documents."

        # 3. Build context from retrieved chunks
        context = "\n\n".join([r["text"] for r in results])

        # 4. Build prompt
        prompt = f"""
                    You are an AI research assistant.

                    Answer the question using ONLY the context below.
                    If the answer is not present in the context, say so clearly.

                    Context:
                    {context}

                    Question:
                    {query}

                    Answer:
                    """

        # 5. Call LLM
        response = self.llm.invoke([HumanMessage(content=prompt)])
        answer = response.content.strip()
    
         # 6. Build citations from metadata
        citations = []
        seen = set()

        for i, r in enumerate(results, start=1):
            meta = r.get("metadata", {})
            source = meta.get("source", "Unknown source")
            source_name = Path(source).name if source else "Unknown source"
            page = meta.get("page", "N/A")

            citation_key = f"{source_name}-{page}"
            if citation_key not in seen:
                seen.add(citation_key)
                citations.append(f"[{len(citations)+1}] {source_name} - page {page}")

        # 7. Append citations
        if citations:
            answer += "\n\nSources:\n" + "\n".join(citations)

        return answer

        
        