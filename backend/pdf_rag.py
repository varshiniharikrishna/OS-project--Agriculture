"""
PDF Knowledge Retrieval / Retrieval-Augmented Generation (RAG) Engine.
Parses agriculture1.pdf - agriculture4.pdf, indexes chunks, and performs semantic keyword retrieval.
"""

import os
import sys
import re
import math
from collections import Counter

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.config import AGRICULTURE_PDFS


class AgriculturalPDFRAG:
    """PDF Chunking and Semantic Keyword Retrieval Engine."""

    def __init__(self, pdf_paths=AGRICULTURE_PDFS):
        self.pdf_paths = pdf_paths
        self.chunks = []
        self.index_built = False
        self._build_index()

    def _extract_text_from_pdf(self, pdf_path):
        """Extract text page-by-page from a PDF file."""
        pages = []
        filename = os.path.basename(pdf_path)

        if not os.path.exists(pdf_path):
            return pages

        # Try pypdf library if available
        try:
            import pypdf
            reader = pypdf.PdfReader(pdf_path)
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({"doc": filename, "page": page_idx + 1, "text": text})
            if pages:
                return pages
        except Exception as e:
            pass

        # Fallback pure-python basic stream text extractor
        try:
            with open(pdf_path, "rb") as f:
                content = f.read().decode("latin-1", errors="ignore")
                # Extract text blocks between BT and ET
                text_blocks = re.findall(r"BT(.*?)ET", content, re.DOTALL)
                raw_text = ""
                for block in text_blocks:
                    strings = re.findall(r"\((.*?)\)", block)
                    raw_text += " ".join(strings) + "\n"
                
                if raw_text.strip():
                    pages.append({"doc": filename, "page": 1, "text": raw_text})
        except Exception as e:
            print(f"Notice: Fallback stream reader on {filename} ({e})")

        return pages

    def _chunk_text(self, text, chunk_size=400, overlap=50):
        """Split page text into overlapping word chunks."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.strip()) > 30:
                chunks.append(chunk)
        return chunks

    def _build_index(self):
        """Load and index all PDF documents."""
        self.chunks = []
        for pdf_path in self.pdf_paths:
            pages = self._extract_text_from_pdf(pdf_path)
            for p in pages:
                text_chunks = self._chunk_text(p["text"])
                for c in text_chunks:
                    self.chunks.append({
                        "doc": p["doc"],
                        "page": p["page"],
                        "text": c,
                        "tokens": set(re.findall(r"\w+", c.lower()))
                    })

        self.index_built = True

    def query(self, query_str, top_k=3, score_threshold=0.12):
        """Perform TF-IDF / Term overlap retrieval over indexed PDF chunks."""
        if not self.chunks:
            self._build_index()

        if not self.chunks:
            return {
                "answer": "I could not find enough information in the available agricultural sources to answer this reliably.",
                "sources": [],
                "retrieved": False
            }

        query_tokens = re.findall(r"\w+", query_str.lower())
        query_tokens = [t for t in query_tokens if len(t) > 2 and t not in {"what", "how", "why", "when", "does", "should", "this", "that", "the", "and", "for", "with"}]

        if not query_tokens:
            query_tokens = re.findall(r"\w+", query_str.lower())

        scored_chunks = []
        for chunk in self.chunks:
            # Calculate term overlap and term frequency
            match_count = sum(1 for token in query_tokens if token in chunk["tokens"])
            if match_count == 0:
                continue

            score = match_count / (math.sqrt(len(query_tokens)) * math.sqrt(len(chunk["tokens"]) + 1))
            scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        if not scored_chunks or scored_chunks[0][0] < score_threshold:
            return {
                "answer": "I could not find enough information in the available agricultural sources (agriculture1.pdf - agriculture4.pdf) to answer this reliably.",
                "sources": [],
                "retrieved": False
            }

        top_matches = scored_chunks[:top_k]
        primary_chunk = top_matches[0][1]

        # Extract answer passage from top matching chunk
        answer_text = primary_chunk["text"][:600] + "..." if len(primary_chunk["text"]) > 600 else primary_chunk["text"]

        sources = []
        seen = set()
        for score, c in top_matches:
            source_id = f"{c['doc']} (Page {c['page']})"
            if source_id not in seen:
                sources.append({
                    "doc": c["doc"],
                    "page": c["page"],
                    "relevance_score": round(score, 3),
                    "source_label": source_id
                })
                seen.add(source_id)

        return {
            "answer": answer_text,
            "sources": sources,
            "retrieved": True,
            "top_source": sources[0]["source_label"] if sources else None
        }


# Singleton RAG Instance
rag_engine = AgriculturalPDFRAG()

def retrieve_agricultural_knowledge(user_query):
    """Public helper function to execute RAG over agricultural PDFs."""
    return rag_engine.query(user_query)


if __name__ == "__main__":
    test_q = "What should I do if my crop has blight?"
    print(f"Testing Query: '{test_q}'")
    res = retrieve_agricultural_knowledge(test_q)
    print("Answer:", res["answer"])
    print("Sources:", res["sources"])
