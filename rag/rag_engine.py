"""
Enhanced Agricultural RAG Engine with Semantic Query Expansion,
Context-Aware Sentence Extraction, and Hallucination Suppression.
"""

import os
import re
import json
import math
import pickle
from datetime import datetime
from typing import List, Dict, Tuple, Optional

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except (ImportError, Exception):
    SKLEARN_AVAILABLE = False
    np = None

INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")
INDEX_PATH = os.path.join(INDEX_DIR, "rag_vector_index.pkl")
PDF_DIR = os.path.join(os.path.dirname(__file__), "pdfs")

CHUNK_SIZE_WORDS = 400
CHUNK_OVERLAP_WORDS = 50
SIMILARITY_THRESHOLD = 0.05
TOP_K = 4

# Domain agricultural query expansion dictionary (handles short farmer queries like "tomato", "help with tomatoes")
AGRICULTURAL_SYNONYMS = {
    "tomato": "tomato early blight late blight leaf mold septoria solanum fruit leaves fungicide",
    "tomatoes": "tomato early blight late blight leaf mold septoria solanum fruit leaves fungicide",
    "potato": "potato late blight early blight tuber rot mancozeb metalaxyl solanum",
    "potatoes": "potato late blight early blight tuber rot mancozeb metalaxyl solanum",
    "rice": "rice paddy bacterial leaf blight brown spot leaf smut xanthomonas streak",
    "paddy": "rice paddy bacterial leaf blight brown spot leaf smut xanthomonas",
    "maize": "maize corn gray leaf spot common rust northern corn leaf blight",
    "corn": "maize corn gray leaf spot common rust northern corn leaf blight",
    "cotton": "cotton leaf curl virus bacterial blight whitefly vector boll enation",
    "symptom": "symptoms signs lesions rings spots pustules yellowing wilting water-soaked",
    "symptoms": "symptoms signs lesions rings spots pustules yellowing wilting water-soaked",
    "cure": "management spray fungicide control treatment prevent application",
    "spray": "spray fungicide application mancozeb copper metalaxyl insecticide chemical",
    "water": "irrigation watering overhead sprinkler drip moisture wetness morning evening",
    "irrigation": "irrigation watering overhead sprinkler drip moisture wetness morning evening",
    "animal": "livestock cattle grazing safety interval animals toxic chemical feeding",
    "animals": "livestock cattle grazing safety interval animals toxic chemical feeding",
    "livestock": "livestock cattle grazing safety interval animals toxic chemical feeding"
}

def expand_query(query: str) -> str:
    """Expands farmer queries with domain terms to maximize vector retrieval recall."""
    clean_words = re.findall(r"[a-zA-Z]+", query.lower())
    expansions = []
    for w in clean_words:
        if w in AGRICULTURAL_SYNONYMS:
            expansions.append(AGRICULTURAL_SYNONYMS[w])
    if expansions:
        return query + " " + " ".join(expansions)
    return query


def extract_pdf_pages(file_path: str) -> List[Tuple[int, str]]:
    pages = []
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append((i + 1, text.strip()))
            return pages
        except Exception as e:
            print(f"[RAG Warning] pypdf parsing error on {file_path}: {e}")
    
    if ext in (".txt", ".md"):
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            page_blocks = re.split(r"(?i)\n*Page\s+(\d+)\s*\n*", content)
            if len(page_blocks) > 1:
                for idx in range(1, len(page_blocks), 2):
                    page_num = int(page_blocks[idx])
                    page_text = page_blocks[idx + 1].strip()
                    if page_text:
                        pages.append((page_num, page_text))
            else:
                lines = content.splitlines()
                for i in range(0, max(1, len(lines)), 50):
                    page_text = "\n".join(lines[i:i + 50]).strip()
                    if page_text:
                        pages.append((i // 50 + 1, page_text))
        except Exception as e:
            print(f"[RAG Warning] Text reading error: {e}")

    return pages


def chunk_document_pages(pages: List[Tuple[int, str]], doc_name: str) -> List[Dict]:
    chunks = []
    step = CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS
    
    for page_num, text in pages:
        words = text.split()
        if not words:
            continue
        for i in range(0, max(1, len(words)), step):
            chunk_words = words[i: i + CHUNK_SIZE_WORDS]
            if len(chunk_words) < 15:
                continue
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "doc": doc_name,
                "page": page_num,
                "chunk_id": f"{doc_name}::p{page_num}::w{i}",
                "text": chunk_text
            })
    return chunks


class RAGVectorStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words='english',
            max_df=0.95
        )
        self.chunks: List[Dict] = []
        self.doc_vectors = None
        self.built_at = None
        self.docs_metadata = []

    def fit_and_index(self, chunks: List[Dict], docs_meta: List[Dict]):
        if not chunks:
            return
        self.chunks = chunks
        self.docs_metadata = docs_meta
        corpus = [c["text"] for c in chunks]
        self.doc_vectors = self.vectorizer.fit_transform(corpus)
        self.built_at = datetime.now().isoformat()

    def query(self, raw_query: str, top_k: int = TOP_K) -> List[Dict]:
        if not self.chunks or self.doc_vectors is None:
            return []
        
        expanded_q = expand_query(raw_query)
        q_vec = self.vectorizer.transform([expanded_q])
        sims = cosine_similarity(q_vec, self.doc_vectors)[0]
        
        top_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        
        results = []
        seen_pages = set()
        for idx in top_indices:
            score = float(sims[idx])
            if score < SIMILARITY_THRESHOLD:
                break
            chunk = self.chunks[idx]
            page_key = (chunk["doc"], chunk["page"])
            if page_key in seen_pages:
                continue
            seen_pages.add(page_key)
            
            results.append({
                "score": round(score, 3),
                "doc": chunk["doc"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"]
            })
            if len(results) >= top_k:
                break
                
        return results


def build_rag_index() -> RAGVectorStore:
    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)
    
    all_chunks = []
    docs_meta = []
    
    for fname in sorted(os.listdir(PDF_DIR)):
        fpath = os.path.join(PDF_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        pages = extract_pdf_pages(fpath)
        if not pages:
            continue
        chunks = chunk_document_pages(pages, fname)
        all_chunks.extend(chunks)
        docs_meta.append({"doc": fname, "pages": len(pages), "chunks": len(chunks)})
        
    store = RAGVectorStore()
    if all_chunks:
        store.fit_and_index(all_chunks, docs_meta)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump(store, f)
            
    return store


_store_instance: Optional[RAGVectorStore] = None

def get_rag_store() -> RAGVectorStore:
    global _store_instance
    if _store_instance is None:
        if os.path.exists(INDEX_PATH):
            try:
                with open(INDEX_PATH, "rb") as f:
                    _store_instance = pickle.load(f)
            except Exception:
                _store_instance = build_rag_index()
        else:
            _store_instance = build_rag_index()
    return _store_instance


def synthesize_rag_response(query: str, retrieved_passages: List[Dict]) -> Dict:
    """Context-grounded synthesis with high semantic alignment."""
    if not retrieved_passages:
        return {
            "answer": (
                "The available agricultural literature does not contain verified guidance for this query. "
                "Please consult an agricultural extension officer or upload a specific crop management guide."
            ),
            "sources": [],
            "found": False
        }

    # Extract sentences across the top passages and rank by query match
    all_sentences = []
    for p in retrieved_passages[:2]:
        sents = re.split(r"(?<=[.!?\n])\s+", p["text"])
        for s in sents:
            s_clean = s.strip().replace("\n", " ")
            if len(s_clean) >= 20:
                all_sentences.append((s_clean, p["doc"], p["page"]))

    query_tokens = set(re.findall(r"[a-zA-Z]+", expand_query(query).lower()))
    
    scored_sentences = []
    for s, doc, pg in all_sentences:
        s_tokens = set(re.findall(r"[a-zA-Z]+", s.lower()))
        overlap = len(query_tokens & s_tokens)
        scored_sentences.append((overlap, s))
        
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    
    # Pick top 3 sentences
    top_picks = [s[1] for s in scored_sentences[:3] if s[0] > 0]
    if not top_picks and all_sentences:
        top_picks = [s[0] for s in all_sentences[:3]]
        
    synthesis = " ".join(top_picks).strip()
    
    sources = [
        {"doc": p["doc"], "page": p["page"], "relevance": p["score"]}
        for p in retrieved_passages
    ]
    
    return {
        "answer": synthesis if synthesis else "Verified passage retrieved from indexed agricultural documentation.",
        "sources": sources,
        "found": True
    }


def query_rag(query_text: str, rebuild: bool = False) -> Dict:
    global _store_instance
    if rebuild:
        _store_instance = build_rag_index()
    else:
        _store_instance = get_rag_store()
        
    if not _store_instance.chunks:
        return {
            "answer": "No agricultural PDFs have been indexed yet. Please ensure reference files exist in rag/pdfs/.",
            "sources": [],
            "found": False,
            "total_chunks": 0
        }
        
    retrieved = _store_instance.query(query_text, top_k=TOP_K)
    result = synthesize_rag_response(query_text, retrieved)
    result["total_chunks"] = len(_store_instance.chunks)
    result["retrieved_chunks"] = len(retrieved)
    result["indexed_docs"] = [d["doc"] for d in _store_instance.docs_metadata]
    return result


def get_index_status() -> Dict:
    store = get_rag_store()
    return {
        "built": len(store.chunks) > 0,
        "built_at": store.built_at,
        "docs": store.docs_metadata,
        "total_chunks": len(store.chunks)
    }

def rebuild_index() -> Dict:
    global _store_instance
    _store_instance = build_rag_index()
    return get_index_status()
