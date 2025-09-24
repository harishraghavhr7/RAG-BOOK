"""
Demo RAG System - Works without external model downloads
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

class MistralRAG:
    """Demo RAG system that works without external dependencies"""
    
    def __init__(self):
        """Initialize demo RAG system"""
        logger.info("Initializing DEMO RAG system...")
        
        # Demo data storage
        self.documents = []
        self.chunks = []
        self.chunk_metadata = []
        
        # Simple demo responses
        self.demo_responses = [
            "Based on the documents, this is a sample response from the RAG system.",
            "This is a demo response. In the full version, I would analyze your documents to provide accurate answers.",
            "Demo mode: I can see you're asking about this topic. The full RAG system would search through your documents for relevant information.",
            "This is a demonstration response. When properly configured, I would provide detailed answers based on your document content."
        ]
        
        logger.info("✅ Demo RAG system initialized successfully")
    
    def add_document_from_file(self, file_path: str):
        """Add a document from file (demo version)"""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                return
                
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Simple chunking for demo
            chunks = self._simple_chunk(content)
            
            doc_info = {
                'filename': os.path.basename(file_path),
                'path': file_path,
                'size': len(content),
                'chunks': len(chunks)
            }
            
            self.documents.append(doc_info)
            self.chunks.extend(chunks)
            
            # Add metadata for each chunk
            for i, chunk in enumerate(chunks):
                self.chunk_metadata.append({
                    'doc_index': len(self.documents) - 1,
                    'chunk_index': i,
                    'filename': doc_info['filename']
                })
            
            logger.info(f"✅ Loaded document: {doc_info['filename']} ({len(chunks)} chunks)")
            
        except Exception as e:
            logger.error(f"❌ Error loading document {file_path}: {e}")
    
    def _simple_chunk(self, text: str, chunk_size: int = 500) -> List[str]:
        """Simple text chunking"""
        if not text.strip():
            return []
        
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                current_chunk += ("\n\n" + paragraph) if current_chunk else paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def search_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Simple keyword-based search for demo"""
        if not self.chunks:
            return []
        
        query_words = set(query.lower().split())
        results = []
        
        for i, chunk in enumerate(self.chunks):
            chunk_words = set(chunk.lower().split())
            # Simple word overlap scoring
            overlap = len(query_words.intersection(chunk_words))
            
            if overlap > 0:
                score = overlap / len(query_words)
                results.append({
                    'chunk': chunk,
                    'score': score,
                    'metadata': self.chunk_metadata[i],
                    'index': i
                })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Answer a question using demo logic"""
        # Search for relevant chunks
        search_results = self.search_documents(question, top_k=3)
        
        if search_results:
            # Use the first demo response with context
            context_chunks = [r['chunk'] for r in search_results]
            response = f"{self.demo_responses[0]}\n\nRelevant context from your documents:\n\n"
            response += "\n\n---\n\n".join(context_chunks[:2])  # Limit context
            
            sources = [r['metadata']['filename'] for r in search_results]
        else:
            # Fallback response
            response = "I don't have specific information about that in the loaded documents. This is a demo response."
            sources = []
        
        return {
            'response': response,
            'sources': list(set(sources)),  # Remove duplicates
            'search_results_count': len(search_results),
            'mode': 'demo'
        }
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded documents"""
        return {
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks),
            'documents': self.documents,
            'mode': 'demo'
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for the RAG system"""
        return {
            'status': 'healthy',
            'mode': 'demo',
            'documents_loaded': len(self.documents),
            'chunks_available': len(self.chunks),
            'ready': True
        }