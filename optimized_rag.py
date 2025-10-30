import os
import re
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Configure logging early so we can use logger during module import
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Handle FAISS import with fallback
FAISS_AVAILABLE = False
try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS loaded successfully")
except ImportError:
    try:
        import faiss_cpu as faiss
        FAISS_AVAILABLE = True
        logger.info("FAISS-CPU loaded successfully")
    except ImportError:
        logger.warning("FAISS not available. Attempting to install faiss-cpu...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
        try:
            import faiss
            FAISS_AVAILABLE = True
            logger.info("FAISS installed and loaded successfully")
        except ImportError:
            logger.warning("Failed to install FAISS. Using sklearn as fallback.")
            FAISS_AVAILABLE = False

# Fallback imports if FAISS is not available
if not FAISS_AVAILABLE:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer

# Import Gemini client
try:
    from gemini_client import GeminiConversation, query_gemini
    GEMINI_AVAILABLE = True
    logger.info("Gemini client loaded successfully")
except ImportError:
    logger.warning("Gemini client not available")
    GEMINI_AVAILABLE = False

from typing import List, Dict, Tuple, Any

class MistralRAG:
    def __init__(self):
        """Initialize RAG system with Ollama Mistral integration"""
        try:
            # Try to load the sentence transformer with a timeout approach
            import signal
            import os
            
            # Allow overriding the embedding model with an environment variable
            model_name = os.getenv('EMBEDDING_MODEL', 'all-mpnet-base-v2')
            logger.info(f"Loading sentence transformer model ({model_name})...")
            try:
                self.embeddings_model = SentenceTransformer(model_name)
                logger.info(f"✅ Sentence transformer loaded successfully ({model_name})")
            except Exception as e:
                logger.error(f"❌ Failed to load {model_name}: {e}")
                # Fallback to a smaller model that is faster to download
                fallback = 'all-MiniLM-L6-v2'
                logger.info(f"Falling back to smaller embedding model: {fallback}")
                self.embeddings_model = SentenceTransformer(fallback)
                logger.info(f"✅ Sentence transformer loaded successfully ({fallback})")
        except Exception as e:
            logger.error(f"❌ Failed to load sentence transformer: {e}")
            logger.info("💡 This might be due to network issues downloading the model.")
            logger.info("💡 Please ensure you have internet connectivity or try again later.")
            raise Exception("Failed to initialize embeddings model") from e
            
        self.chunks = []
        self.chunk_metadata = []
        self.index = None
        self.embeddings_matrix = None
        self.use_faiss = FAISS_AVAILABLE
        self.gemini_conversation = None
        self.setup_gemini()
    
    def setup_gemini(self):
        """Setup Gemini conversation client"""
        if GEMINI_AVAILABLE:
            try:
                # Quick connection test with short timeout
                from gemini_client import test_gemini_connection
                if test_gemini_connection():
                    self.gemini_conversation = GeminiConversation()
                    logger.info("✅ Gemini conversation client initialized")
                    # Skip the actual test call during initialization to avoid hanging
                    logger.info("✅ Gemini connection successful")
                else:
                    logger.warning("⚠️ Gemini connection test failed, will retry during first query")
                    self.gemini_conversation = None
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to setup Gemini (will use fallback): {e}")
                self.gemini_conversation = None
        else:
            logger.info("💡 Gemini not available, will use keyword extraction only")

    def smart_chunk_text(self, text: str, chunk_size: int = 1200, overlap: int = 300) -> List[Dict]:
        """Enhanced smart chunking that preserves educational content structure with concepts, definitions, and examples"""
        chunks = []
        
        # Updated patterns to match the actual text structure
        chapter_patterns = [
            r'CC\s+PP\s+MM.*?(?=CC\s+PP\s+MM|\Z)',  # CC PP MM pattern
            r'MM\s*::\s*MM.*?(?=MM\s*::\s*MM|\Z)',   # MM :: MM pattern  
            r'SS\s+FF\s+PP.*?(?=SS\s+FF\s+PP|\Z)',   # SS FF PP pattern
            r'CC\s+--\s+SS.*?(?=CC\s+--\s+SS|\Z)',   # CC -- SS pattern
            r'RR\s+AA.*?(?=RR\s+AA|\Z)',             # RR AA pattern
            r'CC\s+FF.*?(?=CC\s+FF|\Z)',             # CC FF pattern
        ]
        
        # Try to find chapters using different patterns
        found_chapters = []
        for pattern in chapter_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                found_chapters.extend(matches)
                break
        
        if not found_chapters:
            # Fallback: split by major headings or numbered sections
            heading_patterns = [
                r'\d+\.\d+\s+[A-Z][^.]*?(?=\d+\.\d+\s+[A-Z]|\Z)',
                r'[A-Z][A-Z\s]{10,}.*?(?=[A-Z][A-Z\s]{10,}|\Z)',
                r'Activity\s+\d+\.\d+.*?(?=Activity\s+\d+\.\d+|\Z)',
            ]
            
            for pattern in heading_patterns:
                matches = re.findall(pattern, text, re.DOTALL)
                if matches and len(matches) > 3:
                    found_chapters = matches
                    break
        
        if not found_chapters:
            # Final fallback: simple text splitting
            logger.info("No chapter structure found, using simple text splitting")
            return self.simple_chunk_text(text, chunk_size, overlap)
        
        # Process found chapters/sections
        for i, chapter_text in enumerate(found_chapters):
            # Extract chapter title
            lines = chapter_text.strip().split('\n')
            chapter_title = lines[0][:100] if lines else f"Section {i+1}"
            
            # Clean the text
            clean_text = re.sub(r'\s+', ' ', chapter_text).strip()
            
            # If chapter is small enough, keep as one chunk
            if len(clean_text) <= chunk_size * 1.5:
                chunks.append({
                    'text': clean_text,
                    'chapter': chapter_title,
                    'section': 'Complete Section',
                    'type': 'complete_section'
                })
            else:
                # Enhanced chunking to preserve educational elements
                educational_chunks = self.create_educational_chunks(clean_text, chapter_title, chunk_size)
                chunks.extend(educational_chunks)
        
        # If still no chunks, use simple splitting
        if not chunks:
            logger.info("Chapter parsing failed, using simple chunking")
            return self.simple_chunk_text(text, chunk_size, overlap)
        
        logger.info(f"✅ Created {len(chunks)} smart chunks from {len(found_chapters)} sections")
        return chunks
    
    def create_educational_chunks(self, text: str, chapter_title: str, max_chunk_size: int = 1200) -> List[Dict]:
        """Create educational chunks that preserve concepts, definitions, examples, and explanations"""
        chunks = []
        
        # Educational patterns to identify key content types
        definition_patterns = [
            r'(.*?(?:is defined as|means|refers to|is called|is known as).*?)(?=\.|$)',
            r'(\b[A-Z][a-z]+\b.*?(?:definition|meaning).*?)(?=\.|$)',
        ]
        
        example_patterns = [
            r'((?:For example|Example|For instance|Such as).*?)(?=\.|$)',
            r'(.*?(?:includes|like|such as).*?)(?=\.|$)',
        ]
        
        concept_patterns = [
            r'(\d+\.\d+\s+[A-Z][^.]*?)(?=\d+\.\d+|\.|$)',
            r'([A-Z][A-Z\s]{5,}[^.]*?)(?=[A-Z][A-Z\s]{5,}|\.|$)',
        ]
        
        # Split text into sentences first
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        current_chunk = ""
        chunk_count = 1
        chunk_metadata = {
            'has_definition': False,
            'has_example': False,
            'concept_type': 'general'
        }
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if this sentence contains important educational elements
            is_definition = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in definition_patterns)
            is_example = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in example_patterns)
            is_concept = any(re.search(pattern, sentence) for pattern in concept_patterns)
            
            # If adding this sentence would exceed chunk size, save current chunk
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'chapter': chapter_title,
                    'section': f'Part {chunk_count}',
                    'type': 'educational_chunk',
                    'part': str(chunk_count),
                    'metadata': chunk_metadata.copy()
                })
                chunk_count += 1
                current_chunk = ""
                chunk_metadata = {
                    'has_definition': False,
                    'has_example': False,
                    'concept_type': 'general'
                }
            
            # Add sentence to current chunk
            current_chunk += sentence + " "
            
            # Update metadata
            if is_definition:
                chunk_metadata['has_definition'] = True
                chunk_metadata['concept_type'] = 'definition'
            if is_example:
                chunk_metadata['has_example'] = True
            if is_concept and chunk_metadata['concept_type'] == 'general':
                chunk_metadata['concept_type'] = 'concept'
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'chapter': chapter_title,
                'section': f'Part {chunk_count}',
                'type': 'educational_chunk',
                'part': str(chunk_count),
                'metadata': chunk_metadata
            })
        
        return chunks
    
    def simple_chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 150) -> List[Dict]:
        """Improved simple text chunking with better sentence boundary detection"""
        chunks = []
        
        # First, clean and normalize the text
        text = re.sub(r'\n\s*\n+', '\n\n', text)  # Normalize paragraph breaks
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Split into meaningful sections based on content structure
        sections = []
        
        # Look for numbered sections first
        section_splits = re.split(r'(\d+\.\d+\s+[A-Z][^\n]*)', text)
        if len(section_splits) > 3:
            current_section = ""
            section_title = "Introduction"
            
            for i, part in enumerate(section_splits):
                if re.match(r'\d+\.\d+\s+[A-Z]', part):
                    if current_section.strip():
                        sections.append((section_title, current_section.strip()))
                    section_title = part.strip()
                    current_section = ""
                else:
                    current_section += part
            
            # Add the last section
            if current_section.strip():
                sections.append((section_title, current_section.strip()))
        else:
            # Fallback: split by paragraphs
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 50]
            sections = [(f"Section {i+1}", para) for i, para in enumerate(paragraphs)]
        
        # Now chunk each section appropriately
        chunk_count = 1
        for section_title, section_content in sections:
            if len(section_content) <= chunk_size:
                # Section fits in one chunk
                chunks.append({
                    'text': section_content,
                    'chapter': section_title[:80],
                    'section': 'Complete',
                    'type': 'complete_section'
                })
            else:
                # Split large sections by sentences
                sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', section_content)
                current_chunk = ""
                part_num = 1
                
                for sentence in sentences:
                    # Check if adding this sentence would exceed chunk size
                    if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                        chunks.append({
                            'text': current_chunk.strip(),
                            'chapter': section_title[:80],
                            'section': f'Part {part_num}',
                            'type': 'partial_section'
                        })
                        part_num += 1
                        
                        # Start new chunk with some overlap
                        last_sentences = current_chunk.split('. ')[-2:]
                        current_chunk = '. '.join(last_sentences) + '. ' + sentence
                    else:
                        current_chunk += sentence + ' '
                
                # Add remaining content
                if current_chunk.strip():
                    chunks.append({
                        'text': current_chunk.strip(),
                        'chapter': section_title[:80],
                        'section': f'Part {part_num}',
                        'type': 'partial_section'
                    })
        
        # Filter out very short chunks
        chunks = [chunk for chunk in chunks if len(chunk['text']) > 100]
        
        logger.info(f"✅ Created {len(chunks)} improved chunks")
        return chunks

    def load_and_process_text(self, file_path: str) -> bool:
        """Load and process the educational text"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Smart chunking
            chunk_data = self.smart_chunk_text(text)
            
            self.chunks = [chunk['text'] for chunk in chunk_data]
            self.chunk_metadata = chunk_data
            
            logger.info(f"✅ Processed {len(self.chunks)} chunks from educational content")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing text: {e}")
            return False
    
    def create_embeddings(self) -> bool:
        """Create optimized embeddings for educational content"""
        try:
            if not self.chunks:
                logger.error("No chunks to embed")
                return False
            
            # Create embeddings with educational context
            enhanced_chunks = []
            for i, chunk in enumerate(self.chunks):
                metadata = self.chunk_metadata[i] if i < len(self.chunk_metadata) else {}
                
                # Safely get chapter and section with defaults
                chapter = metadata.get('chapter', f'Chapter {(i // 50) + 1}')
                section = metadata.get('section', f'Part {(i % 50) + 1}')
                
                # Add context to help with retrieval
                enhanced_text = f"Chapter: {chapter} | Section: {section} | Content: {chunk}"
                enhanced_chunks.append(enhanced_text)
            
            embeddings = self.embeddings_model.encode(enhanced_chunks, show_progress_bar=True)
            
            if self.use_faiss:
                # Create FAISS index
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                
                # Normalize for cosine similarity
                faiss.normalize_L2(embeddings)
                self.index.add(embeddings.astype('float32'))
                logger.info(f"✅ Created FAISS index with {len(self.chunks)} vectors")
            else:
                # Fallback: store embeddings matrix for sklearn similarity
                self.embeddings_matrix = embeddings
                logger.info(f"✅ Created embeddings matrix with {len(self.chunks)} vectors (sklearn fallback)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating embeddings: {e}")
            return False

    def rebuild_embeddings(self) -> bool:
        """Helper to rebuild embeddings from current chunks (use after switching embedding model)

        This clears the current index/matrix and re-creates embeddings using the active
        SentenceTransformer model. Call this after changing EMBEDDING_MODEL environment variable.
        """
        logger.info("Rebuilding embeddings and index...")
        # Clear existing index and embeddings
        self.index = None
        self.embeddings_matrix = None
        try:
            return self.create_embeddings()
        except Exception as e:
            logger.error(f"Failed to rebuild embeddings: {e}")
            return False
    
    def retrieve_relevant_chunks(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve most relevant chunks with enhanced scoring"""
        try:
            # Enhance query for better retrieval
            enhanced_query = f"Question about: {query}"
            query_embedding = self.embeddings_model.encode([enhanced_query])
            
            if self.use_faiss and self.index is not None:
                # Use FAISS for similarity search
                faiss.normalize_L2(query_embedding)
                scores, indices = self.index.search(query_embedding.astype('float32'), min(k*2, len(self.chunks)))
                
                relevant_chunks = []
                seen_sections = set()
                
                for i, idx in enumerate(indices[0]):
                    if idx < len(self.chunks):
                        metadata = self.chunk_metadata[idx] if idx < len(self.chunk_metadata) else {}
                        
                        # Safely get chapter and section with defaults
                        chapter = metadata.get('chapter', f'Chapter {(idx // 50) + 1}')
                        section = metadata.get('section', f'Part {(idx % 50) + 1}')
                        section_key = f"{chapter}_{section}"
                        
                        # Prefer complete sections and avoid duplicates
                        score = float(scores[0][i])
                        chunk_type = metadata.get('type', 'unknown')
                        if chunk_type == 'complete_section':
                            score += 0.1  # Boost complete sections
                        
                        if section_key not in seen_sections or len(relevant_chunks) < k//2:
                            relevant_chunks.append({
                                'text': self.chunks[idx],
                                'score': score,
                                'metadata': metadata
                            })
                            seen_sections.add(section_key)
                        
                        if len(relevant_chunks) >= k:
                            break
            
            elif self.embeddings_matrix is not None:
                # Fallback: use sklearn cosine similarity
                similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]
                
                # Get top k indices
                top_indices = np.argsort(similarities)[::-1][:k*2]
                
                relevant_chunks = []
                seen_sections = set()
                
                for idx in top_indices:
                    if idx < len(self.chunks):
                        metadata = self.chunk_metadata[idx] if idx < len(self.chunk_metadata) else {}
                        
                        # Safely get chapter and section with defaults
                        chapter = metadata.get('chapter', f'Chapter {(idx // 50) + 1}')
                        section = metadata.get('section', f'Part {(idx % 50) + 1}')
                        section_key = f"{chapter}_{section}"
                        
                        score = float(similarities[idx])
                        chunk_type = metadata.get('type', 'unknown')
                        if chunk_type == 'complete_section':
                            score += 0.1
                        
                        if section_key not in seen_sections or len(relevant_chunks) < k//2:
                            relevant_chunks.append({
                                'text': self.chunks[idx],
                                'score': score,
                                'metadata': metadata
                            })
                            seen_sections.add(section_key)
                        
                        if len(relevant_chunks) >= k:
                            break
            
            else:
                logger.error("No similarity search method available")
                return []
            
            # Sort by score
            relevant_chunks.sort(key=lambda x: x['score'], reverse=True)
            return relevant_chunks[:k]
            
        except Exception as e:
            logger.error(f"❌ Error retrieving chunks: {e}")
            return []
    
    def generate_educational_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Improved answer generation with better prompt engineering"""
        if not self.llm or not context_chunks:
            return "❌ Unable to generate answer - no relevant information found."
        
        try:
            # Select best context chunks and clean them
            best_chunks = context_chunks[:3]  # Use top 3 most relevant
            
            # Combine and clean context
            context_parts = []
            for chunk in best_chunks:
                # Clean the chunk text
                clean_text = chunk['text']
                # Remove incomplete sentences at the beginning
                sentences = clean_text.split('. ')
                if len(sentences) > 1:
                    # Skip first sentence if it seems incomplete (starts with lowercase)
                    if sentences[0] and sentences[0][0].islower():
                        clean_text = '. '.join(sentences[1:])
                
                context_parts.append(clean_text)
            
            combined_context = '\n\n'.join(context_parts)
            
            # Strong, format-enforcing prompt so the LLM outputs a structured teaching response directly
            prompt = f"""You are an expert elementary/middle-school science teacher. Use ONLY the information in the context below to produce a concise, classroom-style teaching response. DO NOT add external facts or commentary.

Context:
{combined_context[:3000]}

Student Question: {query}

OUTPUT RULES (follow exactly):
- Return ONLY the structured Teaching Response described below. No analysis, no extra headings, no apologies, no step-by-step reasoning.
- Use numbered sections 1-5. Use '*' for sub-bullets. Put each bullet on its own line.
- If the context lacks required information, explicitly say: "I cannot find this information in the provided text."

    REQUIRED STRUCTURE (example - fill in with context-derived content):
📚 Hello class! Today's topic: [Main Topic]

1. What is [Main Concept]?
* Brief definition or explanation
* Key characteristic 1
* Key characteristic 2

2. Key Textbook Points
* Important fact from textbook
* Another important fact
* Connection or relationship
"""
            # New approach: ask the LLM to return a JSON-structured Teaching Response.
            # Define expected JSON schema and ask the model to return JSON only.
            json_prompt = f"""You are an expert science teacher. Use ONLY the context below and the question to produce a Teaching Response in VALID JSON only. Do NOT output anything other than the JSON object.

Context:
{combined_context[:3000]}

Question: {query}

JSON SCHEMA:
{{
  "topic": "string",
  "sections": [
    {{"title": "string", "bullets": ["string"]}}
  ],
  "study_tip": "string"
}}

Rules:
- Return exactly one JSON object following the schema above.
- Use concise bullets. Bullets should be short phrases (not long paragraphs).
- If necessary information is missing from the Context, include the field but set its value to "I cannot find this information in the provided text.".

Now RETURN the JSON object only."""
            
            # Ask the LLM for a JSON-structured Teaching Response, parse it, render as template
            # Try up to 2 attempts to get valid JSON
            answer = None
            answer_obj = None
            for attempt in range(2):
                resp = self.llm(
                    json_prompt,
                    max_new_tokens=400,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                generated_text = resp[0].get('generated_text', '')
                try:
                    m = re.search(r"\{.*\}", generated_text, flags=re.DOTALL)
                    jtxt = m.group(0) if m else generated_text
                    answer_obj = json.loads(jtxt)
                    break
                except Exception:
                    # On failure, append a short clarifying instruction and retry
                    json_prompt = json_prompt + "\n\nPLEASE RETURN A VALID JSON OBJECT FOLLOWING THE SCHEMA."

            if not answer_obj:
                logger.warning("⚠️ Could not parse JSON from LLM; falling back to plain text extraction")
                # Fallback: produce a plain text answer using conservative settings
                resp = self.llm(
                    json_prompt,
                    max_new_tokens=250,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                generated_text = resp[0].get('generated_text', '')
                answer = generated_text.replace('</s>', '').strip()
                if len(answer) < 20:
                    return self.extract_direct_answer(query, combined_context)
            else:
                # Render JSON into the classroom template
                answer = self.render_from_json(answer_obj)

            # Add source information
            sources = []
            for chunk in best_chunks:
                metadata = chunk.get('metadata', {})
                chapter = metadata.get('chapter', 'Unknown Chapter')[:50]
                if chapter not in sources:
                    sources.append(chapter)
            
            if sources and len(answer) > 20:
                answer += f"\n\n📚 Source: {sources[0]}"
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ Error generating answer: {e}")
            return self.extract_direct_answer(query, combined_context if 'combined_context' in locals() else "")
    
    def extract_direct_answer(self, query: str, context: str) -> str:
        """Fallback method to extract direct answers from context"""
        if not context:
            return "❌ No relevant information found."
        
        # Simple keyword-based extraction for common question types
        query_lower = query.lower()
        context_sentences = [s.strip() for s in context.split('.') if s.strip()]
        
        # Look for sentences that might contain the answer
        relevant_sentences = []
        
        # Extract key terms from the question
        key_terms = []
        if 'what is' in query_lower or 'what are' in query_lower:
            # Find the main subject
            if 'what is' in query_lower:
                subject = query_lower.split('what is')[1].split('?')[0].strip()
            else:
                subject = query_lower.split('what are')[1].split('?')[0].strip()
            key_terms.append(subject)
        
        # Add other important words from the question
        important_words = [word for word in query.split() if len(word) > 3 and word.lower() not in ['what', 'how', 'why', 'when', 'where', 'does', 'are', 'the', 'and', 'or']]
        key_terms.extend(important_words[:3])
        
        # Find sentences containing key terms
        for sentence in context_sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for term in key_terms if term.lower() in sentence_lower)
            if score > 0:
                relevant_sentences.append((sentence, score))
        
        # Sort by relevance and take the best ones
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        
        if relevant_sentences:
            answer = '. '.join([sent[0] for sent in relevant_sentences[:3]])
            return answer + '.'
        
        return "❌ I cannot find specific information about this question in the provided text."
    
    def keyword_based_answer_extraction(self, query: str, context_chunks: List[Dict]) -> str:
        """Extract answers using exact keywords and phrases from documents"""
        if not context_chunks:
            return "❌ No relevant information found."
        
        # Combine all relevant context
        full_context = "\n".join([chunk['text'] for chunk in context_chunks])
        
        # Extract key terms from query
        query_lower = query.lower()
        key_terms = self.extract_key_terms(query)
        
        # Find exact matches and relevant sentences
        relevant_info = []
        context_sentences = [s.strip() for s in full_context.split('.') if s.strip()]
        
        # Score sentences based on keyword matches
        scored_sentences = []
        for sentence in context_sentences:
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            sentence_lower = sentence.lower()
            score = 0
            matched_terms = []
            
            # Count exact keyword matches
            for term in key_terms:
                if term.lower() in sentence_lower:
                    score += 1
                    matched_terms.append(term)
            
            # Boost score for definition patterns
            if any(pattern in sentence_lower for pattern in ['is called', 'is known as', 'refers to', 'means', 'is defined as']):
                score += 2
            
            # Boost score for question-specific patterns
            if 'what is' in query_lower and ('is' in sentence_lower or 'are' in sentence_lower):
                score += 1
            if 'how' in query_lower and ('process' in sentence_lower or 'method' in sentence_lower):
                score += 1
            
            if score > 0:
                scored_sentences.append((sentence, score, matched_terms))
        
        # Sort by relevance score
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Build answer from top sentences
        if scored_sentences:
            # Take top 3-5 most relevant sentences
            top_sentences = scored_sentences[:min(5, len(scored_sentences))]
            
            # Filter sentences that are too similar
            unique_sentences = []
            for sentence, score, terms in top_sentences:
                is_duplicate = False
                for existing in unique_sentences:
                    if self.calculate_similarity(sentence, existing[0]) > 0.8:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_sentences.append((sentence, score, terms))
            
            # Create structured answer
            answer_parts = []
            for i, (sentence, score, terms) in enumerate(unique_sentences[:3]):
                # Clean and format sentence
                clean_sentence = sentence.strip()
                if not clean_sentence.endswith('.'):
                    clean_sentence += '.'
                answer_parts.append(clean_sentence)
            
            final_answer = ' '.join(answer_parts)
            
            # Add source information
            sources = []
            for chunk in context_chunks[:2]:
                metadata = chunk.get('metadata', {})
                chapter = metadata.get('chapter', 'Unknown Chapter')[:40]
                if chapter not in sources:
                    sources.append(chapter)
                    
            if sources:
                final_answer += f"\n\n📚 Source: {sources[0]}"
            
            return final_answer
        
        return "❌ Could not find specific information about this question in the documents."
    
    def extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from the query for better matching"""
        # Remove common stop words and question words
        stop_words = {'what', 'is', 'are', 'how', 'why', 'when', 'where', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'do', 'does', 'can', 'will', 'would', 'should'}
        
        # Extract words and phrases
        words = [word.strip('.,!?()[]{}') for word in query.split()]
        key_terms = [word for word in words if len(word) > 2 and word.lower() not in stop_words]
        
        # Also look for phrases (2-3 word combinations)
        phrases = []
        for i in range(len(words) - 1):
            if len(words[i]) > 2 and len(words[i+1]) > 2:
                phrase = f"{words[i]} {words[i+1]}"
                if not any(stop in phrase.lower() for stop in stop_words):
                    phrases.append(phrase)
        
        return key_terms + phrases
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple similarity between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def select_educational_chunks(self, chunks: List[Dict], query: str) -> List[Dict]:
        """Select the most educational chunks based on content type and query relevance"""
        
        # Categorize chunks by educational value
        definition_chunks = []
        example_chunks = []
        concept_chunks = []
        general_chunks = []
        
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            concept_type = metadata.get('concept_type', 'general')
            has_example = metadata.get('has_example', False)
            
            if concept_type == 'definition':
                definition_chunks.append(chunk)
            elif has_example or 'example' in chunk['text'].lower():
                example_chunks.append(chunk)
            elif concept_type == 'concept':
                concept_chunks.append(chunk)
            else:
                general_chunks.append(chunk)
        
        # Prioritize based on query type
        query_lower = query.lower()
        selected = []
        
        # For definition questions, prioritize definitions
        if any(word in query_lower for word in ['what is', 'define', 'meaning']):
            selected.extend(definition_chunks[:2])
            selected.extend(concept_chunks[:1])
            selected.extend(example_chunks[:1])
        
        # For explanation questions, prioritize concepts and examples
        elif any(word in query_lower for word in ['how', 'why', 'explain', 'describe']):
            selected.extend(concept_chunks[:2])
            selected.extend(example_chunks[:2])
            selected.extend(definition_chunks[:1])
        
        # For general questions, balanced selection
        else:
            selected.extend(definition_chunks[:1])
            selected.extend(concept_chunks[:2])
            selected.extend(example_chunks[:1])
        
        # Fill remaining slots with general chunks if needed
        remaining_slots = 4 - len(selected)
        selected.extend(general_chunks[:remaining_slots])
        
        # If we don't have enough, fall back to original order
        if len(selected) < 2:
            selected = chunks[:3]
        
        return selected
    
    def generate_gemini_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate answer using Google Gemini model"""
        if not self.gemini_conversation:
            logger.info("🔄 Gemini not available, using keyword extraction")
            return self.keyword_based_answer_extraction(query, context_chunks)
        
        try:
            # Enhanced context preparation prioritizing educational content
            selected_chunks = self.select_educational_chunks(context_chunks, query)
            context_parts = []
            sources = []
            
            for chunk in selected_chunks[:4]:  # Up to 4 best educational chunks
                # Use more content for better context
                chunk_text = chunk['text'][:1000]  # Further increased chunk size for richer context
                context_parts.append(chunk_text)
                
                # Safely access metadata
                metadata = chunk.get('metadata', {})
                chapter = metadata.get('chapter', 'Unknown Chapter')
                section = metadata.get('section', 'Unknown Section')
                source = f"{chapter} - {section}"
                if source not in sources:
                    sources.append(source)
            
            combined_context = "\n\n".join(context_parts)
            
            # Create comprehensive, format-enforcing prompt for Gemini to produce the teaching response
            educational_prompt = f"""You are an expert science teacher. Use ONLY the textbook excerpts below to produce a concise, classroom-style Teaching Response. Do NOT add outside information.

Textbook Content:
{combined_context[:2500]}

Student Question: {query}

OUTPUT RULES (follow exactly):
- Return ONLY the structured Teaching Response described below. No extra commentary.
- Use numbered sections 1-5 and '*' for sub-bullets. Put each bullet on its own line.
- If the content needed is missing, state: "I cannot find this information in the provided text."

REQUIRED STRUCTURE (fill with textbook-derived content):
📚 Hello class! Today's topic: [Main Topic]

1. What is [Main Concept]?
* Brief definition or explanation
* Key characteristic 1
* Key characteristic 2

2. Key Textbook Points
* Important fact from textbook
* Another important fact
* Connection or relationship

3. [Relevant Section Title]
* Main point about the topic
* Supporting detail
* Real-world application

4. [Additional Section if needed]
* Extra information
* Examples or analogies

5. Study Tip
Think of [concept] as [simple analogy]:
* Input: [what goes in]
* Output: [what comes out]

Only return the structured Teaching Response (do not include any other text)."""
            
            # Get response from Gemini with timeout handling
            logger.info("🤖 Generating answer with Gemini...")
            
            try:
                answer = query_gemini(combined_context, educational_prompt, self.gemini_conversation)
            except Exception as gemini_error:
                logger.warning(f"⚠️ Gemini generation failed: {gemini_error}")
                return self.keyword_based_answer_extraction(query, context_chunks)
            
            # Enhanced educational response processing
            if answer and len(answer.strip()) > 10:
                # Clean and format the response
                answer = re.sub(r'\s+', ' ', answer).strip()
                
                # Remove common artifacts and prefixes
                prefixes_to_remove = ['Assistant:', 'Answer:', 'Response:', 'Teaching Response:', 'Explanation:']
                for prefix in prefixes_to_remove:
                    answer = answer.replace(prefix, '').strip()
                
                # Add educational formatting and structure
                answer = self.format_educational_response(answer, query, context_chunks)
                
                # Add source information and tip
                if sources:
                    source_info = self.create_source_attribution(sources, context_chunks)
                    answer += f"\n\n{source_info}"
                
                # Add educational tip
                answer += f"\n\n💡 Tip: Check diagrams in your textbook to visualize the concepts better."
                
                return answer
            else:
                logger.warning("⚠️ Empty or short response from Ollama")
                return self.keyword_based_answer_extraction(query, context_chunks)
                
        except Exception as e:
            logger.error(f"❌ Error with Gemini generation: {e}")
            return self.keyword_based_answer_extraction(query, context_chunks)
    
    def format_educational_response(self, answer: str, query: str, context_chunks: List[Dict]) -> str:
        """Format response with clean educational structure"""
        
        # If the answer already follows the structured format (starts with 📚), return as-is
        if answer.strip().startswith("📚"):
            return answer.strip()

        # Otherwise, try to render into the classroom template
        try:
            return self.render_classroom_template(answer, query, context_chunks)
        except Exception:
            # Fallback: minimal cleaning
            formatted_answer = answer.strip()
            formatted_answer = re.sub(r'\n{3,}', '\n\n', formatted_answer)
            # Convert long paragraphs into bullet points by splitting on sentences
            sents = re.split(r'(?<=[.!?])\s+', formatted_answer)
            bullets = []
            for s in sents:
                s = s.strip()
                if not s:
                    continue
                # Keep short summary style
                if len(s) > 200:
                    # truncate long sentences
                    s = s[:197].rstrip() + '...'
                bullets.append(f"- {s}")
            return '\n'.join(bullets)

    def render_classroom_template(self, answer: str, query: str, context_chunks: List[Dict]) -> str:
        """Convert a free-form answer into the classroom-style template requested by the user.

        Sections:
        1. What is [Concept]? (definition + key features)
        2. Key Textbook Points
        3. Role of [important term]
        4. Where it happens
        5. Study Tip
        """
        # Helper to pick sentences from text
        def pick_sentences(text: str, keywords: List[str], max_sent: int = 2) -> List[str]:
            sents = re.split(r'(?<=[.!?])\s+', text)
            picked = []
            for sent in sents:
                low = sent.lower()
                if any(k in low for k in keywords):
                    picked.append(sent.strip())
                    if len(picked) >= max_sent:
                        break
            return picked

        topic = self.extract_main_concept(query) or query.title()

        # 1. Definition: prefer explicit definition sentences from answer, else from chunks
        def_sentences = pick_sentences(answer, [topic.lower(), 'is the', 'is a', 'is an', 'means', 'defined as'], 1)
        if not def_sentences:
            # search chunks for definition-like sentences
            combined = ' '.join([c['text'] for c in context_chunks[:6]])
            def_sentences = pick_sentences(combined, ['is', 'are', 'process', 'means', topic.lower()], 1)

        definition = def_sentences[0] if def_sentences else f"{topic.title()} is a concept described in the textbook."

        # 1b. Key features: look for short feature phrases in answer or chunks
        feature_sents = pick_sentences(answer, ['require', 'requires', 'requires', 'need', 'need', 'requires', 'requirement', 'including'], 3)
        if not feature_sents:
            feature_sents = pick_sentences(' '.join([c['text'] for c in context_chunks[:6]]), ['sunlight', 'water', 'carbon', 'chlorophyll', 'leaves'], 3)
        key_features = feature_sents if feature_sents else []

        # 2. Key textbook points: extract 3 important short points from top chunks
        key_points = []
        for c in context_chunks[:6]:
            txt = c.get('text', '').strip()
            # take first short sentence from chunk
            sent = re.split(r'(?<=[.!?])\s+', txt)
            if sent:
                short = sent[0].strip()
                if len(short) > 20 and short not in key_points:
                    key_points.append(short)
            if len(key_points) >= 3:
                break

        # 3. Role section: look for 'chlorophyll' or other important terms; fallback to first chunk sentences mentioning pigment/role
        role_sents = pick_sentences(answer, ['chlorophyll', 'pigment', 'absorb', 'convert', 'convert'], 2)
        if not role_sents:
            role_sents = pick_sentences(' '.join([c['text'] for c in context_chunks[:6]]), ['chlorophyll', 'pigment', 'absorb', 'convert'], 2)

        # 4. Location: look for 'chloroplast', 'leaf', 'mesophyll'
        location_sents = pick_sentences(answer, ['chloroplast', 'leaf', 'mesophyll', 'cells', 'leaf cells'], 2)
        if not location_sents:
            location_sents = pick_sentences(' '.join([c['text'] for c in context_chunks[:6]]), ['chloroplast', 'leaf', 'mesophyll'], 1)

        # 5. Study tip: craft a simple analogy using topic
        study_tip = f"Think of {topic.title()} as a “solar-powered kitchen” for plants!"

        # Build the formatted template
        lines = []
        # Plain, easy-to-read classroom template using numbered sections and '-' bullets
        lines.append(f"📚 Hello class! Today's topic: {topic.title()}")

        # Section 1: Definition and key characteristics
        lines.append("\n1. What is {0}?".format(topic.title()))
        lines.append(f"- Brief definition: {definition}")
        for i, feat in enumerate(key_features, start=1):
            lines.append(f"- Key characteristic {i}: {feat}")

        # Section 2: Key textbook points
        lines.append("\n2. Key Textbook Points")
        if key_points:
            for kp in key_points:
                lines.append(f"- {kp}")
        else:
            lines.append("- See textbook sections for more details.")

        # Section 3: Role (e.g., Chloroplasts / Chlorophyll)
        lines.append("\n3. Role of Chlorophyll and Leaves")
        if role_sents:
            for rs in role_sents:
                lines.append(f"- {rs}")
        else:
            lines.append("- Chlorophyll absorbs sunlight and helps drive the synthesis of food in plants.")
            lines.append("- Leaves are the primary organs where most photosynthesis occurs.")

        # Section 4: Where it happens / supporting detail
        lines.append("\n4. Where it happens / Supporting details")
        if location_sents:
            for ls in location_sents:
                lines.append(f"- {ls}")
        else:
            lines.append("- In the chloroplasts of the leaf cells (mesophyll tissue).")

        # Section 5: Study tip
        lines.append("\n5. Study Tip")
        lines.append(f"- {study_tip}")
        lines.append("- Tip: Check diagrams in your textbook to visualize the concepts better.")

        return '\n'.join(lines)

    def render_from_json(self, obj: Dict[str, Any]) -> str:
        """Render the structured JSON object returned by the LLM into the classroom template."""
        try:
            topic = obj.get('topic', '') or 'Unknown Topic'
            sections = obj.get('sections', []) if isinstance(obj.get('sections', []), list) else []
            study_tip = obj.get('study_tip', '') or ''

            lines = []
            lines.append(f"📚 Hello class! Today's topic: {topic}")

            # Render up to 4 sections (remaining will be put into Additional Section)
            for i in range(4):
                idx = i
                sec_title = sections[idx]['title'] if idx < len(sections) and isinstance(sections[idx], dict) else f"Section {i+1}"
                bullets = []
                if idx < len(sections) and isinstance(sections[idx], dict):
                    bullets = sections[idx].get('bullets', []) or []

                lines.append(f"\n{ i+1 }. {sec_title}")
                if bullets:
                    for b in bullets:
                        bstr = b.strip()
                        if bstr:
                            lines.append(f"* {bstr}")
                else:
                    lines.append("* I cannot find this information in the provided text.")

            # Section 5: study tip
            lines.append("\n5. Study Tip")
            if study_tip:
                lines.append(f"* {study_tip}")
            else:
                lines.append("* I cannot find this information in the provided text.")

            return '\n'.join(lines)
        except Exception as e:
            logger.error(f"Error rendering from json: {e}")
            return "I cannot format the teaching response." 
    
    def extract_main_concept(self, query: str) -> str:
        """Extract the main concept/topic from a question"""
        # Remove question words and common phrases
        question_words = ['what', 'is', 'are', 'define', 'explain', 'describe', 'how', 'why', 'when', 'where']
        
        words = query.lower().split()
        filtered_words = [word for word in words if word not in question_words and len(word) > 2]
        
        # Take the first significant word or phrase
        if filtered_words:
            # Look for compound concepts (e.g., "cell membrane", "photosynthesis process")
            if len(filtered_words) >= 2:
                return ' '.join(filtered_words[:2])
            else:
                return filtered_words[0]
        
        return ""
    
    def create_source_attribution(self, sources: List[str], context_chunks: List[Dict]) -> str:
        """Create simple source attribution"""
        
        if not sources:
            return "� Source: Educational content"
        
        # Simple, clean source format
        source_text = sources[0] if sources else "Educational content"
        attribution = f"� Source: {source_text}"
        
        return attribution

    def query(self, question: str) -> str:
        """Main query method with Ollama integration"""
        logger.info(f"\n❓ Question: {question}")
        
        # Retrieve relevant chunks (increased for better context)
        relevant_chunks = self.retrieve_relevant_chunks(question, k=8)
        
        if not relevant_chunks:
            return "❌ I couldn't find relevant information in the textbook to answer your question."
        
        logger.info(f"📚 Found {len(relevant_chunks)} relevant sections")
        
        # Try Gemini first, fallback to keyword extraction
        if self.gemini_conversation:
            logger.info("🤖 Generating answer with Google Gemini model...")
            answer = self.generate_gemini_answer(question, relevant_chunks)
        else:
            logger.info("🔍 Using keyword-based extraction...")
            answer = self.keyword_based_answer_extraction(question, relevant_chunks)
        
        logger.info(f"✅ Generated answer with {len(answer)} characters")
        
        # Show relevance scores for debugging
        scores = [f"{chunk['score']:.3f}" for chunk in relevant_chunks]
        logger.info(f"📊 Relevance scores: {scores}")
        
        return answer
    
    def get_chapter_overview(self) -> str:
        """Get an overview of available chapters"""
        if not self.chunk_metadata:
            return "📖 No content loaded yet."
        
        chapters = set()
        sections_count = {}
        
        for metadata in self.chunk_metadata:
            chapter = metadata['chapter']
            chapters.add(chapter)
            if chapter in sections_count:
                sections_count[chapter] += 1
            else:
                sections_count[chapter] = 1
        
        overview = "📖 Available content in the textbook:\n"
        for i, chapter in enumerate(sorted(chapters), 1):
            chunk_count = sections_count[chapter]
            overview += f"{i}. {chapter[:60]}... ({chunk_count} sections)\n"
        
        overview += f"\n📊 Total: {len(chapters)} chapters, {len(self.chunk_metadata)} text sections"
        return overview

    def debug_retrieval(self, query: str, k: int = 5) -> Dict:
        """Debug method to show detailed retrieval information"""
        print(f"\n🔍 DEBUG: Analyzing retrieval for query: '{query}'")
        print("="*60)
        
        # Get relevant chunks with detailed info
        relevant_chunks = self.retrieve_relevant_chunks(query, k)
        
        debug_info = {
            'query': query,
            'total_chunks_in_database': len(self.chunks),
            'retrieved_chunks': len(relevant_chunks),
            'chunks_details': []
        }
        
        if not relevant_chunks:
            print("❌ No chunks retrieved!")
            return debug_info
        
        print(f"📊 Retrieved {len(relevant_chunks)} chunks out of {len(self.chunks)} total chunks")
        print("\n📝 RETRIEVED CHUNKS DETAILS:")
        print("-" * 60)
        
        for i, chunk in enumerate(relevant_chunks, 1):
            metadata = chunk.get('metadata', {})
            chunk_info = {
                'rank': i,
                'score': chunk['score'],
                'chapter': metadata.get('chapter', 'Unknown Chapter'),
                'section': metadata.get('section', 'Unknown Section'),
                'type': metadata.get('type', 'unknown'),
                'text_preview': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                'text_length': len(chunk['text'])
            }
            debug_info['chunks_details'].append(chunk_info)
            
            print(f"\n🎯 CHUNK #{i}")
            print(f"   📊 Relevance Score: {chunk['score']:.4f}")
            print(f"   📚 Chapter: {metadata.get('chapter', 'Unknown Chapter')[:60]}...")
            print(f"   📄 Section: {metadata.get('section', 'Unknown Section')}")
            print(f"   🔧 Type: {metadata.get('type', 'unknown')}")
            print(f"   📏 Length: {len(chunk['text'])} characters")
            print(f"   📝 Preview: {chunk['text'][:200]}...")
            
            # Show keyword matches
            query_words = query.lower().split()
            chunk_text_lower = chunk['text'].lower()
            matches = [word for word in query_words if word in chunk_text_lower and len(word) > 2]
            if matches:
                print(f"   🎯 Keyword matches: {matches}")
            print("-" * 40)
        
        return debug_info
    
    def test_retrieval_quality(self, test_queries: List[str]) -> None:
        """Test retrieval quality with multiple queries"""
        print("\n🧪 TESTING RETRIEVAL QUALITY")
        print("="*70)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔬 TEST {i}: {query}")
            print("-" * 50)
            
            chunks = self.retrieve_relevant_chunks(query, k=3)
            if chunks:
                print(f"✅ Retrieved {len(chunks)} chunks")
                avg_score = sum(chunk['score'] for chunk in chunks) / len(chunks)
                print(f"📊 Average relevance score: {avg_score:.4f}")
                
                # Show best match preview
                best_match = chunks[0]
                print(f"🏆 Best match (score: {best_match['score']:.4f}):")
                print(f"   📚 From: {best_match['metadata']['chapter'][:40]}...")
                print(f"   📝 Text: {best_match['text'][:150]}...")
            else:
                print("❌ No chunks retrieved")
            print()
    
    def show_database_stats(self) -> None:
        """Show statistics about the loaded database"""
        if not self.chunks:
            print("❌ No data loaded in database")
            return
        
        print("\n📊 DATABASE STATISTICS")
        print("="*50)
        print(f"📄 Total chunks: {len(self.chunks)}")
        print(f"💾 Using FAISS: {self.use_faiss}")
        
        # Chunk length statistics
        lengths = [len(chunk) for chunk in self.chunks]
        print(f"📏 Average chunk length: {sum(lengths)/len(lengths):.0f} characters")
        print(f"📏 Min chunk length: {min(lengths)} characters")
        print(f"📏 Max chunk length: {max(lengths)} characters")
        
        # Chapter/section breakdown
        chapters = {}
        section_types = {'complete_section': 0, 'partial_section': 0}
        
        for metadata in self.chunk_metadata:
            chapter = metadata['chapter'][:30] + "..." if len(metadata['chapter']) > 30 else metadata['chapter']
            chapters[chapter] = chapters.get(chapter, 0) + 1
            section_types[metadata['type']] = section_types.get(metadata['type'], 0) + 1
        
        print(f"\n📚 Content breakdown:")
        print(f"   🔹 Complete sections: {section_types.get('complete_section', 0)}")
        print(f"   🔹 Partial sections: {section_types.get('partial_section', 0)}")
        
        print(f"\n📖 Top chapters by chunk count:")
        sorted_chapters = sorted(chapters.items(), key=lambda x: x[1], reverse=True)
        for chapter, count in sorted_chapters[:5]:
            print(f"   📚 {chapter}: {count} chunks")
    
    def search_raw_text(self, search_term: str, case_sensitive: bool = False, return_results: bool = False) -> List[Dict]:
        """Search for exact text matches in the database"""
        if not return_results:
            print(f"\n🔍 SEARCHING RAW TEXT FOR: '{search_term}'")
            print("="*60)
        
        matches = []
        search_term_processed = search_term if case_sensitive else search_term.lower()
        
        for i, chunk in enumerate(self.chunks):
            chunk_text = chunk if case_sensitive else chunk.lower()
            if search_term_processed in chunk_text:
                # Find context around the match
                match_pos = chunk_text.find(search_term_processed)
                start = max(0, match_pos - 50)
                end = min(len(chunk), match_pos + len(search_term) + 50)
                context = chunk[start:end]
                
                match_info = {
                    'chunk_index': i,
                    'metadata': self.chunk_metadata[i],
                    'match_position': match_pos,
                    'context': context,
                    'full_text': chunk
                }
                matches.append(match_info)
        
        if not return_results:
            print(f"📊 Found {len(matches)} exact matches")
            
            for i, match in enumerate(matches[:5], 1):  # Show first 5 matches
                print(f"\n🎯 MATCH #{i}")
                print(f"   📚 Chapter: {match['metadata']['chapter'][:50]}...")
                print(f"   📄 Section: {match['metadata']['section']}")
                print(f"   📍 Position: {match['match_position']}")
                print(f"   📝 Context: ...{match['context']}...")
                print("-" * 40)
        
        return matches
    
    def verify_retrieval_pipeline(self, test_query: str = "What is a cell?") -> None:
        """Comprehensive verification of the entire retrieval pipeline"""
        print("\n🔧 VERIFYING RETRIEVAL PIPELINE")
        print("="*70)
        
        # Step 1: Check if data is loaded
        print("1️⃣ Checking data loading...")
        if not self.chunks:
            print("❌ No chunks loaded!")
            return
        print(f"✅ {len(self.chunks)} chunks loaded")
        
        # Step 2: Check embeddings
        print("\n2️⃣ Checking embeddings...")
        if self.use_faiss and self.index is not None:
            print(f"✅ FAISS index available with {self.index.ntotal} vectors")
        elif self.embeddings_matrix is not None:
            print(f"✅ Sklearn embeddings matrix available: {self.embeddings_matrix.shape}")
        else:
            print("❌ No embeddings available!")
            return
        
        # Step 3: Test query embedding
        print("\n3️⃣ Testing query embedding...")
        try:
            query_embedding = self.embeddings_model.encode([test_query])
            print(f"✅ Query embedded successfully: shape {query_embedding.shape}")
        except Exception as e:
            print(f"❌ Query embedding failed: {e}")
            return
        
        # Step 4: Test similarity search
        print("\n4️⃣ Testing similarity search...")
        relevant_chunks = self.retrieve_relevant_chunks(test_query, k=3)
        if relevant_chunks:
            print(f"✅ Retrieved {len(relevant_chunks)} relevant chunks")
            scores_list = [f"{chunk['score']:.3f}" for chunk in relevant_chunks]
            print(f"📊 Scores: {scores_list}")
        else:
            print("❌ No relevant chunks retrieved!")
            return
        
        # Step 5: Test answer generation
        print("\n5️⃣ Testing answer generation...")
        if self.gemini_conversation:
            try:
                answer = self.generate_gemini_answer(test_query, relevant_chunks)
                print(f"✅ Gemini answer generated: {len(answer)} characters")
                print(f"📝 Preview: {answer[:100]}...")
            except Exception as e:
                print(f"⚠️ Gemini failed, testing keyword extraction: {e}")
        
        keyword_answer = self.keyword_based_answer_extraction(test_query, relevant_chunks)
        print(f"✅ Keyword answer generated: {len(keyword_answer)} characters")
        print(f"📝 Preview: {keyword_answer[:100]}...")
        
        print("\n✅ PIPELINE VERIFICATION COMPLETE!")

    # Additional methods for Flask app compatibility
    def add_document_from_file(self, file_path: str) -> bool:
        """Add a document from file (compatibility wrapper for load_and_process_text)"""
        success = self.load_and_process_text(file_path)
        if success:
            # Create embeddings after loading the document
            self.create_embeddings()
        return success
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question and get formatted response (compatibility wrapper for query)"""
        try:
            answer = self.query(question)
            
            # Get relevant chunks for sources
            relevant_chunks = self.retrieve_relevant_chunks(question, k=5)
            
            # Format sources
            sources = []
            for i, chunk in enumerate(relevant_chunks[:5]):  # Limit to top 5 sources
                sources.append({
                    'text': chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'],
                    'score': chunk['score'],
                    'chunk_index': chunk.get('index', i)
                })
            
            return {
                'response': answer,
                'sources': sources,
                'mode': 'gemini' if self.gemini_conversation else 'keyword'
            }
        except Exception as e:
            logger.error(f"Error in ask_question: {e}")
            return {
                'response': f"Error generating answer: {str(e)}",
                'sources': [],
                'mode': 'error'
            }
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search documents and return formatted results"""
        try:
            relevant_chunks = self.retrieve_relevant_chunks(query, k=top_k)
            
            results = []
            for i, chunk in enumerate(relevant_chunks):
                # Get full metadata from chunk
                original_metadata = chunk.get('metadata', {})
                
                results.append({
                    'chunk': chunk['text'],
                    'score': chunk['score'],
                    'metadata': {
                        'filename': 'document.txt',  # Default filename
                        'chunk_index': chunk.get('index', i),
                        'chapter': original_metadata.get('chapter', f'Chapter {(i // 50) + 1}'),
                        'section': original_metadata.get('section', f'Part {(i % 50) + 1}')
                    }
                })
            
            return results
        except Exception as e:
            logger.error(f"Error in search_documents: {e}")
            return []
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded documents"""
        try:
            # Create document information structure expected by frontend
            documents_info = {}
            chunk_details = []
            
            if self.chunks:
                # Create detailed chunk information
                for i, chunk in enumerate(self.chunks):
                    chunk_info = {
                        'id': i,
                        'content': chunk[:500] + "..." if len(chunk) > 500 else chunk,  # Truncate for display
                        'full_length': len(chunk),
                        'word_count': len(chunk.split()),
                        'preview': chunk[:100] + "..." if len(chunk) > 100 else chunk
                    }
                    
                    # Add metadata if available
                    if hasattr(self, 'chunk_metadata') and self.chunk_metadata and i < len(self.chunk_metadata):
                        metadata = self.chunk_metadata[i]
                        chunk_info.update({
                            'section': metadata.get('section', 'Unknown'),
                            'start_char': metadata.get('start_char', 0),
                            'end_char': metadata.get('end_char', len(chunk))
                        })
                    else:
                        chunk_info.update({
                            'section': f'Section {(i // 50) + 1}',  # Group chunks into sections
                            'start_char': 0,
                            'end_char': len(chunk)
                        })
                    
                    chunk_details.append(chunk_info)
                
                # Document summary
                documents_info['example.txt'] = {
                    'chunks': len(self.chunks),
                    'characters': sum(len(chunk) for chunk in self.chunks),
                    'sections': list(set([chunk['section'] for chunk in chunk_details])),
                    'chunk_details': chunk_details
                }
            
            return {
                'total_chunks': len(self.chunks),
                'total_characters': sum(len(chunk) for chunk in self.chunks),
                'average_chunk_size': sum(len(chunk) for chunk in self.chunks) / len(self.chunks) if self.chunks else 0,
                'embeddings_created': (self.index is not None) or (self.embeddings_matrix is not None),
                'gemini_available': self.gemini_conversation is not None,
                'documents': documents_info,
                'chunks': chunk_details  # Add chunk details at top level for easy access
            }
        except Exception as e:
            logger.error(f"Error getting document stats: {e}")
            return {
                'total_chunks': 0,
                'total_characters': 0,
                'average_chunk_size': 0,
                'embeddings_created': (self.index is not None) or (self.embeddings_matrix is not None) if hasattr(self, 'index') and hasattr(self, 'embeddings_matrix') else False,
                'gemini_available': self.gemini_conversation is not None if hasattr(self, 'gemini_conversation') else False,
                'documents': {},
                'chunks': []
            }

def main():
    """Main function to run the Gemini-powered RAG system"""
    logger.info("🚀 Initializing Gemini-powered Educational RAG System...")
    
    # Initialize with Ollama
    rag = MistralRAG()
    
    # Load and process the data
    data_path = "d:/projects/RAG/data/example.txt"
    if not rag.load_and_process_text(data_path):
        logger.error("Failed to load text data")
        return
    
    # Create embeddings
    if not rag.create_embeddings():
        logger.error("Failed to create embeddings")
        return
    
    # Show available content
    print("\n" + "="*70)
    print("GOOGLE GEMINI-POWERED EDUCATIONAL RAG SYSTEM")
    print("="*70)
    print("🤖 Using: Google Gemini via API")
    print("🔗 Endpoint: https://generativelanguage.googleapis.com")
    print(rag.get_chapter_overview())
    
    # Show database statistics
    rag.show_database_stats()
    
    # Test retrieval quality
    test_queries = [
        "What is a cell?",
        "photosynthesis",
        "unicellular organisms",
        "plant nutrition"
    ]
    rag.test_retrieval_quality(test_queries)
    
    # Verify the pipeline
    rag.verify_retrieval_pipeline()
    
    # Test with educational questions
    test_questions = [
        "What is a cell?",
        "What are unicellular organisms?", 
        "What is photosynthesis?",
        "How do plants make their food?",
    ]
    
    print("\n📋 Testing with educational questions:")
    print("-" * 70)
    
    for i, question in enumerate(test_questions[:2], 1):  # Test first 2
        print(f"\n{i}. {question}")
        answer = rag.query(question)
        print(f"🤖 Answer: {answer}")
        print("-" * 70)
    
    # Interactive mode
    print("\n🎯 Interactive Mode - Ask questions about the science textbook:")
    print("(Type 'quit', 'exit', 'chapters', 'debug <question>', 'search <term>', or 'stats')")
    print("💡 This system uses Ollama Mistral for intelligent responses")
    print("⚠️  Make sure Ollama is running: ollama serve")
    
    while True:
        user_question = input("\n❓ Your question: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye! Happy learning!")
            break
        elif user_question.lower() == 'chapters':
            print(rag.get_chapter_overview())
            continue
        elif user_question.lower() == 'stats':
            rag.show_database_stats()
            continue
        elif user_question.lower().startswith('debug '):
            debug_query = user_question[6:].strip()
            if debug_query:
                rag.debug_retrieval(debug_query)
            else:
                print("Usage: debug <your question>")
            continue
        elif user_question.lower().startswith('search '):
            search_term = user_question[7:].strip()
            if search_term:
                rag.search_raw_text(search_term)
            else:
                print("Usage: search <search term>")
            continue
        elif user_question.lower() == 'test':
            # Test Gemini connection
            if rag.gemini_conversation:
                try:
                    test_response = query_gemini("Science textbook", "Hello, can you help with science questions?")
                    print(f"🧪 Gemini test: {test_response[:100]}...")
                except Exception as e:
                    print(f"❌ Gemini test failed: {e}")
            else:
                print("❌ Gemini not available")
            continue
        elif user_question.lower() == 'verify':
            rag.verify_retrieval_pipeline()
            continue
        elif not user_question:
            continue
        
        answer = rag.query(user_question)
        print(f"🤖 Answer: {answer}")

if __name__ == "__main__":
    main()