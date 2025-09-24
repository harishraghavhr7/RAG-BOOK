import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer

# Handle FAISS import with fallback
FAISS_AVAILABLE = False
try:
    import faiss
    FAISS_AVAILABLE = True
    print("✅ FAISS loaded successfully")
except ImportError:
    try:
        import faiss_cpu as faiss
        FAISS_AVAILABLE = True
        print("✅ FAISS-CPU loaded successfully")
    except ImportError:
        print("❌ FAISS not available. Installing faiss-cpu...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
        try:
            import faiss
            FAISS_AVAILABLE = True
            print("✅ FAISS installed and loaded successfully")
        except ImportError:
            print("❌ Failed to install FAISS. Using sklearn as fallback.")
            FAISS_AVAILABLE = False

# Fallback imports if FAISS is not available
if not FAISS_AVAILABLE:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer

# Import Ollama client
try:
    from ollama_client import OllamaConversation, query_ollama
    OLLAMA_AVAILABLE = True
    print("✅ Ollama client loaded successfully")
except ImportError:
    print("❌ Ollama client not available")
    OLLAMA_AVAILABLE = False

from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MistralRAG:
    def __init__(self):
        """Initialize RAG system with Ollama Mistral integration"""
        try:
            # Try to load the sentence transformer with a timeout approach
            import signal
            import os
            
            logger.info("Loading sentence transformer model...")
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Sentence transformer loaded successfully")
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
        self.ollama_conversation = None
        self.setup_ollama()
    
    def setup_ollama(self):
        """Setup Ollama conversation client"""
        if OLLAMA_AVAILABLE:
            try:
                # Quick connection test with short timeout
                from ollama_client import test_ollama_connection
                if test_ollama_connection():
                    self.ollama_conversation = OllamaConversation()
                    logger.info("✅ Ollama conversation client initialized")
                    # Skip the actual test call during initialization to avoid hanging
                    logger.info("✅ Ollama connection successful")
                else:
                    logger.warning("⚠️ Ollama connection test failed, will retry during first query")
                    self.ollama_conversation = None
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to setup Ollama (will use fallback): {e}")
                self.ollama_conversation = None
        else:
            logger.info("💡 Ollama not available, will use keyword extraction only")

    def smart_chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 200) -> List[Dict]:
        """Smart chunking that respects educational content structure"""
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
                # Split large chapters into smaller chunks
                sentences = re.split(r'[.!?]+\s+', clean_text)
                current_chunk = ""
                chunk_count = 1
                
                for sentence in sentences:
                    if len(current_chunk + sentence) <= chunk_size:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk.strip():
                            chunks.append({
                                'text': current_chunk.strip(),
                                'chapter': chapter_title,
                                'section': f'Part {chunk_count}',
                                'type': 'partial_section',
                                'part': str(chunk_count)
                            })
                            chunk_count += 1
                        current_chunk = sentence + ". "
                
                # Add the last chunk
                if current_chunk.strip():
                    chunks.append({
                        'text': current_chunk.strip(),
                        'chapter': chapter_title,
                        'section': f'Part {chunk_count}',
                        'type': 'partial_section',
                        'part': str(chunk_count)
                    })
        
        # If still no chunks, use simple splitting
        if not chunks:
            logger.info("Chapter parsing failed, using simple chunking")
            return self.simple_chunk_text(text, chunk_size, overlap)
        
        logger.info(f"✅ Created {len(chunks)} smart chunks from {len(found_chapters)} sections")
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
                metadata = self.chunk_metadata[i]
                # Add context to help with retrieval
                enhanced_text = f"Chapter: {metadata['chapter']} | Section: {metadata['section']} | Content: {chunk}"
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
                        metadata = self.chunk_metadata[idx]
                        section_key = f"{metadata['chapter']}_{metadata['section']}"
                        
                        # Prefer complete sections and avoid duplicates
                        score = float(scores[0][i])
                        if metadata['type'] == 'complete_section':
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
                        metadata = self.chunk_metadata[idx]
                        section_key = f"{metadata['chapter']}_{metadata['section']}"
                        
                        score = float(similarities[idx])
                        if metadata['type'] == 'complete_section':
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
            
            # Improved prompt with better instructions
            prompt = f"""You are a helpful science teacher. Answer the student's question using only the information provided in the context below.

Context:
{combined_context[:2000]}

Question: {query}

Instructions:
1. Give a clear, complete answer based only on the context provided
2. Use simple language suitable for students
3. If you cannot find the answer in the context, say "I cannot find this information in the provided text"
4. Do not add information not present in the context
5. Structure your answer clearly with main points

Answer:"""
            
            # Generate with more conservative settings
            response = self.llm(
                prompt, 
                max_new_tokens=200,
                temperature=0.1,  # Lower temperature for more focused answers
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract and clean the answer
            generated_text = response[0]['generated_text']
            
            # Find the answer part
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                # If no "Answer:" marker, take everything after the prompt
                answer = generated_text[len(prompt):].strip()
            
            # Clean up the answer
            answer = re.sub(r'\s+', ' ', answer)  # Normalize whitespace
            answer = answer.replace('</s>', '').strip()  # Remove end tokens
            
            # Ensure the answer is not just repeating the context
            if len(answer) < 20 or answer.lower().startswith('context'):
                # Fallback: extract key information directly
                return self.extract_direct_answer(query, combined_context)
            
            # Add source information
            sources = list(set([chunk['metadata']['chapter'][:50] for chunk in best_chunks]))
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
            sources = list(set([chunk['metadata']['chapter'][:40] for chunk in context_chunks[:2]]))
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
    
    def generate_ollama_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate answer using Ollama Mistral model"""
        if not self.ollama_conversation:
            logger.info("🔄 Ollama not available, using keyword extraction")
            return self.keyword_based_answer_extraction(query, context_chunks)
        
        try:
            # Prepare context from chunks with length limits
            context_parts = []
            sources = []
            
            for chunk in context_chunks[:2]:  # Reduced to top 2 chunks
                context_parts.append(chunk['text'][:400])  # Reduced chunk size
                source = f"{chunk['metadata']['chapter']} - {chunk['metadata']['section']}"
                if source not in sources:
                    sources.append(source)
            
            combined_context = "\n\n".join(context_parts)
            
            # Create shorter, more focused prompt for Mistral
            educational_prompt = f"""Answer this science question using only the textbook content provided.

Textbook Content:
{combined_context[:800]}

Question: {query}

Provide a clear, short answer based only on the textbook content above:"""
            
            # Get response from Ollama with timeout handling
            logger.info("🤖 Generating answer with Ollama Mistral...")
            
            try:
                answer = query_ollama(combined_context, educational_prompt, self.ollama_conversation)
            except Exception as ollama_error:
                logger.warning(f"⚠️ Ollama generation failed: {ollama_error}")
                return self.keyword_based_answer_extraction(query, context_chunks)
            
            # Clean up the response
            if answer and len(answer.strip()) > 5:
                # Remove any repetitive parts
                answer = re.sub(r'\s+', ' ', answer).strip()
                
                # Remove common artifacts
                answer = answer.replace('Assistant:', '').strip()
                answer = answer.replace('Answer:', '').strip()
                
                # Limit answer length to prevent timeouts
                sentences = answer.split('.')
                if len(sentences) > 4:
                    answer = '. '.join(sentences[:4]) + '.'
                
                # Add source information
                if sources and len(answer) > 15:
                    answer += f"\n\n📚 Source: {sources[0][:40]}"
                
                return answer
            else:
                logger.warning("⚠️ Empty or short response from Ollama")
                return self.keyword_based_answer_extraction(query, context_chunks)
                
        except Exception as e:
            logger.error(f"❌ Error with Ollama generation: {e}")
            return self.keyword_based_answer_extraction(query, context_chunks)

    def query(self, question: str) -> str:
        """Main query method with Ollama integration"""
        logger.info(f"\n❓ Question: {question}")
        
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(question, k=5)
        
        if not relevant_chunks:
            return "❌ I couldn't find relevant information in the textbook to answer your question."
        
        logger.info(f"📚 Found {len(relevant_chunks)} relevant sections")
        
        # Try Ollama first, fallback to keyword extraction
        if self.ollama_conversation:
            logger.info("🤖 Generating answer with Ollama Mistral model...")
            answer = self.generate_ollama_answer(question, relevant_chunks)
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
            chunk_info = {
                'rank': i,
                'score': chunk['score'],
                'chapter': chunk['metadata']['chapter'],
                'section': chunk['metadata']['section'],
                'type': chunk['metadata']['type'],
                'text_preview': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                'text_length': len(chunk['text'])
            }
            debug_info['chunks_details'].append(chunk_info)
            
            print(f"\n🎯 CHUNK #{i}")
            print(f"   📊 Relevance Score: {chunk['score']:.4f}")
            print(f"   📚 Chapter: {chunk['metadata']['chapter'][:60]}...")
            print(f"   📄 Section: {chunk['metadata']['section']}")
            print(f"   🔧 Type: {chunk['metadata']['type']}")
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
        if self.ollama_conversation:
            try:
                answer = self.generate_ollama_answer(test_query, relevant_chunks)
                print(f"✅ Ollama answer generated: {len(answer)} characters")
                print(f"📝 Preview: {answer[:100]}...")
            except Exception as e:
                print(f"⚠️ Ollama failed, testing keyword extraction: {e}")
        
        keyword_answer = self.keyword_based_answer_extraction(test_query, relevant_chunks)
        print(f"✅ Keyword answer generated: {len(keyword_answer)} characters")
        print(f"📝 Preview: {keyword_answer[:100]}...")
        
        print("\n✅ PIPELINE VERIFICATION COMPLETE!")

def main():
    """Main function to run the Ollama-powered RAG system"""
    logger.info("🚀 Initializing Ollama-powered Educational RAG System...")
    
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
    print("OLLAMA MISTRAL-POWERED EDUCATIONAL RAG SYSTEM")
    print("="*70)
    print("🤖 Using: Local Mistral via Ollama")
    print("🔗 Endpoint: http://localhost:11434")
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
            # Test Ollama connection
            if rag.ollama_conversation:
                try:
                    test_response = query_ollama("Science textbook", "Hello, can you help with science questions?")
                    print(f"🧪 Ollama test: {test_response[:100]}...")
                except Exception as e:
                    print(f"❌ Ollama test failed: {e}")
            else:
                print("❌ Ollama not available")
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