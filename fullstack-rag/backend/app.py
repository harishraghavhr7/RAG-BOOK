from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import sys
import logging
import json
from pdf_utils import process_pdf
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
ALLOWED_EXTENSIONS = {'pdf'}

# Add the parent directory to sys.path to import our RAG modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Using optimized RAG system with Gemini
logger.info("🔄 Using optimized RAG system with Gemini")
from optimized_rag import MistralRAG
import optimized_rag_demo as demo_rag

def check_numpy_compatibility():
    try:
        import numpy as _np
        ver = _np.__version__
        major = int(ver.split('.')[0]) if ver else 0
        if major >= 2:
            logger.warning(
                "Detected NumPy >=2.0 which can be incompatible with some compiled packages (torch/pybind).\n"
                "Recommended: install a NumPy 1.x build compatible with your environment.\n"
                "Example (in a virtualenv): python -m pip install \"numpy<2.0\"\n"
            )
        else:
            logger.info(f"NumPy version {ver} detected — compatible.")
    except Exception:
        logger.error(
            "NumPy is not installed or could not be imported. Embeddings (sentence-transformers/FAISS) will not work.\n"
            "To fix: create and activate a virtual environment then run:\n"
            "python -m venv .venv\n"
            ".\\.venv\\Scripts\\Activate.ps1  # (PowerShell)\n"
            "python -m pip install -r requirements.txt\n"
        )

check_numpy_compatibility()

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes

# Global RAG instance
rag_instance = None
rag_initialized = False

def initialize_rag():
    """Initialize the RAG system"""
    global rag_instance, rag_initialized
    
    if rag_initialized:
        return rag_instance
    
    try:
        logger.info("Initializing RAG system...")
        
        # Try to initialize RAG, but handle network errors gracefully
        try:
            rag_instance = MistralRAG()
        except Exception as model_error:
            logger.error(f"❌ Failed to initialize RAG system: {model_error}")
            logger.info("💡 This might be due to network issues downloading the model.")
            logger.info("💡 Falling back to demo RAG implementation to keep the server running.")
            try:
                rag_instance = demo_rag.MistralRAG()
                rag_initialized = True
                logger.info("✅ Demo RAG initialized as a fallback")
            except Exception as demo_error:
                logger.error(f"❌ Failed to initialize demo RAG fallback: {demo_error}")
                logger.info("💡 You can try to run the demo directly: python fullstack-rag/backend/app_demo.py")
                return None
        
        # Load data from the data directory
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.endswith('.txt'):
                    file_path = os.path.join(data_dir, filename)
                    logger.info(f"Loading document: {filename}")
                    try:
                        rag_instance.add_document_from_file(file_path)
                    except Exception as load_error:
                        logger.error(f"❌ Failed to load document {filename}: {load_error}")
            
            rag_initialized = True
            logger.info(f"✅ RAG system initialized with {len(rag_instance.chunks)} chunks")
        else:
            logger.warning("⚠️ Data directory not found")
            rag_initialized = True  # Still initialize even without data
            
        return rag_instance
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG: {e}")
        return None

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'rag_initialized': rag_initialized,
        'chunks_loaded': len(rag_instance.chunks) if rag_instance else 0
    })







@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        message = data['message'].strip()
        
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Initialize RAG if not already done
        rag = initialize_rag()
        
        if not rag:
            return jsonify({'error': 'RAG system not initialized'}), 500
        
        if not rag_initialized:
            return jsonify({'error': 'No documents loaded'}), 500
        
        # Get response from RAG
        logger.info(f"Processing query: {message}")
        result = rag.ask_question(message)
        # Normalize and structure the response for consistent frontend rendering.
        raw_response = result.get('response', '') if isinstance(result, dict) else str(result)

        # Normalize line endings and split into paragraphs. Then join paragraphs
        # with a blank line (i.e., two newlines) so frontends display spacing.
        def format_response_text(text: str):
            if text is None:
                return '', []
            # normalize CRLF
            txt = text.replace('\r\n', '\n').replace('\r', '\n')
            # split on one-or-more blank lines
            paragraphs = [p.strip() for p in re.split(r'\n\s*\n+', txt) if p.strip()]
            # fallback: split on single newlines if we didn't find paragraph separators
            if not paragraphs:
                paragraphs = [line.strip() for line in txt.split('\n') if line.strip()]
            formatted = '\n\n'.join(paragraphs)
            return formatted, paragraphs

        formatted_text, paragraphs = format_response_text(raw_response)

        # Derive a list of structured points from the paragraphs.
        def extract_points(paragraphs_list):
            pts = []
            for p in paragraphs_list:
                if not p:
                    continue

                # 1) If the paragraph contains explicit list items (bullets or numbers), extract them.
                list_items = [m.group(1).strip() for m in re.finditer(r'^\s*(?:[-*\u2022]|\d+[\).])\s*(.+)$', p, re.MULTILINE)]
                if list_items:
                    pts.extend(list_items)
                    continue

                # 2) Otherwise, split paragraph into sentences and treat each sentence as a point.
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', p) if s.strip()]
                if sentences:
                    pts.extend(sentences)
                else:
                    # Fallback: use the whole paragraph
                    pts.append(p.strip())

            # Trim and dedupe small duplicates while preserving order
            seen = set()
            cleaned = []
            for t in pts:
                if not t:
                    continue
                key = t.strip()
                if key in seen:
                    continue
                seen.add(key)
                cleaned.append(key)

            # Limit to a reasonable number of points
            return cleaned[:50]

        response_points = extract_points(paragraphs)

        return jsonify({
            'response': formatted_text,
            'response_paragraphs': paragraphs,
            'response_points': response_points,
            'sources': result.get('sources', []) if isinstance(result, dict) else [],
            'status': 'success',
            'mode': result.get('mode', 'unknown') if isinstance(result, dict) else 'unknown'
        })
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/search', methods=['POST'])
def search():
    """Search endpoint for raw text search"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query'].strip()
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Initialize RAG if not already done
        rag = initialize_rag()
        
        if not rag or not rag_initialized:
            return jsonify({'error': 'RAG system not available'}), 500
        
        # Perform search
        search_results = rag.search_documents(query, top_k=5)
        
        # Format results for frontend
        results = []
        for result in search_results:
            results.append({
                'text': result['chunk'],
                'score': result['score'],
                'filename': result['metadata']['filename'],
                'chunk_index': result['metadata']['chunk_index']
            })
        
        return jsonify({
            'results': results,
            'status': 'success',
            'count': len(results)
        })
        
    except Exception as e:
        logger.error(f"Error processing search request: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Get information about loaded documents"""
    try:
        rag = initialize_rag()
        
        if not rag or not rag_initialized:
            return jsonify({'error': 'RAG system not available'}), 500
        
        # Get document statistics
        stats = rag.get_document_stats()
        
        return jsonify({
            'status': 'success',
            'stats': stats
        })

        
    except Exception as e:
        logger.error(f"Error getting documents info: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/documents', methods=['DELETE'])
def delete_document():
    """Delete a document (text file) and remove its chunks from the RAG index"""
    try:
        data = request.get_json()
        if not data or 'filename' not in data:
            return jsonify({'error': 'filename is required'}), 400

        filename = secure_filename(data['filename'])
        txt_path = os.path.join(UPLOAD_FOLDER, filename)

        # Initialize RAG
        rag = initialize_rag()
        if not rag:
            return jsonify({'error': 'RAG not initialized'}), 500

        # Remove chunks from RAG
        removed = rag.remove_document_by_filename(filename)
        if not removed:
            # Even if no chunks were removed, attempt to delete the file if present
            if os.path.exists(txt_path):
                try:
                    os.remove(txt_path)
                except Exception:
                    pass
            return jsonify({'error': 'Document not found in RAG or removal failed'}), 404

        # Delete the file from disk if present
        try:
            if os.path.exists(txt_path):
                os.remove(txt_path)
        except Exception as e:
            logger.warning(f"Failed to delete file from disk: {e}")

        return jsonify({'status': 'success', 'message': f'{filename} removed'}), 200

    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        return jsonify({'error': 'Internal server error'}), 500

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        # Check if file part exists
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        
        # Check if a file was selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
            
        # Save and process the file
        try:
            # Secure the filename and save the PDF
            filename = secure_filename(file.filename)
            pdf_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(pdf_path)
            
            # Process the PDF and extract text
            txt_path = process_pdf(pdf_path, UPLOAD_FOLDER)
            
            # Remove the PDF file after processing
            os.remove(pdf_path)
            # Ensure RAG is initialized and then add the new text file incrementally
            rag = initialize_rag()
            if not rag:
                logger.error("Failed to initialize RAG after file upload")
                return jsonify({'error': 'Failed to process document'}), 500

            try:
                added = rag.add_document_from_file(txt_path)
                if not added:
                    logger.error(f"Failed to add document to RAG: {txt_path}")
                    return jsonify({'error': 'Failed to add document to RAG'}), 500
            except Exception as e:
                logger.error(f"Exception while adding document: {e}")
                return jsonify({'error': 'Failed to add document to RAG'}), 500
            
            return jsonify({
                'status': 'success',
                'message': 'File uploaded and processed successfully',
                'filename': os.path.basename(txt_path)
            })
            
        except Exception as e:
            logger.error(f"Error processing uploaded file: {e}")
            return jsonify({'error': 'Failed to process file'}), 500
            
    except Exception as e:
        logger.error(f"Error handling file upload: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting RAG Backend Server...")
    print("📊 Initializing RAG system...")
    
    # Try to initialize RAG system on startup, but continue even if it fails
    try:
        initialize_rag()
        if rag_initialized:
            print("✅ RAG system initialized successfully!")
        else:
            print("⚠️ RAG system initialization failed, but server will start anyway")
            print("💡 You can try to reinitialize by calling /api/health endpoint")
    except Exception as e:
        print(f"❌ RAG initialization error: {e}")
        print("⚠️ Server will start but RAG functionality may be limited")
    
    print("✅ Server starting on http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)