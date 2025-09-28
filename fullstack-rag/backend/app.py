from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path to import our RAG modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Using optimized RAG system
logger.info("🔄 Using optimized RAG system")
from optimized_rag import MistralRAG

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
            logger.info("💡 Please ensure you have internet connectivity or try again later.")
            logger.info("💡 You can also run the demo version: python fullstack-rag/backend/app_demo.py")
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
        
        return jsonify({
            'response': result.get('response', 'No response generated'),
            'sources': result.get('sources', []),
            'status': 'success',
            'mode': result.get('mode', 'unknown')
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