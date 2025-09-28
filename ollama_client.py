"""
Ollama Client for RAG System
Handles communication with Ollama API
"""

import ollama
import logging
from typing import Optional, Dict, Any
import json
import time

logger = logging.getLogger(__name__)

class OllamaConversation:
    """Manages conversation with Ollama models"""
    
    def __init__(self, model: str = "mistral", timeout: float = 30.0):
        """
        Initialize Ollama conversation client
        
        Args:
            model: The Ollama model to use (default: "mistral")
            timeout: Timeout for requests in seconds (default: 30.0)
        """
        self.model = model
        self.timeout = timeout
        self.conversation_history = []
        self.client = ollama.Client(timeout=timeout)
        
        logger.info(f"Initialized Ollama client with model: {model} (timeout: {timeout}s)")
    
    def add_message(self, role: str, content: str):
        """Add a message to conversation history"""
        self.conversation_history.append({
            'role': role,
            'content': content
        })
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def chat(self, message: str, context: Optional[str] = None, system_prompt: Optional[str] = None) -> str:
        """
        Send a chat message to Ollama
        
        Args:
            message: User message
            context: Optional context for RAG
            system_prompt: Optional system prompt
            
        Returns:
            Model response
        """
        try:
            # Prepare messages
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({
                    'role': 'system',
                    'content': system_prompt
                })
            
            # Add conversation history
            messages.extend(self.conversation_history)
            
            # Add context if provided
            if context:
                context_message = f"Context: {context}\n\nQuestion: {message}"
                messages.append({
                    'role': 'user',
                    'content': context_message
                })
            else:
                messages.append({
                    'role': 'user',
                    'content': message
                })
            
            # Send request to Ollama with timeout
            response = self.client.chat(
                model=self.model,
                messages=messages,
                stream=False
            )
            
            assistant_response = response['message']['content']
            
            # Add to conversation history
            self.add_message('user', message)
            self.add_message('assistant', assistant_response)
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"Error in Ollama chat: {e}")
            return f"Error: Could not get response from Ollama - {str(e)}"

def test_ollama_connection(model: str = "mistral") -> bool:
    """
    Test connection to Ollama
    
    Args:
        model: Model to test (default: "mistral")
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        client = ollama.Client()
        
        # Try to list models first
        models = client.list()
        available_models = [m.get('name', m.get('model', '')) for m in models.get('models', [])]
        
        # Check if the specified model is available
        if not any(model in m for m in available_models):
            logger.warning(f"Model '{model}' not found in available models: {available_models}")
            return False
        
        # Test a simple chat
        response = client.chat(
            model=model,
            messages=[
                {
                    'role': 'user',
                    'content': 'Hello, are you working?'
                }
            ],
            stream=False
        )
        
        if response and 'message' in response:
            logger.info(f"✅ Ollama connection test successful with model: {model}")
            return True
        else:
            logger.error("❌ Ollama connection test failed - no valid response")
            return False
            
    except Exception as e:
        logger.error(f"❌ Ollama connection test failed: {e}")
        return False

def query_ollama(context: str, question: str, conversation: Optional[OllamaConversation] = None) -> str:
    """
    Query Ollama with context and question
    
    Args:
        context: Context information for RAG
        question: User question
        conversation: Optional conversation object
        
    Returns:
        Model response
    """
    try:
        if conversation:
            # Use existing conversation
            system_prompt = """You are a helpful AI assistant that answers questions based on provided context. 
            Use the context to provide accurate and informative answers. 
            If the context doesn't contain relevant information, say so clearly.
            Be concise but thorough in your responses."""
            
            return conversation.chat(question, context=context, system_prompt=system_prompt)
        else:
            # Create new conversation for single query
            client = ollama.Client()
            
            system_prompt = """You are a helpful AI assistant that answers questions based on provided context. 
            Use the context to provide accurate and informative answers. 
            If the context doesn't contain relevant information, say so clearly.
            Be concise but thorough in your responses."""
            
            messages = [
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': f"Context: {context}\n\nQuestion: {question}"
                }
            ]
            
            response = client.chat(
                model="mistral",
                messages=messages,
                stream=False
            )
            
            return response['message']['content']
            
    except Exception as e:
        logger.error(f"Error querying Ollama: {e}")
        return f"Error: Could not get response from Ollama - {str(e)}"

def list_available_models() -> list:
    """
    List available Ollama models
    
    Returns:
        List of available model names
    """
    try:
        client = ollama.Client()
        models = client.list()
        return [model.get('name', model.get('model', '')) for model in models.get('models', [])]
    except Exception as e:
        logger.error(f"Error listing Ollama models: {e}")
        return []

def pull_model(model_name: str) -> bool:
    """
    Pull a model from Ollama registry
    
    Args:
        model_name: Name of the model to pull
        
    Returns:
        True if successful, False otherwise
    """
    try:
        client = ollama.Client()
        logger.info(f"Pulling model: {model_name}")
        
        # Pull the model
        client.pull(model_name)
        logger.info(f"✅ Successfully pulled model: {model_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error pulling model {model_name}: {e}")
        return False

# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Ollama client...")
    
    # Test connection
    if test_ollama_connection():
        print("✅ Ollama connection successful!")
        
        # Test conversation
        conv = OllamaConversation()
        response = conv.chat("What is Python?")
        print(f"Response: {response}")
        
        # Test query with context
        context = "Python is a high-level programming language known for its simplicity and readability."
        question = "What are the main characteristics of this programming language?"
        response = query_ollama(context, question)
        print(f"RAG Response: {response}")
        
    else:
        print("❌ Ollama connection failed!")
        print("Make sure Ollama is running and the mistral model is available.")
        print("Try running: ollama pull mistral")