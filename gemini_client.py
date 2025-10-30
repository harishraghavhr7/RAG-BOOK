"""
Google Gemini Client for RAG System
Handles communication with Google Gemini API
"""

import google.generativeai as genai
import logging
from typing import Optional, Dict, Any, List
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class GeminiConversation:
    """Manages conversation with Google Gemini models"""
    
    def __init__(self, model: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        """
        Initialize Gemini conversation client
        
        Args:
            model: The Gemini model to use (default: "gemini-1.5-flash")
            api_key: Google API key (if not provided, will look for GOOGLE_API_KEY env var)
        """
        self.model = model
        self.conversation_history = []
        
        # Configure API key
        if api_key:
            genai.configure(api_key=api_key)
        else:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
            genai.configure(api_key=api_key)
        
        # Initialize the model with enhanced generation settings
        try:
            # Configure generation parameters for longer, more detailed responses
            generation_config = genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=2048,  # Increased from default for longer responses
                temperature=0.3,  # Lower temperature for more focused educational content
                top_p=0.8,
                top_k=20
            )
            
            self.client = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config
            )
            logger.info(f"Initialized Gemini client with model: {model} (max_tokens: 2048)")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise
    
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
        Send a chat message to Gemini
        
        Args:
            message: User message
            context: Optional context for RAG
            system_prompt: Optional system prompt
            
        Returns:
            Model response
        """
        try:
            # Prepare the prompt
            full_prompt = ""
            
            # Add system prompt if provided
            if system_prompt:
                full_prompt += f"System: {system_prompt}\n\n"
            
            # Add conversation history
            if self.conversation_history:
                for msg in self.conversation_history[-10:]:  # Keep last 10 messages for context
                    full_prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
                full_prompt += "\n"
            
            # Add context if provided
            if context:
                full_prompt += f"Context: {context}\n\n"
            
            # Add the current message
            full_prompt += f"User: {message}\n\nAssistant:"
            
            # Generate response
            response = self.client.generate_content(full_prompt)
            
            if response.text:
                assistant_response = response.text.strip()
                
                # Add to conversation history
                self.add_message('user', message)
                self.add_message('assistant', assistant_response)
                
                return assistant_response
            else:
                logger.error("Empty response from Gemini")
                return "Error: Received empty response from Gemini"
            
        except Exception as e:
            logger.error(f"Error in Gemini chat: {e}")
            return f"Error: Could not get response from Gemini - {str(e)}"

def test_gemini_connection(model: str = "gemini-2.5-flash", api_key: Optional[str] = None) -> bool:
    """
    Test connection to Google Gemini
    
    Args:
        model: Model to test (default: "gemini-1.5-flash")
        api_key: Google API key
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        # Configure API key
        if api_key:
            genai.configure(api_key=api_key)
        else:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                logger.error("Google API key is required")
                return False
            genai.configure(api_key=api_key)
        
        # Initialize model and test
        client = genai.GenerativeModel(model)
        
        # Test with a simple prompt
        response = client.generate_content("Hello, are you working? Just reply with 'Yes' if you can understand this.")
        
        if response.text:
            logger.info(f"✅ Gemini connection test successful with model: {model}")
            return True
        else:
            logger.error("❌ Gemini connection test failed - no valid response")
            return False
            
    except Exception as e:
        logger.error(f"❌ Gemini connection test failed: {e}")
        return False

def query_gemini(context: str, question: str, conversation: Optional[GeminiConversation] = None, api_key: Optional[str] = None) -> str:
    """
    Query Gemini with context and question
    
    Args:
        context: Context information for RAG
        question: User question
        conversation: Optional conversation object
        api_key: Google API key
        
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
            if api_key:
                genai.configure(api_key=api_key)
            else:
                api_key = os.getenv('GOOGLE_API_KEY')
                if not api_key:
                    return "Error: Google API key is required"
                genai.configure(api_key=api_key)
            
            client = genai.GenerativeModel("gemini-2.5-flash")
            
            system_prompt = """You are a helpful AI assistant that answers questions based on provided context. 
            Use the context to provide accurate and informative answers. 
            If the context doesn't contain relevant information, say so clearly.
            Be concise but thorough in your responses."""
            
            full_prompt = f"{system_prompt}\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
            
            response = client.generate_content(full_prompt)
            
            if response.text:
                return response.text.strip()
            else:
                return "Error: Received empty response from Gemini"
            
    except Exception as e:
        logger.error(f"Error querying Gemini: {e}")
        return f"Error: Could not get response from Gemini - {str(e)}"

def list_available_models() -> List[str]:
    """
    List available Gemini models
    
    Returns:
        List of available model names
    """
    try:
        models = genai.list_models()
        return [model.name for model in models if 'generateContent' in model.supported_generation_methods]
    except Exception as e:
        logger.error(f"Error listing Gemini models: {e}")
        return ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-pro"]  # Default fallback

# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Gemini client...")
    print("Make sure you have set the GOOGLE_API_KEY environment variable")
    
    # Test connection
    if test_gemini_connection():
        print("✅ Gemini connection successful!")
        
        # Test conversation
        try:
            conv = GeminiConversation()
            response = conv.chat("What is Python?")
            print(f"Response: {response}")
            
            # Test query with context
            context = "Python is a high-level programming language known for its simplicity and readability."
            question = "What are the main characteristics of this programming language?"
            response = query_gemini(context, question)
            print(f"RAG Response: {response}")
        except Exception as e:
            print(f"Error during testing: {e}")
        
    else:
        print("❌ Gemini connection failed!")
        print("Make sure you have:")
        print("1. Set the GOOGLE_API_KEY environment variable")
        print("2. Have internet connectivity")
        print("3. Valid Google Cloud API credentials")