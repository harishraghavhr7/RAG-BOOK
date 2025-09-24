import React, { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, FileText, Search, AlertCircle, Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import './App.css';

const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? '/api' 
  : 'http://localhost:5000/api';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [documentsInfo, setDocumentsInfo] = useState(null);
  const [activeTab, setActiveTab] = useState('chat');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    checkConnection();
    loadDocuments();
    
    // Retry connection check every 5 seconds if disconnected
    const interval = setInterval(() => {
      if (!isConnected) {
        checkConnection();
      }
    }, 5000);
    
    return () => clearInterval(interval);
  }, [isConnected]);

  const checkConnection = async () => {
    try {
      console.log('Checking connection to:', `${API_BASE_URL}/health`);
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Health check response:', data);
      setIsConnected(data.status === 'healthy');
    } catch (error) {
      console.error('Connection check failed:', error);
      setIsConnected(false);
    }
  };

  const loadDocuments = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/documents`);
      const data = await response.json();
      if (data.status === 'success') {
        setDocumentsInfo(data);
      }
    } catch (error) {
      console.error('Failed to load documents info:', error);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = inputMessage.trim();
    setInputMessage('');
    setMessages(prev => [...prev, { type: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userMessage }),
      });

      const data = await response.json();

      if (data.status === 'success') {
        setMessages(prev => [...prev, { type: 'bot', content: data.response }]);
      } else {
        setMessages(prev => [...prev, { 
          type: 'error', 
          content: data.error || 'Failed to get response' 
        }]);
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      setMessages(prev => [...prev, { 
        type: 'error', 
        content: 'Failed to connect to the server. Please try again.' 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const performSearch = async () => {
    if (!searchQuery.trim() || isSearching) return;

    setIsSearching(true);
    try {
      const response = await fetch(`${API_BASE_URL}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: searchQuery }),
      });

      const data = await response.json();

      if (data.status === 'success') {
        setSearchResults(data.results);
      } else {
        setSearchResults([]);
      }
    } catch (error) {
      console.error('Search failed:', error);
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  const handleSearchKeyPress = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      performSearch();
    }
  };

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <div className="header-content">
            <div className="title-section">
              <Bot className="logo" />
              <h1>RAG Chatbot</h1>
            </div>
            <div className="status-section">
              <div className={`status ${isConnected ? 'connected' : 'disconnected'}`}>
                <div className="status-dot"></div>
                <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
              </div>
            </div>
          </div>
          
          <nav className="nav-tabs">
            <button 
              className={`tab ${activeTab === 'chat' ? 'active' : ''}`}
              onClick={() => setActiveTab('chat')}
            >
              <Bot size={16} />
              Chat
            </button>
            <button 
              className={`tab ${activeTab === 'search' ? 'active' : ''}`}
              onClick={() => setActiveTab('search')}
            >
              <Search size={16} />
              Search
            </button>
            <button 
              className={`tab ${activeTab === 'documents' ? 'active' : ''}`}
              onClick={() => setActiveTab('documents')}
            >
              <FileText size={16} />
              Documents
            </button>
          </nav>
        </header>

        <main className="main-content">
          {activeTab === 'chat' && (
            <div className="chat-section">
              <div className="messages-container">
                {messages.length === 0 && (
                  <div className="welcome-message">
                    <Bot size={48} className="welcome-icon" />
                    <h2>Welcome to RAG Chatbot!</h2>
                    <p>Ask me anything about the documents I have loaded. I'll help you find the information you need.</p>
                    <div className="example-questions">
                      <h3>Try asking:</h3>
                      <div className="example-item">"What is a cell?"</div>
                      <div className="example-item">"Explain photosynthesis"</div>
                      <div className="example-item">"Tell me about DNA"</div>
                    </div>
                  </div>
                )}
                
                {messages.map((message, index) => (
                  <div key={index} className={`message ${message.type}`}>
                    <div className="message-icon">
                      {message.type === 'user' ? (
                        <User size={20} />
                      ) : message.type === 'bot' ? (
                        <Bot size={20} />
                      ) : (
                        <AlertCircle size={20} />
                      )}
                    </div>
                    <div className="message-content">
                      {message.type === 'bot' ? (
                        <ReactMarkdown>{message.content}</ReactMarkdown>
                      ) : (
                        <p>{message.content}</p>
                      )}
                    </div>
                  </div>
                ))}
                
                {isLoading && (
                  <div className="message bot">
                    <div className="message-icon">
                      <Loader2 size={20} className="spinning" />
                    </div>
                    <div className="message-content">
                      <p>Thinking...</p>
                    </div>
                  </div>
                )}
                
                <div ref={messagesEndRef} />
              </div>

              <div className="input-section">
                <div className="input-container">
                  <textarea
                    ref={inputRef}
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Type your message here..."
                    rows="1"
                    className="message-input"
                    disabled={!isConnected}
                  />
                  <button 
                    onClick={sendMessage} 
                    disabled={!inputMessage.trim() || isLoading || !isConnected}
                    className="send-button"
                  >
                    <Send size={20} />
                  </button>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'search' && (
            <div className="search-section">
              <div className="search-header">
                <h2>Search Documents</h2>
                <p>Search for exact text matches in the loaded documents</p>
              </div>
              
              <div className="search-input-container">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyPress={handleSearchKeyPress}
                  placeholder="Enter search terms..."
                  className="search-input"
                  disabled={!isConnected}
                />
                <button 
                  onClick={performSearch}
                  disabled={!searchQuery.trim() || isSearching || !isConnected}
                  className="search-button"
                >
                  {isSearching ? <Loader2 size={20} className="spinning" /> : <Search size={20} />}
                </button>
              </div>

              <div className="search-results">
                {searchResults.length > 0 && (
                  <div className="results-header">
                    <h3>Found {searchResults.length} matches</h3>
                  </div>
                )}
                
                {searchResults.map((result, index) => (
                  <div key={index} className="search-result">
                    <div className="result-header">
                      <strong>{result.metadata.chapter}</strong>
                      <span className="result-section">{result.metadata.section}</span>
                    </div>
                    <div className="result-context">
                      ...{result.context}...
                    </div>
                  </div>
                ))}
                
                {searchQuery && searchResults.length === 0 && !isSearching && (
                  <div className="no-results">
                    <Search size={48} className="no-results-icon" />
                    <h3>No matches found</h3>
                    <p>Try different search terms or check your spelling</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'documents' && (
            <div className="documents-section">
              <div className="documents-header">
                <h2>Loaded Documents</h2>
                <p>Information about the documents currently available for querying</p>
              </div>
              
              {documentsInfo ? (
                <div className="documents-info">
                  <div className="documents-stats">
                    <div className="stat-item">
                      <strong>{documentsInfo.total_chunks}</strong>
                      <span>Total Chunks</span>
                    </div>
                    <div className="stat-item">
                      <strong>{Object.keys(documentsInfo.documents).length}</strong>
                      <span>Documents</span>
                    </div>
                  </div>
                  
                  <div className="documents-list">
                    {Object.entries(documentsInfo.documents).map(([docName, docInfo], index) => (
                      <div key={index} className="document-item">
                        <div className="document-header">
                          <FileText size={20} />
                          <h3>{docName}</h3>
                          <span className="chunk-count">{docInfo.chunks} chunks</span>
                        </div>
                        <div className="document-sections">
                          <strong>Sections:</strong>
                          <div className="sections-list">
                            {docInfo.sections.map((section, idx) => (
                              <span key={idx} className="section-tag">{section}</span>
                            ))}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="loading-documents">
                  <Loader2 size={48} className="spinning" />
                  <p>Loading document information...</p>
                </div>
              )}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;