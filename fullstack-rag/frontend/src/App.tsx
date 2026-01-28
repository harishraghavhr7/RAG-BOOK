import { useState } from 'react';
import { DocumentList } from './components/DocumentList';
import { ChatInterface } from './components/ChatInterface';
import { FileUpload } from './components/FileUpload';
import { LandingPage } from './components/LandingPage';
import { NotesView } from './components/NotesView';
import { NotesLandingPage } from './components/NotesLandingPage';
import { UploadModal } from './components/UploadModal';
import { ArrowLeft } from 'lucide-react';
import { Button } from './components/ui/button';

export interface Document {
  id: string;
  name: string;
  uploadedAt: Date;
  size: number;
  category?: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: { page: number; excerpt: string }[];
  highlights?: { start: number; end: number; text: string }[];
}

export interface Note {
  id: string;
  content: string;
  type: 'highlight' | 'custom';
  timestamp: Date;
  sourceMessageId?: string;
  page?: number;
  isTextHighlight?: boolean;
}

export default function App() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDocument, setSelectedDocument] = useState<string | null>(null);
  const [chatHistories, setChatHistories] = useState<Record<string, Message[]>>({});
  const [notesData, setNotesData] = useState<Record<string, Note[]>>({});
  const [showNotes, setShowNotes] = useState(false);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [viewMode, setViewMode] = useState<'landing' | 'chat' | 'notes'>('landing');
  const [selectedCategory, setSelectedCategory] = useState<string>('ALL');

  const handleFileUpload = (files: FileList, category?: string) => {
    // Upload files to backend. Use provided category or fallback.
    const uploadCategory = category || (selectedCategory !== 'ALL' ? selectedCategory : 'Uncategorized');

    Array.from(files).forEach(async (file) => {
      try {
        const form = new FormData();
        form.append('file', file, file.name);

        // POST to backend upload endpoint
        const resp = await fetch('/api/upload', {
          method: 'POST',
          body: form
        });

        if (!resp.ok) {
          console.error('Upload failed', await resp.text());
          return;
        }

        const data = await resp.json();

        // Use returned filename or fall back to original file name
        const displayName = data.filename || file.name;

        const doc: Document = {
          id: Math.random().toString(36).substr(2, 9),
          name: displayName,
          uploadedAt: new Date(),
          size: file.size,
          category: uploadCategory
        };

        setDocuments(prev => [...prev, doc]);

        // Initialize chat and notes for this document
        setChatHistories(prev => ({
          ...prev,
          [doc.id]: [{
            id: '1',
            role: 'assistant',
            content: `I've processed "${doc.name}". You can now ask me questions about the content of this document.`,
            timestamp: new Date()
          }]
        }));

        setNotesData(prev => ({
          ...prev,
          [doc.id]: []
        }));

        // Select the newly uploaded document and switch to chat view
        setSelectedDocument(doc.id);
        setViewMode('chat');
        setShowUploadModal(false);

      } catch (err) {
        console.error('Error uploading file', err);
      }
    });
  };

  const handleSelectDocument = (id: string) => {
    setSelectedDocument(id);
    setViewMode('chat');
  };

  const handleBackToLanding = () => {
    setViewMode('landing');
    setSelectedDocument(null);
  };

  const handleNavigateToNotes = () => {
    setViewMode('notes');
  };

  const handleNavigateToChat = (documentId: string, noteId?: string) => {
    setSelectedDocument(documentId);
    setViewMode('chat');
    // TODO: Optionally scroll to the specific message if noteId is provided
  };

  const handleCategorySelect = (category: string) => {
    setSelectedCategory(category);
  };

  const handleOpenUpload = () => {
    setShowUploadModal(true);
  };

  const handleSendMessage = (content: string) => {
    if (!selectedDocument) return;

    // Add user message locally
    const userMessage: Message = {
      id: Math.random().toString(36).substr(2, 9),
      role: 'user',
      content,
      timestamp: new Date()
    };

    setChatHistories(prev => ({
      ...prev,
      [selectedDocument]: [...(prev[selectedDocument] || []), userMessage]
    }));

    // Send the query to the backend chat endpoint
    (async () => {
      try {
        const resp = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: content })
        });

        if (!resp.ok) {
          const txt = await resp.text();
          console.error('Chat API error', txt);
          return;
        }

        const data = await resp.json();
        const assistantText = data.response || 'No response from server';
        const assistantMessage: Message = {
          id: Math.random().toString(36).substr(2, 9),
          role: 'assistant',
          content: assistantText,
          timestamp: new Date(),
          sources: data.sources || []
        };

        setChatHistories(prev => ({
          ...prev,
          [selectedDocument]: [...(prev[selectedDocument] || []), assistantMessage]
        }));

      } catch (err) {
        console.error('Error calling chat API', err);
      }
    })();
  };

  const handleDeleteDocument = (id: string) => {
    setDocuments(documents.filter(doc => doc.id !== id));
    
    // Remove chat history for deleted document
    const newChatHistories = { ...chatHistories };
    delete newChatHistories[id];
    setChatHistories(newChatHistories);
    
    // Remove notes for deleted document
    const newNotesData = { ...notesData };
    delete newNotesData[id];
    setNotesData(newNotesData);
    
    if (selectedDocument === id) {
      setSelectedDocument(null);
    }
  };

  const handleSaveNote = (content: string, messageId?: string, page?: number) => {
    if (!selectedDocument) return;

    const newNote: Note = {
      id: Math.random().toString(36).substr(2, 9),
      content,
      type: messageId ? 'highlight' : 'custom',
      timestamp: new Date(),
      sourceMessageId: messageId,
      page,
      isTextHighlight: !!messageId && !page
    };

    const currentNotes = notesData[selectedDocument] || [];
    setNotesData({
      ...notesData,
      [selectedDocument]: [...currentNotes, newNote]
    });

    // If this is a text highlight, update the message to mark the highlight
    if (messageId && !page) {
      const currentMessages = chatHistories[selectedDocument] || [];
      const updatedMessages = currentMessages.map(msg => {
        if (msg.id === messageId) {
          const messageContent = msg.content;
          const start = messageContent.indexOf(content);
          if (start !== -1) {
            const end = start + content.length;
            const highlights = msg.highlights || [];
            return {
              ...msg,
              highlights: [...highlights, { start, end, text: content }]
            };
          }
        }
        return msg;
      });

      setChatHistories({
        ...chatHistories,
        [selectedDocument]: updatedMessages
      });
    }
  };

  const handleDeleteNote = (noteId: string) => {
    if (!selectedDocument) return;

    const currentNotes = notesData[selectedDocument] || [];
    setNotesData({
      ...notesData,
      [selectedDocument]: currentNotes.filter(note => note.id !== noteId)
    });
  };

  // Get messages for selected document
  const currentMessages = selectedDocument ? (chatHistories[selectedDocument] || []) : [];
  const currentNotes = selectedDocument ? (notesData[selectedDocument] || []) : [];

  // Show landing page if in landing mode
  if (viewMode === 'landing') {
    return (
      <>
        <NotesLandingPage
          notes={notesData}
          documents={documents}
          onNavigateToChat={handleNavigateToChat}
          onOpenUpload={handleOpenUpload}
          selectedCategory={selectedCategory}
          onCategorySelect={handleCategorySelect}
        />
        
        {/* Upload Modal */}
        {showUploadModal && (
          <UploadModal
            onFileUpload={handleFileUpload}
            onClose={() => setShowUploadModal(false)}
            existingCategories={documents.map(d => d.category || 'Uncategorized')}
            currentCategory={selectedCategory !== 'ALL' ? selectedCategory : undefined}
          />
        )}
      </>
    );
  }

  // Show notes view
  if (viewMode === 'notes') {
    return (
      <NotesView
        notes={notesData}
        documents={documents}
        onBackToLanding={handleBackToLanding}
        onNavigateToChat={handleNavigateToChat}
      />
    );
  }

  // Show chat interface
  return (
    <div className="flex h-screen bg-gradient-to-br from-slate-50 to-slate-100 overflow-hidden">
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        <ChatInterface
          messages={currentMessages}
          onSendMessage={handleSendMessage}
          selectedDocument={documents.find(d => d.id === selectedDocument)}
          onSaveNote={handleSaveNote}
          showNotes={showNotes}
          onToggleNotes={() => setShowNotes(!showNotes)}
          notes={currentNotes}
          onDeleteNote={handleDeleteNote}
          onBackToLanding={handleBackToLanding}
        />
      </div>
    </div>
  );
}