import { useState, useRef, useEffect } from 'react';
import { Send, FileText, StickyNote, Menu } from 'lucide-react';
import { Button } from './ui/button';
import { Message, Document, Note } from '../App';
import { ChatMessage } from './ChatMessage';
import { NotesPanel } from './NotesPanel';

interface ChatInterfaceProps {
  messages: Message[];
  onSendMessage: (content: string) => void;
  selectedDocument?: Document;
  notes: Note[];
  onSaveNote: (content: string, messageId?: string, page?: number) => void;
  onDeleteNote: (noteId: string) => void;
  showNotes: boolean;
  onToggleNotes: () => void;
  onBackToLanding: () => void;
}

export function ChatInterface({ messages, onSendMessage, selectedDocument, notes, onSaveNote, onDeleteNote, showNotes, onToggleNotes, onBackToLanding }: ChatInterfaceProps) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && selectedDocument) {
      onSendMessage(input.trim());
      setInput('');
    }
  };

  if (!selectedDocument) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center max-w-md">
          <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <FileText className="w-8 h-8 text-blue-600" />
          </div>
          <h2 className="text-slate-900 mb-2">No Document Selected</h2>
          <p className="text-slate-500">
            Upload a PDF document and select it from the sidebar to start chatting and extracting information.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="bg-white border-b border-slate-200 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button
              onClick={onBackToLanding}
              variant="ghost"
              size="sm"
              className="p-2"
            >
              <Menu className="w-5 h-5" />
            </Button>
            <div className="p-2 bg-blue-100 rounded">
              <FileText className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <h2 className="text-slate-900">{selectedDocument.name}</h2>
              <p className="text-slate-500">Ask questions about this document</p>
            </div>
          </div>
          <Button
            onClick={onToggleNotes}
            variant={showNotes ? "default" : "outline"}
            className={showNotes ? "bg-blue-600 hover:bg-blue-700" : ""}
          >
            <StickyNote className="w-4 h-4 mr-2" />
            Notes {notes.length > 0 && `(${notes.length})`}
          </Button>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Messages */}
        <div className={`flex-1 overflow-y-auto p-6 space-y-6 ${showNotes ? 'border-r border-slate-200' : ''}`}>
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center max-w-md">
                <p className="text-slate-500 mb-4">Start a conversation by asking questions about your document</p>
                <div className="space-y-2">
                  <button
                    onClick={() => onSendMessage("What is this document about?")}
                    className="block w-full p-3 text-left bg-white border border-slate-200 rounded-lg hover:border-blue-300 hover:bg-blue-50 transition-colors"
                  >
                    <p className="text-slate-700">What is this document about?</p>
                  </button>
                  <button
                    onClick={() => onSendMessage("Summarize the key points")}
                    className="block w-full p-3 text-left bg-white border border-slate-200 rounded-lg hover:border-blue-300 hover:bg-blue-50 transition-colors"
                  >
                    <p className="text-slate-700">Summarize the key points</p>
                  </button>
                  <button
                    onClick={() => onSendMessage("What are the main conclusions?")}
                    className="block w-full p-3 text-left bg-white border border-slate-200 rounded-lg hover:border-blue-300 hover:bg-blue-50 transition-colors"
                  >
                    <p className="text-slate-700">What are the main conclusions?</p>
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <>
              {messages.map(message => (
                <ChatMessage 
                  key={message.id} 
                  message={message} 
                  onSaveNote={onSaveNote}
                />
              ))}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Notes Panel */}
        {showNotes && (
          <NotesPanel
            notes={notes}
            onSaveNote={onSaveNote}
            onDeleteNote={onDeleteNote}
          />
        )}
      </div>

      {/* Input */}
      <div className="bg-white border-t border-slate-200 p-4">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about the document..."
            className="flex-1 px-4 py-3 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <Button
            type="submit"
            disabled={!input.trim()}
            className="bg-blue-600 hover:bg-blue-700 px-6"
          >
            <Send className="w-4 h-4" />
          </Button>
        </form>
      </div>
    </div>
  );
}