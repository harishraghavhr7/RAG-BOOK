import { FileText, Clock, MessageSquare, Upload, Plus, StickyNote } from 'lucide-react';
import { Document } from '../App';
import { Button } from './ui/button';

interface LandingPageProps {
  documents: Document[];
  onSelectDocument: (id: string) => void;
  onOpenUpload: () => void;
  onViewAllNotes?: () => void;
  chatHistories: Record<string, any[]>;
}

export function LandingPage({ documents, onSelectDocument, onOpenUpload, onViewAllNotes, chatHistories }: LandingPageProps) {
  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  const formatDate = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - new Date(date).getTime();
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (hours < 1) return 'Just now';
    if (hours < 24) return `${hours}h ago`;
    if (days === 1) return 'Yesterday';
    if (days < 7) return `${days} days ago`;
    return new Date(date).toLocaleDateString();
  };

  const getMessageCount = (docId: string) => {
    const messages = chatHistories[docId] || [];
    return messages.filter(m => m.role === 'user').length;
  };

  // Sort documents by most recent
  const recentDocuments = [...documents].sort((a, b) => 
    new Date(b.uploadedAt).getTime() - new Date(a.uploadedAt).getTime()
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header */}
      <header className="bg-white border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-slate-900 mb-1">RAG Document Assistant</h1>
              <p className="text-slate-500">Chat with your PDF documents using AI</p>
            </div>
            <Button
              onClick={onOpenUpload}
              className="bg-blue-600 hover:bg-blue-700"
            >
              <Plus className="w-4 h-4 mr-2" />
              Upload PDF
            </Button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {documents.length === 0 ? (
          /* Empty State */
          <div className="flex flex-col items-center justify-center py-20">
            <div className="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mb-6">
              <Upload className="w-10 h-10 text-blue-600" />
            </div>
            <h2 className="text-slate-900 mb-3">No documents yet</h2>
            <p className="text-slate-500 mb-8 text-center max-w-md">
              Upload your first PDF document to start chatting and extracting insights with AI-powered assistance.
            </p>
            <Button
              onClick={onOpenUpload}
              className="bg-blue-600 hover:bg-blue-700"
              size="lg"
            >
              <Plus className="w-5 h-5 mr-2" />
              Upload Your First PDF
            </Button>

            {/* Features */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-16 max-w-4xl">
              <div className="bg-white rounded-lg p-6 border border-slate-200">
                <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
                  <MessageSquare className="w-6 h-6 text-blue-600" />
                </div>
                <h3 className="text-slate-900 mb-2">Chat with Documents</h3>
                <p className="text-slate-500">
                  Ask questions and get instant answers from your PDF documents using AI.
                </p>
              </div>
              <div className="bg-white rounded-lg p-6 border border-slate-200">
                <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mb-4">
                  <FileText className="w-6 h-6 text-green-600" />
                </div>
                <h3 className="text-slate-900 mb-2">Smart Highlights</h3>
                <p className="text-slate-500">
                  Highlight important text and save it as notes for easy reference later.
                </p>
              </div>
              <div className="bg-white rounded-lg p-6 border border-slate-200">
                <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mb-4">
                  <Clock className="w-6 h-6 text-purple-600" />
                </div>
                <h3 className="text-slate-900 mb-2">Separate Histories</h3>
                <p className="text-slate-500">
                  Each document maintains its own chat history and notes for better organization.
                </p>
              </div>
            </div>
          </div>
        ) : (
          /* Documents Grid */
          <>
            <div className="flex items-center justify-between mb-8">
              <div>
                <h2 className="text-slate-900 mb-1">Recent Documents</h2>
                <p className="text-slate-500">{documents.length} {documents.length === 1 ? 'document' : 'documents'} uploaded</p>
              </div>
              {onViewAllNotes && (
                <Button
                  onClick={onViewAllNotes}
                  className="bg-blue-600 hover:bg-blue-700"
                >
                  <StickyNote className="w-4 h-4 mr-2" />
                  View All Notes
                </Button>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {recentDocuments.map((doc) => {
                const messageCount = getMessageCount(doc.id);
                return (
                  <button
                    key={doc.id}
                    onClick={() => onSelectDocument(doc.id)}
                    className="bg-white rounded-lg border border-slate-200 hover:border-blue-300 hover:shadow-lg transition-all p-6 text-left group"
                  >
                    <div className="flex items-start gap-4 mb-4">
                      <div className="p-3 bg-blue-100 rounded-lg group-hover:bg-blue-200 transition-colors">
                        <FileText className="w-6 h-6 text-blue-600" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <h3 className="text-slate-900 mb-1 truncate group-hover:text-blue-600 transition-colors">
                          {doc.name}
                        </h3>
                        <p className="text-slate-500">{formatFileSize(doc.size)}</p>
                      </div>
                    </div>

                    <div className="flex items-center gap-4 text-slate-500 pt-4 border-t border-slate-100">
                      <div className="flex items-center gap-1">
                        <Clock className="w-4 h-4" />
                        <span>{formatDate(doc.uploadedAt)}</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <MessageSquare className="w-4 h-4" />
                        <span>{messageCount} {messageCount === 1 ? 'message' : 'messages'}</span>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </>
        )}
      </main>
    </div>
  );
}