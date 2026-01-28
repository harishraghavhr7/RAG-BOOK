import { ArrowLeft, Search, FileText, Clock, MessageSquare } from 'lucide-react';
import { Button } from './ui/button';
import { Note, Document } from '../App';

interface NotesViewProps {
  notes: Record<string, Note[]>;
  documents: Document[];
  onBackToLanding: () => void;
  onNavigateToChat: (documentId: string, noteId?: string) => void;
}

export function NotesView({ notes, documents, onBackToLanding, onNavigateToChat }: NotesViewProps) {
  // Flatten all notes with document info
  const allNotes = Object.entries(notes).flatMap(([docId, docNotes]) => 
    docNotes.map(note => ({
      ...note,
      documentId: docId,
      documentName: documents.find(d => d.id === docId)?.name || 'Unknown Document'
    }))
  ).sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

  const formatDate = (date: Date) => {
    const d = new Date(date);
    return d.toLocaleDateString('en-US', { year: 'numeric', month: '2-digit', day: '2-digit' });
  };

  const getPreview = (content: string, maxLength: number = 150) => {
    if (content.length <= maxLength) return content;
    return content.substring(0, maxLength) + '...';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between py-4">
            <div className="flex items-center gap-3">
              <Button
                onClick={onBackToLanding}
                variant="ghost"
                size="sm"
                className="p-2"
              >
                <ArrowLeft className="w-5 h-5" />
              </Button>
              <div>
                <h1 className="text-slate-900">All Highlights</h1>
                <p className="text-slate-500">{allNotes.length} saved highlights</p>
              </div>
            </div>
            <Button variant="ghost" size="sm" className="p-2">
              <Search className="w-5 h-5" />
            </Button>
          </div>

          {/* Tabs */}
          <div className="flex gap-2 pb-3 overflow-x-auto">
            <button className="px-5 py-2 bg-blue-600 text-white rounded-full whitespace-nowrap">
              All
            </button>
            {documents.map(doc => {
              const docNoteCount = notes[doc.id]?.length || 0;
              if (docNoteCount === 0) return null;
              return (
                <button
                  key={doc.id}
                  className="px-5 py-2 bg-slate-200 text-slate-700 rounded-full hover:bg-slate-300 transition-colors whitespace-nowrap"
                >
                  {doc.name.replace('.pdf', '')}
                </button>
              );
            })}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {allNotes.length === 0 ? (
          /* Empty State */
          <div className="flex flex-col items-center justify-center py-20">
            <div className="w-20 h-20 bg-yellow-100 rounded-full flex items-center justify-center mb-6">
              <FileText className="w-10 h-10 text-yellow-600" />
            </div>
            <h2 className="text-slate-900 mb-3">No highlights yet</h2>
            <p className="text-slate-500 mb-8 text-center max-w-md">
              Start highlighting important text in your chat conversations. All your highlights will appear here.
            </p>
            <Button
              onClick={onBackToLanding}
              className="bg-blue-600 hover:bg-blue-700"
            >
              Go to Documents
            </Button>
          </div>
        ) : (
          /* Notes Grid */
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {allNotes.map((note) => (
              <button
                key={note.id}
                onClick={() => onNavigateToChat(note.documentId, note.id)}
                className="bg-gradient-to-br from-yellow-100 to-yellow-200 rounded-2xl p-5 text-left hover:shadow-lg hover:scale-105 transition-all duration-200 group"
              >
                <div className="space-y-3">
                  {/* Document Name as Title */}
                  <h3 className="text-slate-900 line-clamp-1 group-hover:text-blue-700 transition-colors">
                    {note.documentName.replace('.pdf', '')}
                  </h3>

                  {/* Highlight Content */}
                  <p className="text-slate-700 whitespace-pre-wrap line-clamp-6">
                    {getPreview(note.content, 200)}
                  </p>

                  {/* Footer with Date and Badge */}
                  <div className="flex items-center justify-between pt-2">
                    <div className="flex items-center gap-1 text-slate-600">
                      <Clock className="w-3 h-3" />
                      <span className="text-xs">{formatDate(note.timestamp)}</span>
                    </div>
                    {note.isTextHighlight && (
                      <div className="flex items-center gap-1 text-xs text-blue-700 bg-blue-100 px-2 py-1 rounded-full">
                        <MessageSquare className="w-3 h-3" />
                        Chat
                      </div>
                    )}
                  </div>
                </div>
              </button>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
