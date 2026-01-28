import { useState } from 'react';
import { Trash2, FileText, Plus, Bookmark, Highlighter } from 'lucide-react';
import { Button } from './ui/button';
import { Note } from '../App';

interface NotesPanelProps {
  notes: Note[];
  onSaveNote: (content: string) => void;
  onDeleteNote: (noteId: string) => void;
}

export function NotesPanel({ notes, onSaveNote, onDeleteNote }: NotesPanelProps) {
  const [newNoteInput, setNewNoteInput] = useState('');
  const [showAddNote, setShowAddNote] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (newNoteInput.trim()) {
      onSaveNote(newNoteInput.trim());
      setNewNoteInput('');
      setShowAddNote(false);
    }
  };

  const formatDate = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    
    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (minutes < 1440) return `${Math.floor(minutes / 60)}h ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className="w-80 bg-white flex flex-col h-full">
      <div className="p-4 border-b border-slate-200">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-slate-900">Document Notes</h3>
          <Button
            onClick={() => setShowAddNote(!showAddNote)}
            variant="ghost"
            size="sm"
            className="text-blue-600 hover:text-blue-700"
          >
            <Plus className="w-4 h-4" />
          </Button>
        </div>
        <p className="text-slate-500">{notes.length} note{notes.length !== 1 ? 's' : ''}</p>
      </div>

      {showAddNote && (
        <div className="p-4 border-b border-slate-200 bg-slate-50">
          <form onSubmit={handleSubmit}>
            <textarea
              value={newNoteInput}
              onChange={(e) => setNewNoteInput(e.target.value)}
              placeholder="Write a custom note..."
              className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              rows={3}
              autoFocus
            />
            <div className="flex gap-2 mt-2">
              <Button
                type="submit"
                disabled={!newNoteInput.trim()}
                className="bg-blue-600 hover:bg-blue-700 flex-1"
                size="sm"
              >
                Save Note
              </Button>
              <Button
                type="button"
                onClick={() => {
                  setShowAddNote(false);
                  setNewNoteInput('');
                }}
                variant="outline"
                size="sm"
              >
                Cancel
              </Button>
            </div>
          </form>
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {notes.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center px-4">
            <Bookmark className="w-12 h-12 text-slate-300 mb-3" />
            <p className="text-slate-500 mb-2">No notes yet</p>
            <p className="text-slate-400">
              Save important excerpts from chat responses or add custom notes
            </p>
          </div>
        ) : (
          notes.map(note => (
            <div
              key={note.id}
              className={`border rounded-lg p-3 group hover:border-slate-300 transition-colors ${
                note.isTextHighlight 
                  ? 'bg-yellow-50 border-yellow-200' 
                  : 'bg-slate-50 border-slate-200'
              }`}
            >
              <div className="flex items-start justify-between gap-2 mb-2">
                <div className="flex items-center gap-2 min-w-0">
                  {note.isTextHighlight ? (
                    <Highlighter className="w-4 h-4 text-yellow-600 flex-shrink-0" />
                  ) : note.type === 'highlight' ? (
                    <Bookmark className="w-4 h-4 text-blue-600 flex-shrink-0" />
                  ) : (
                    <FileText className="w-4 h-4 text-slate-500 flex-shrink-0" />
                  )}
                  <span className="text-slate-500 truncate">
                    {note.isTextHighlight ? 'Text highlight' : note.type === 'highlight' ? 'From chat' : 'Custom note'}
                    {note.page && ` • Page ${note.page}`}
                  </span>
                </div>
                <button
                  onClick={() => onDeleteNote(note.id)}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-100 rounded transition-opacity flex-shrink-0"
                >
                  <Trash2 className="w-3 h-3 text-red-600" />
                </button>
              </div>
              <p className="text-slate-700">{note.content}</p>
              <p className="text-slate-400 mt-2">{formatDate(note.timestamp)}</p>
            </div>
          ))
        )}
      </div>
    </div>
  );
}