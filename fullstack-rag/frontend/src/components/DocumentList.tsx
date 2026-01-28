import { FileText, Trash2, MessageSquare } from 'lucide-react';
import { Document } from '../App';
import { Message } from '../App';

interface DocumentListProps {
  documents: Document[];
  selectedDocument: string | null;
  onSelectDocument: (id: string) => void;
  onDeleteDocument: (id: string) => void;
  chatHistories: Record<string, Message[]>;
}

export function DocumentList({
  documents,
  selectedDocument,
  onSelectDocument,
  onDeleteDocument,
  chatHistories
}: DocumentListProps) {
  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
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

  if (documents.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="text-center">
          <FileText className="w-12 h-12 text-slate-300 mx-auto mb-3" />
          <p className="text-slate-500">No documents uploaded yet</p>
          <p className="text-slate-400 mt-1">Upload a PDF to get started</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="p-4">
        <p className="text-slate-500 mb-3">{documents.length} document{documents.length !== 1 ? 's' : ''}</p>
        <div className="space-y-2">
          {documents.map(doc => (
            <div
              key={doc.id}
              onClick={() => onSelectDocument(doc.id)}
              className={`group relative p-3 rounded-lg border cursor-pointer transition-all ${
                selectedDocument === doc.id
                  ? 'bg-blue-50 border-blue-200'
                  : 'bg-white border-slate-200 hover:border-blue-200 hover:bg-slate-50'
              }`}
            >
              <div className="flex items-start gap-3">
                <div className={`p-2 rounded ${
                  selectedDocument === doc.id ? 'bg-blue-100' : 'bg-slate-100'
                }`}>
                  <FileText className={`w-4 h-4 ${
                    selectedDocument === doc.id ? 'text-blue-600' : 'text-slate-600'
                  }`} />
                </div>
                <div className="flex-1 min-w-0">
                  <p className={`truncate ${
                    selectedDocument === doc.id ? 'text-blue-900' : 'text-slate-900'
                  }`}>
                    {doc.name}
                  </p>
                  <div className="flex items-center gap-2 mt-1">
                    <p className="text-slate-500">
                      {formatFileSize(doc.size)} · {formatDate(doc.uploadedAt)}
                    </p>
                    {chatHistories[doc.id] && chatHistories[doc.id].length > 1 && (
                      <div className="flex items-center gap-1 text-slate-500">
                        <MessageSquare className="w-3 h-3" />
                        <span className="text-xs">{chatHistories[doc.id].length - 1}</span>
                      </div>
                    )}
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteDocument(doc.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-100 rounded transition-opacity"
                >
                  <Trash2 className="w-4 h-4 text-red-600" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}