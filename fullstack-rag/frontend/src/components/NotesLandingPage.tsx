import { Search, Menu, Plus } from 'lucide-react';
import { Note, Document } from '../App';
import { useState } from 'react';
import svgPaths from '../imports/svg-ypbpvcg4ow';

interface NotesLandingPageProps {
  notes: Record<string, Note[]>;
  documents: Document[];
  onNavigateToChat: (documentId: string, noteId?: string) => void;
  onOpenUpload: () => void;
  onOpenMenu?: () => void;
  selectedCategory?: string;
  onCategorySelect?: (category: string) => void;
}

export function NotesLandingPage({ notes, documents, onNavigateToChat, onOpenUpload, onOpenMenu, selectedCategory, onCategorySelect }: NotesLandingPageProps) {
  const [selectedFilter, setSelectedFilter] = useState<string>(selectedCategory || 'ALL');

  // Get unique categories from documents
  const categories = Array.from(new Set(documents.map(d => d.category || 'Uncategorized')));

  // Get documents sorted by upload date
  const allDocuments = documents.sort((a, b) => 
    new Date(b.uploadedAt).getTime() - new Date(a.uploadedAt).getTime()
  );

  // Filter documents based on selected category
  const filteredDocuments = selectedFilter === 'ALL' 
    ? allDocuments 
    : allDocuments.filter(doc => (doc.category || 'Uncategorized') === selectedFilter);

  const handleCategoryClick = (category: string) => {
    setSelectedFilter(category);
    onCategorySelect?.(category);
  };

  const formatDate = (date: Date) => {
    const d = new Date(date);
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  return (
    <div className="bg-white relative min-h-screen">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-6">
        <button onClick={onOpenMenu} className="p-2">
          <div className="h-9 w-12 flex items-center justify-center">
            <svg className="w-9 h-[18px]" fill="none" preserveAspectRatio="none" viewBox="0 0 36 18">
              <path d={svgPaths.p3985a626} fill="#1D1B20" />
            </svg>
          </div>
        </button>
        <button className="p-2">
          <div className="h-9 w-8 flex items-center justify-center">
            <svg className="w-6 h-7" fill="none" preserveAspectRatio="none" viewBox="0 0 25 27">
              <path d={svgPaths.p194dda00} fill="#1D1B20" />
            </svg>
          </div>
        </button>
      </div>

      {/* Filter Tabs */}
      <div className="flex gap-3 px-5 mb-6 overflow-x-auto pb-2">
        <button
          onClick={() => setSelectedFilter('ALL')}
          className={`px-6 py-2 rounded-full text-xs whitespace-nowrap transition-colors ${
            selectedFilter === 'ALL' 
              ? 'bg-slate-300 text-black' 
              : 'bg-slate-200 text-black hover:bg-slate-300'
          }`}
        >
          ALL
        </button>
        {categories.map(category => (
          <button
            key={category}
            onClick={() => handleCategoryClick(category)}
            className={`px-6 py-2 rounded-full text-xs whitespace-nowrap transition-colors ${
              selectedFilter === category 
                ? 'bg-slate-300 text-black' 
                : 'bg-slate-200 text-black hover:bg-slate-300'
            }`}
          >
            {category}
          </button>
        ))}
      </div>

      {/* PDF Documents Grid */}
      <div className="px-5 pb-24">
        {filteredDocuments.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 text-center">
            <p className="text-slate-500 text-sm mb-4">No documents yet</p>
            <p className="text-slate-400 text-xs max-w-xs">
              Upload PDF documents to start chatting with your content
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-4">
            {filteredDocuments.map((doc) => (
              <button
                key={doc.id}
                onClick={() => onNavigateToChat(doc.id)}
                className="bg-[#d9d9d9] rounded-[30px] p-5 text-left hover:bg-[#c9c9c9] transition-all duration-200 hover:shadow-md active:scale-95"
              >
                <p className="text-xs text-black mb-3 truncate">
                  {doc.name}
                </p>
                <div className="text-xs text-black leading-relaxed space-y-1">
                  <p className="text-[10px] text-slate-600">{formatDate(doc.uploadedAt)}</p>
                  <p className="text-[10px] text-slate-600">{formatFileSize(doc.size)}</p>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Floating Add Button */}
      <button
        onClick={onOpenUpload}
        className="fixed bottom-6 right-6 w-[53px] h-[46px] bg-[#d9d9d9] rounded-full hover:bg-[#c9c9c9] transition-all active:scale-95 shadow-lg flex items-center justify-center"
      >
        <svg className="w-8 h-6" fill="none" preserveAspectRatio="none" viewBox="0 0 31 24">
          <path d={svgPaths.p299a5400} fill="#1D1B20" />
        </svg>
      </button>
    </div>
  );
}