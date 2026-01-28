import { useState, useRef, useEffect } from 'react';
import { Bot, User, FileText, BookmarkPlus, Highlighter } from 'lucide-react';
import { Message } from '../App';
import { Button } from './ui/button';

interface ChatMessageProps {
  message: Message;
  onSaveNote: (content: string, messageId?: string, page?: number) => void;
}

export function ChatMessage({ message, onSaveNote }: ChatMessageProps) {
  const isUser = message.role === 'user';
  const [selectedText, setSelectedText] = useState('');
  const [showHighlightButton, setShowHighlightButton] = useState(false);
  const [buttonPosition, setButtonPosition] = useState({ top: 0, left: 0 });
  const messageRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleSelectionChange = () => {
      const selection = window.getSelection();
      const text = selection?.toString().trim();

      if (text && messageRef.current?.contains(selection.anchorNode)) {
        setSelectedText(text);
        
        // Get the position of the selection
        const range = selection.getRangeAt(0);
        const rect = range.getBoundingClientRect();
        
        setButtonPosition({
          top: rect.top - 45,
          left: rect.left + rect.width / 2
        });
        setShowHighlightButton(true);
      } else if (!text) {
        setShowHighlightButton(false);
      }
    };

    document.addEventListener('selectionchange', handleSelectionChange);
    return () => document.removeEventListener('selectionchange', handleSelectionChange);
  }, []);

  const handleSaveHighlight = () => {
    if (selectedText) {
      onSaveNote(selectedText, message.id);
      setShowHighlightButton(false);
      setSelectedText('');
      window.getSelection()?.removeAllRanges();
    }
  };

  // Function to render text with highlights
  const renderHighlightedText = (text: string, highlights?: { start: number; end: number; text: string }[]) => {
    if (!highlights || highlights.length === 0) {
      return <span>{text}</span>;
    }

    // Sort highlights by start position
    const sortedHighlights = [...highlights].sort((a, b) => a.start - b.start);
    
    const parts = [];
    let lastIndex = 0;

    sortedHighlights.forEach((highlight, idx) => {
      // Add text before highlight
      if (highlight.start > lastIndex) {
        parts.push(
          <span key={`text-${idx}`}>{text.slice(lastIndex, highlight.start)}</span>
        );
      }

      // Add highlighted text
      parts.push(
        <mark
          key={`highlight-${idx}`}
          className="bg-yellow-200 rounded px-0.5"
        >
          {text.slice(highlight.start, highlight.end)}
        </mark>
      );

      lastIndex = highlight.end;
    });

    // Add remaining text
    if (lastIndex < text.length) {
      parts.push(
        <span key="text-end">{text.slice(lastIndex)}</span>
      );
    }

    return <>{parts}</>;
  };

  return (
    <div className={`flex gap-4 ${isUser ? 'flex-row-reverse' : ''}`}>
      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
        isUser ? 'bg-blue-600' : 'bg-slate-200'
      }`}>
        {isUser ? (
          <User className="w-4 h-4 text-white" />
        ) : (
          <Bot className="w-4 h-4 text-slate-700" />
        )}
      </div>
      
      <div className={`flex-1 max-w-3xl ${isUser ? 'flex flex-col items-end' : ''}`}>
        <div
          ref={messageRef}
          className={`rounded-lg p-4 ${
            isUser 
              ? 'bg-blue-600 text-white' 
              : 'bg-white border border-slate-200'
          }`}
        >
          <p className={`select-text ${isUser ? 'text-white' : 'text-slate-900'}`}>
            {renderHighlightedText(message.content, message.highlights)}
          </p>
          
          {!isUser && (
            <Button
              onClick={() => onSaveNote(message.content, message.id)}
              variant="ghost"
              size="sm"
              className="mt-2 text-slate-600 hover:text-blue-600"
            >
              <BookmarkPlus className="w-4 h-4 mr-1" />
              Save to Notes
            </Button>
          )}
        </div>
        
        {message.sources && message.sources.length > 0 && (
          <div className="mt-3 space-y-2">
            <p className="text-slate-500">Sources:</p>
            {message.sources.map((source, index) => (
              <div
                key={index}
                className="bg-slate-50 border border-slate-200 rounded-lg p-3 relative group"
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <FileText className="w-4 h-4 text-slate-500" />
                    <span className="text-slate-600">Page {source.page}</span>
                  </div>
                  <Button
                    onClick={() => onSaveNote(source.excerpt, message.id, source.page)}
                    variant="ghost"
                    size="sm"
                    className="opacity-0 group-hover:opacity-100 transition-opacity text-slate-600 hover:text-blue-600"
                  >
                    <BookmarkPlus className="w-3 h-3 mr-1" />
                    Save
                  </Button>
                </div>
                <p className="text-slate-700 italic select-text">"{source.excerpt}"</p>
              </div>
            ))}
          </div>
        )}
        
        <p className="text-slate-400 mt-2">
          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </p>
      </div>

      {/* Highlight Button Popup */}
      {showHighlightButton && !isUser && (
        <div
          className="fixed z-50"
          style={{
            top: `${buttonPosition.top}px`,
            left: `${buttonPosition.left}px`,
            transform: 'translateX(-50%)'
          }}
        >
          <Button
            onClick={handleSaveHighlight}
            className="bg-yellow-500 hover:bg-yellow-600 text-white shadow-lg"
            size="sm"
          >
            <Highlighter className="w-4 h-4 mr-1" />
            Save Highlight
          </Button>
        </div>
      )}
    </div>
  );
}
