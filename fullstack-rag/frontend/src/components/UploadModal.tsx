import { useState, useRef } from 'react';
import { Upload, X, Plus } from 'lucide-react';
import { Button } from './ui/button';

interface UploadModalProps {
  onFileUpload: (files: FileList, category?: string) => void;
  onClose: () => void;
  existingCategories: string[];
  currentCategory?: string;
}

export function UploadModal({ onFileUpload, onClose, existingCategories, currentCategory }: UploadModalProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null);
  const [category, setCategory] = useState<string>(currentCategory || '');
  const [newCategory, setNewCategory] = useState<string>('');
  const [showNewCategory, setShowNewCategory] = useState(false);

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      setSelectedFiles(files);
    }
  };

  const handleUpload = () => {
    if (selectedFiles) {
      const finalCategory = showNewCategory ? newCategory : (category || currentCategory);
      onFileUpload(selectedFiles, finalCategory);
      onClose();
    }
  };

  const uniqueCategories = Array.from(new Set(existingCategories.filter(c => c)));

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-slate-900">Upload PDF Document</h2>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-600"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Show current category if selected */}
        {currentCategory && (
          <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-xs text-blue-600 mb-1">Uploading to category:</p>
            <p className="text-blue-900">{currentCategory}</p>
          </div>
        )}

        {/* File Selection */}
        <div className="mb-4">
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            multiple
            onChange={handleFileChange}
            className="hidden"
          />
          <Button
            onClick={handleClick}
            variant="outline"
            className="w-full border-dashed border-2 h-24"
          >
            <div className="flex flex-col items-center">
              <Upload className="w-8 h-8 mb-2 text-slate-400" />
              <span className="text-slate-600">
                {selectedFiles ? `${selectedFiles.length} file(s) selected` : 'Click to select PDF files'}
              </span>
            </div>
          </Button>
        </div>

        {/* Category Selection - Only show if no category is selected */}
        {!currentCategory && (
          <div className="mb-6">
            <label className="block text-slate-700 mb-2">Category (optional)</label>
            
            {!showNewCategory ? (
              <div className="space-y-2">
                <select
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">Select a category</option>
                  {uniqueCategories.map((cat) => (
                    <option key={cat} value={cat}>
                      {cat}
                    </option>
                  ))}
                </select>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowNewCategory(true)}
                  className="w-full"
                >
                  <Plus className="w-4 h-4 mr-2" />
                  Create new category
                </Button>
              </div>
            ) : (
              <div className="space-y-2">
                <input
                  type="text"
                  value={newCategory}
                  onChange={(e) => setNewCategory(e.target.value)}
                  placeholder="Enter new category name"
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  autoFocus
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setShowNewCategory(false);
                    setNewCategory('');
                  }}
                  className="w-full"
                >
                  Use existing category
                </Button>
              </div>
            )}
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex gap-3">
          <Button
            onClick={handleUpload}
            disabled={!selectedFiles}
            className="flex-1 bg-blue-600 hover:bg-blue-700"
          >
            Upload
          </Button>
          <Button
            onClick={onClose}
            variant="outline"
          >
            Cancel
          </Button>
        </div>
      </div>
    </div>
  );
}