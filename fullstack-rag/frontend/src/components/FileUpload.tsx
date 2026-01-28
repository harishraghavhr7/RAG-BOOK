import { useRef, useState } from 'react';
import { Upload } from 'lucide-react';
import { Button } from './ui/button';

interface FileUploadProps {
  onFileUpload: (files: FileList, category?: string) => void;
  existingCategories?: string[];
}

export function FileUpload({ onFileUpload, existingCategories }: FileUploadProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [category, setCategory] = useState<string>('');

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onFileUpload(files, category);
      // Reset input
      e.target.value = '';
    }
  };

  return (
    <div>
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf"
        multiple
        onChange={handleFileChange}
        className="hidden"
      />
      <div className="flex items-center">
        <Button
          onClick={handleClick}
          className="w-full bg-blue-600 hover:bg-blue-700"
        >
          <Upload className="w-4 h-4 mr-2" />
          Upload PDF
        </Button>
        <select
          value={category}
          onChange={(e) => setCategory(e.target.value)}
          className="ml-2 p-2 border rounded"
        >
          <option value="">Select Category</option>
          {existingCategories?.map((cat) => (
            <option key={cat} value={cat}>
              {cat}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}