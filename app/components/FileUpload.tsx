'use client';

import { useCallback, useState } from 'react';
import { Upload } from 'lucide-react';
import { Button } from './ui/button';
import { formatBytes } from '@/lib/utils';

const ACCEPTED = ['application/json', 'application/zip', 'application/x-zip-compressed', 'application/octet-stream'];

interface FileUploadProps {
  onUploadComplete: (payload: { fileName: string; publicUrl: string; size: number }) => void;
}

export default function FileUpload({ onUploadComplete }: FileUploadProps) {
  const [dragging, setDragging] = useState(false);
  const [selected, setSelected] = useState<File | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  const handleFile = useCallback(
    async (file: File) => {
      setError(null);
      if (!ACCEPTED.includes(file.type) && !file.name.endsWith('.jsonl')) {
        setError('Only .json, .jsonl, or .zip files are supported.');
        return;
      }

      setSelected(file);
      setUploading(true);
      setProgress(10);

      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        setUploading(false);
        setError('Upload failed. Please try again.');
        return;
      }

      const data = await res.json();
      setProgress(100);
      setUploading(false);
      onUploadComplete({ fileName: file.name, publicUrl: data.publicUrl, size: file.size });
    },
    [onUploadComplete]
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        const file = e.dataTransfer.files?.[0];
        if (file) {
          void handleFile(file);
        }
      }}
      className={`flex flex-col items-center justify-center rounded-2xl border-2 border-dashed ${dragging ? 'border-emerald-400 bg-emerald-500/10' : 'border-slate-800 bg-slate-900/40'} p-6 text-center transition`}
    >
      <Upload className="mb-3 h-10 w-10 text-emerald-300" />
      <p className="text-sm text-slate-300">Drop your dataset or click to upload (.json/.jsonl/.zip)</p>
      <input
        className="hidden"
        id="dataset"
        type="file"
        accept=".json,.jsonl,.zip,application/json,application/zip"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) {
            void handleFile(file);
          }
        }}
      />
      <Button className="mt-4" asChild disabled={uploading}>
        <label htmlFor="dataset">{uploading ? 'Uploading…' : 'Select file'}</label>
      </Button>
      {selected ? (
        <p className="mt-3 text-xs text-slate-400">
          {selected.name} · {formatBytes(selected.size)} · {progress}%
        </p>
      ) : null}
      {error ? <p className="mt-2 text-xs text-red-400">{error}</p> : null}
    </div>
  );
}
