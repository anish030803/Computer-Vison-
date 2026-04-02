"use client";

import { useCallback, useState } from "react";
import { motion } from "motion/react";

interface ImageUploaderProps {
  onImageSelect: (file: File, preview: string) => void;
  isLoading?: boolean;
}

export function ImageUploader({ onImageSelect, isLoading }: ImageUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) return;

      const url = URL.createObjectURL(file);
      setPreview(url);
      onImageSelect(file, url);
    },
    [onImageSelect]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full max-w-xl mx-auto"
    >
      <label
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={`
          relative flex flex-col items-center justify-center
          w-full h-72 border-2 border-dashed rounded-2xl
          cursor-pointer transition-all duration-200
          ${isDragging
            ? "border-blue-500 bg-blue-50 dark:bg-blue-950/20"
            : "border-gray-300 dark:border-gray-700 hover:border-gray-400 dark:hover:border-gray-600"
          }
          ${isLoading ? "opacity-60 pointer-events-none" : ""}
        `}
      >
        <input
          type="file"
          accept="image/*"
          onChange={handleChange}
          className="sr-only"
          disabled={isLoading}
        />

        {preview ? (
          <img
            src={preview}
            alt="Selected fundus image"
            className="max-h-60 rounded-xl object-contain"
          />
        ) : (
          <div className="flex flex-col items-center gap-3 text-gray-500">
            <svg className="w-12 h-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
            </svg>
            <p className="text-lg font-medium">Drop a fundus image here</p>
            <p className="text-sm">or click to browse</p>
            <p className="text-xs text-gray-400 mt-2">PNG, JPG, TIFF up to 10MB</p>
          </div>
        )}
      </label>
    </motion.div>
  );
}
