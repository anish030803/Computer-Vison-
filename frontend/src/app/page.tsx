"use client";

import { useCallback, useState } from "react";
import { ImageUploader } from "@/components/upload/image-uploader";
import { PredictionCard } from "@/components/prediction/prediction-card";
import { fileToBase64, usePrediction } from "@/lib/api";
import type { PredictionResponse } from "@/lib/schemas";

export default function HomePage() {
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const prediction = usePrediction();

  const handleImageSelect = useCallback(
    async (file: File, preview: string) => {
      setImagePreview(preview);
      setResult(null);

      try {
        const base64 = await fileToBase64(file);
        const response = await prediction.mutateAsync({
          imageBase64: base64,
          returnGradcam: true,
        });
        setResult(response);
      } catch {
        // Error is handled by mutation state
      }
    },
    [prediction]
  );

  return (
    <div className="max-w-6xl mx-auto px-6 py-12 space-y-12">
      {/* Hero */}
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold tracking-tight">
          Diabetic Retinopathy Screening
        </h1>
        <p className="text-lg text-gray-500 max-w-2xl mx-auto">
          Upload a retinal fundus image for AI-powered DR severity grading.
          Results include confidence scores and Grad-CAM visualizations.
        </p>
      </div>

      {/* Upload */}
      <ImageUploader
        onImageSelect={handleImageSelect}
        isLoading={prediction.isPending}
      />

      {/* Loading State */}
      {prediction.isPending && (
        <div className="text-center space-y-3">
          <div className="inline-block w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin" />
          <p className="text-gray-500">Analyzing fundus image...</p>
        </div>
      )}

      {/* Error */}
      {prediction.isError && (
        <div className="max-w-xl mx-auto p-4 bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800 rounded-xl text-red-700 dark:text-red-400 text-center">
          <p className="font-medium">Analysis failed</p>
          <p className="text-sm mt-1">{prediction.error.message}</p>
          <p className="text-xs mt-2 text-red-400">
            Make sure the inference server is running at localhost:8000
          </p>
        </div>
      )}

      {/* Results */}
      {result && imagePreview && (
        <PredictionCard result={result} imagePreview={imagePreview} />
      )}
    </div>
  );
}
