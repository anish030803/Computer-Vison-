"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import type { PredictionResponse } from "./schemas";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function predictImage(
  imageBase64: string,
  returnGradcam: boolean = false
): Promise<PredictionResponse> {
  const response = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image: imageBase64,
      return_gradcam: returnGradcam,
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Prediction failed: ${error}`);
  }

  return response.json();
}

async function checkHealth(): Promise<{ status: string; model_loaded: boolean; model_name: string | null }> {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) {
    throw new Error("Server health check failed");
  }
  return response.json();
}

export function usePrediction() {
  return useMutation({
    mutationFn: ({
      imageBase64,
      returnGradcam,
    }: {
      imageBase64: string;
      returnGradcam: boolean;
    }) => predictImage(imageBase64, returnGradcam),
  });
}

export function useHealthCheck() {
  return useQuery({
    queryKey: ["health"],
    queryFn: checkHealth,
    refetchInterval: 30000,
    retry: 1,
  });
}

export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // Remove data URL prefix (data:image/...;base64,)
      const base64 = result.split(",")[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}
