"use client";

import { useState } from "react";

interface GradCAMOverlayProps {
  imagePreview: string;
  gradcamHeatmap: string | null;
}

export function GradCAMOverlay({ imagePreview, gradcamHeatmap }: GradCAMOverlayProps) {
  const [showOverlay, setShowOverlay] = useState(false);

  return (
    <div className="space-y-3">
      <div className="relative rounded-2xl overflow-hidden bg-black">
        <img
          src={showOverlay && gradcamHeatmap ? `data:image/png;base64,${gradcamHeatmap}` : imagePreview}
          alt="Fundus image"
          className="w-full h-auto"
        />
      </div>

      {gradcamHeatmap && (
        <div className="flex justify-center">
          <button
            onClick={() => setShowOverlay(!showOverlay)}
            className={`
              px-4 py-2 text-sm font-medium rounded-lg transition-colors
              ${showOverlay
                ? "bg-blue-600 text-white"
                : "bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
              }
            `}
          >
            {showOverlay ? "Hide" : "Show"} Grad-CAM Overlay
          </button>
        </div>
      )}
    </div>
  );
}
