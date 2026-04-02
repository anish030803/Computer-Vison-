"use client";

import { motion } from "motion/react";
import type { PredictionResponse } from "@/lib/schemas";
import { DR_GRADES } from "@/lib/schemas";
import { formatMs, formatPercentage } from "@/lib/utils";
import { ConfidenceChart } from "./confidence-chart";
import { GradCAMOverlay } from "./gradcam-overlay";

interface PredictionCardProps {
  result: PredictionResponse;
  imagePreview: string;
}

export function PredictionCard({ result, imagePreview }: PredictionCardProps) {
  const { prediction, gradcam, metadata } = result;
  const grade = DR_GRADES[prediction.class];

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="w-full max-w-4xl mx-auto space-y-6"
    >
      {/* Result Header */}
      <div className="text-center space-y-2">
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: "spring", stiffness: 200, delay: 0.2 }}
          className="inline-flex items-center gap-3 px-6 py-3 rounded-2xl"
          style={{ backgroundColor: `${grade.color}15`, border: `2px solid ${grade.color}` }}
        >
          <div className="w-4 h-4 rounded-full" style={{ backgroundColor: grade.color }} />
          <span className="text-xl font-bold" style={{ color: grade.color }}>
            {grade.label}
          </span>
          <span className="text-lg font-medium text-gray-600">
            ({formatPercentage(prediction.confidence)})
          </span>
        </motion.div>
        <p className="text-gray-500">{grade.description}</p>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Image + Grad-CAM */}
        <div className="space-y-4">
          <GradCAMOverlay
            imagePreview={imagePreview}
            gradcamHeatmap={gradcam?.heatmap ?? null}
          />
        </div>

        {/* Confidence Distribution */}
        <div className="bg-white dark:bg-gray-900 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-800">
          <ConfidenceChart
            probabilities={prediction.probabilities}
            predictedClass={prediction.class}
          />
        </div>
      </div>

      {/* Metadata */}
      <div className="flex justify-center gap-6 text-sm text-gray-400">
        <span>Model: {metadata.model}</span>
        <span>Latency: {formatMs(metadata.inference_time_ms)}</span>
        <span>Preprocessing: {metadata.preprocessing_applied}</span>
      </div>
    </motion.div>
  );
}
