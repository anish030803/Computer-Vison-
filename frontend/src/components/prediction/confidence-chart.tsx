"use client";

import { motion } from "motion/react";
import { DR_GRADES } from "@/lib/schemas";
import { formatPercentage } from "@/lib/utils";

interface ConfidenceChartProps {
  probabilities: Record<string, number>;
  predictedClass: number;
}

export function ConfidenceChart({ probabilities, predictedClass }: ConfidenceChartProps) {
  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide">
        Confidence Distribution
      </h3>
      {DR_GRADES.map((grade) => {
        const prob = probabilities[grade.label] ?? 0;
        const isPredicted = grade.grade === predictedClass;

        return (
          <div key={grade.grade} className="space-y-1">
            <div className="flex justify-between text-sm">
              <span className={isPredicted ? "font-semibold" : "text-gray-600 dark:text-gray-400"}>
                {grade.label}
              </span>
              <span className={isPredicted ? "font-semibold" : "text-gray-500"}>
                {formatPercentage(prob)}
              </span>
            </div>
            <div className="h-3 bg-gray-100 dark:bg-gray-800 rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${prob * 100}%` }}
                transition={{ duration: 0.6, ease: "easeOut", delay: grade.grade * 0.1 }}
                className="h-full rounded-full"
                style={{ backgroundColor: grade.color }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
