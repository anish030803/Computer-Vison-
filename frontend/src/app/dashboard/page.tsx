"use client";

import { useState } from "react";
import { motion } from "motion/react";
import { DR_GRADES } from "@/lib/schemas";
import { useHealthCheck } from "@/lib/api";

export default function DashboardPage() {
  const health = useHealthCheck();

  return (
    <div className="max-w-6xl mx-auto px-6 py-12 space-y-8">
      <h1 className="text-3xl font-bold">Dashboard</h1>

      {/* Server Status */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-900 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-800"
      >
        <h2 className="text-lg font-semibold mb-4">Server Status</h2>
        <div className="flex items-center gap-3">
          <div
            className={`w-3 h-3 rounded-full ${
              health.data?.model_loaded ? "bg-green-500" : "bg-red-500"
            }`}
          />
          <span>
            {health.isLoading
              ? "Checking..."
              : health.data?.model_loaded
                ? `Model loaded: ${health.data.model_name}`
                : "Server not available"}
          </span>
        </div>
      </motion.div>

      {/* Severity Guide */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="bg-white dark:bg-gray-900 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-800"
      >
        <h2 className="text-lg font-semibold mb-4">DR Severity Grades</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {DR_GRADES.map((grade) => (
            <div
              key={grade.grade}
              className="p-4 rounded-xl border"
              style={{ borderColor: grade.color + "40" }}
            >
              <div className="flex items-center gap-2 mb-2">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: grade.color }}
                />
                <span className="font-semibold text-sm">{grade.label}</span>
              </div>
              <p className="text-xs text-gray-500">{grade.description}</p>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Placeholder for future analytics */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="bg-white dark:bg-gray-900 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-800"
      >
        <h2 className="text-lg font-semibold mb-4">Prediction History</h2>
        <p className="text-gray-500 text-sm">
          Prediction analytics will be displayed here after the system processes images.
          Upload images on the screening page to generate predictions.
        </p>
      </motion.div>
    </div>
  );
}
