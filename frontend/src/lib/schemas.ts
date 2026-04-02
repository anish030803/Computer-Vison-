import { z } from "zod/v4";

export const predictionRequestSchema = z.object({
  image: z.string().min(1, "Image is required"),
  return_gradcam: z.boolean().default(false),
});

export type PredictionRequest = z.infer<typeof predictionRequestSchema>;

export const predictionResultSchema = z.object({
  class: z.number().int().min(0).max(4),
  label: z.string(),
  confidence: z.number().min(0).max(1),
  probabilities: z.record(z.string(), z.number()),
});

export type PredictionResult = z.infer<typeof predictionResultSchema>;

export const gradcamResultSchema = z.object({
  heatmap: z.string(),
  attention_regions: z.array(z.string()),
});

export type GradCAMResult = z.infer<typeof gradcamResultSchema>;

export const inferenceMetadataSchema = z.object({
  model: z.string(),
  inference_time_ms: z.number(),
  preprocessing_applied: z.string(),
});

export type InferenceMetadata = z.infer<typeof inferenceMetadataSchema>;

export const predictionResponseSchema = z.object({
  prediction: predictionResultSchema,
  gradcam: gradcamResultSchema.nullable().optional(),
  metadata: inferenceMetadataSchema,
});

export type PredictionResponse = z.infer<typeof predictionResponseSchema>;

export const DR_GRADES = [
  { grade: 0, label: "No DR", color: "#22c55e", description: "No visible retinopathy" },
  { grade: 1, label: "Mild NPDR", color: "#eab308", description: "Microaneurysms only" },
  { grade: 2, label: "Moderate NPDR", color: "#f97316", description: "More than just microaneurysms" },
  { grade: 3, label: "Severe NPDR", color: "#ef4444", description: "Extensive intraretinal hemorrhages" },
  { grade: 4, label: "Proliferative DR", color: "#7c3aed", description: "Neovascularization or vitreous hemorrhage" },
] as const;
