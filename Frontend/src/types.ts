export interface DimensionsModel {
  width: number;
  height: number;
}

export interface BBoxModel {
  class: string;
  bbox: [number, number, number, number];
  confidence: number;
}

export interface VehicleModel {
  class: string;
  bbox: [number, number, number, number];
  confidence: number;
}

export interface PlateModel {
  text: string;
  raw_text: string;
  bbox: [number, number, number, number];
  ocr_confidence: number;
}

export interface ScoringModel {
  detection_conf: number;
  spatial_conf: number;
  temporal_conf: number;
  composite: number;
}

export interface ViolationModel {
  id: string;
  type: string;
  confidence: number;
  severity: "HIGH" | "MEDIUM" | "LOW";
  vehicle: VehicleModel;
  evidence: BBoxModel[];
  plate?: PlateModel;
  scoring: ScoringModel;
}

export interface SummaryModel {
  total_violations: number;
  by_type: Record<string, number>;
  by_severity: Record<string, number>;
}

export interface FrameResultModel {
  frame_id: number;
  timestamp: string | null;
  dimensions: DimensionsModel;
  violations: ViolationModel[];
  summary: SummaryModel;
  error?: string;
}

export interface ImageAnalysisResponse {
  report: FrameResultModel;
}

export interface VideoAnalysisResponse {
  total_frames_analyzed: number;
  frames: FrameResultModel[];
  summary: SummaryModel;
}

export interface HealthResponse {
  status: string;
  device: string;
  models: any;
  analyzers: string[];
  timestamp: string;
}
