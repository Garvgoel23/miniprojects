import type { HealthResponse, ImageAnalysisResponse, VideoAnalysisResponse } from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export const api = {
  checkHealth: async (): Promise<HealthResponse> => {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }
    return response.json();
  },

  analyzeImage: async (file: File): Promise<ImageAnalysisResponse> => {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_BASE_URL}/api/analyze/image`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error || `Image analysis failed: ${response.statusText}`);
    }

    return response.json();
  },

  analyzeVideo: async (file: File): Promise<VideoAnalysisResponse> => {
    const formData = new FormData();
    formData.append("file", file);
    // Use a larger stride for frontend video demo to keep it faster
    formData.append("stride", "15"); 

    const response = await fetch(`${API_BASE_URL}/api/analyze/video`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error || `Video analysis failed: ${response.statusText}`);
    }

    return response.json();
  },
};
