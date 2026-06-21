import { useState } from "react";
import { UploadCloud, FileVideo, Image as ImageIcon, Activity } from "lucide-react";
import { api } from "../api";
import type { FrameResultModel } from "../types";

interface UploadSectionProps {
  onResults: (results: FrameResultModel, fileUrl: string, isVideo: boolean) => void;
}

export default function UploadSection({ onResults }: UploadSectionProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Create a local URL for the file to render
    const fileUrl = URL.createObjectURL(file);
    setPreviewUrl(fileUrl);
    setLoading(true);
    setError(null);
    
    // Reset input so the same file can be selected again
    e.target.value = "";

    const isVideo = file.type.startsWith("video/");

    try {
      if (isVideo) {
        const response = await api.analyzeVideo(file);
        const bestFrame = response.frames.find(f => f.violations.length > 0) || response.frames[0];
        if (bestFrame) {
          onResults(bestFrame, fileUrl, true);
        } else {
          setError("No frames could be analyzed.");
          setLoading(false);
        }
      } else {
        const response = await api.analyzeImage(file);
        onResults(response.report, fileUrl, false);
      }
    } catch (err: any) {
      setError(err.message || "Failed to analyze file");
      setLoading(false);
    }
  };

  if (loading && previewUrl) {
    return (
      <div className="upload-section">
        <div style={{ position: 'relative', borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--border-color)', backgroundColor: 'var(--bg-dark)' }}>
          <img 
            src={previewUrl} 
            alt="Processing Preview" 
            style={{ display: 'block', maxWidth: '100%', maxHeight: '60vh', opacity: 0.4, filter: 'blur(4px)', objectFit: 'contain' }} 
          />
          <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', textAlign: 'center', width: '100%' }}>
            <div className="spinner" style={{ marginBottom: '1rem', display: 'flex', justifyContent: 'center' }}>
              <Activity size={48} color="var(--accent)" />
            </div>
            <h2 style={{ fontSize: '1.5rem', fontWeight: 600, marginBottom: '0.5rem', textShadow: '0 2px 4px rgba(0,0,0,0.8)' }}>
              Analyzing Image...
            </h2>
            <p style={{ color: 'var(--text-main)', opacity: 0.9, textShadow: '0 2px 4px rgba(0,0,0,0.8)' }}>
              Running dual-model parallel inference
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="upload-section">
      <div className="upload-card">
        <UploadCloud size={48} className="upload-icon" />
        <h2 className="upload-title">Upload Traffic Media</h2>
        <p className="upload-subtitle">Supported formats: JPG, PNG, MP4, MOV</p>
        
        {error && (
          <div style={{ color: "var(--severity-high)", marginBottom: "1rem" }}>
            {error}
          </div>
        )}

        <label className="upload-button">
          Select File
          <input 
            type="file" 
            accept="image/*,video/*" 
            style={{ display: "none" }} 
            onChange={handleFileUpload}
          />
        </label>
        
        <div style={{ display: 'flex', justifyContent: 'center', gap: '2rem', marginTop: '2rem', color: 'var(--text-muted)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <ImageIcon size={18} /> Images
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <FileVideo size={18} /> Videos
          </div>
        </div>
      </div>
    </div>
  );
}


