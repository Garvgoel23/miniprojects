import { useState } from "react";
import type { FrameResultModel } from "../types";
import ImageCanvas from "./ImageCanvas";
import ViolationList from "./ViolationList";
import { ArrowLeft } from "lucide-react";

interface ResultsDashboardProps {
  result: FrameResultModel;
  mediaUrl: string;
  isVideo: boolean;
  onReset: () => void;
}

export default function ResultsDashboard({ result, mediaUrl, isVideo, onReset }: ResultsDashboardProps) {
  const [selectedViolationId, setSelectedViolationId] = useState<string | null>(null);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
      <div style={{ padding: '1rem 1.5rem', display: 'flex', alignItems: 'center', borderBottom: '1px solid var(--border-color)' }}>
        <button 
          onClick={onReset}
          style={{ 
            background: 'none', border: 'none', color: 'var(--text-muted)', 
            display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer',
            fontSize: '0.875rem'
          }}
        >
          <ArrowLeft size={16} /> Analyze New Media
        </button>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: '1rem', fontSize: '0.875rem' }}>
          <span style={{ color: 'var(--text-muted)' }}>Analyzed Resolution:</span>
          <span>{result.dimensions.width}x{result.dimensions.height}</span>
        </div>
      </div>
      
      <div className="results-dashboard">
        <ImageCanvas 
          result={result} 
          mediaUrl={mediaUrl} 
          isVideo={isVideo} 
          selectedViolationId={selectedViolationId} 
        />
        <ViolationList 
          violations={result.violations} 
          selectedId={selectedViolationId} 
          onSelect={setSelectedViolationId} 
        />
      </div>
    </div>
  );
}
