import type { ViolationModel } from "../types";
import { AlertCircle, CreditCard, ScanLine, Clock } from "lucide-react";

interface ViolationCardProps {
  violation: ViolationModel;
  isActive: boolean;
  onClick: () => void;
}

export default function ViolationCard({ violation, isActive, onClick }: ViolationCardProps) {
  return (
    <button 
      className={`violation-card ${isActive ? 'active' : ''}`}
      onClick={onClick}
    >
      <div className={`severity-indicator severity-${violation.severity}`} />
      
      <div className="card-header">
        <div className="card-type">
          {violation.type.replace(/_/g, ' ')}
        </div>
        <div className={`severity-badge ${violation.severity}`}>
          {violation.severity}
        </div>
      </div>
      
      <div className="card-details">
        <div className="detail-row">
          <ScanLine size={14} />
          <span>Vehicle: {violation.vehicle.class}</span>
        </div>
        
        {violation.plate && (
          <div className="detail-row">
            <CreditCard size={14} />
            <span className="plate-text">{violation.plate.text}</span>
            <span style={{ fontSize: "0.75rem", opacity: 0.7 }}>
              (OCR: {(violation.plate.ocr_confidence * 100).toFixed(0)}%)
            </span>
          </div>
        )}
        
        <div className="detail-row" style={{ marginTop: '0.25rem' }}>
          <AlertCircle size={14} />
          <span>Confidence: {(violation.confidence * 100).toFixed(1)}%</span>
        </div>
        
        {violation.scoring.temporal_conf > 0 && (
          <div className="detail-row">
            <Clock size={14} />
            <span>Temporal consistency: {(violation.scoring.temporal_conf * 100).toFixed(0)}%</span>
          </div>
        )}
      </div>
    </button>
  );
}
