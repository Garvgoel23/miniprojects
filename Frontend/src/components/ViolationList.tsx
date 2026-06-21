import type { ViolationModel } from "../types";
import ViolationCard from "./ViolationCard";
import { ListFilter } from "lucide-react";

interface ViolationListProps {
  violations: ViolationModel[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}

export default function ViolationList({ violations, selectedId, onSelect }: ViolationListProps) {
  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-title">
          <ListFilter size={18} style={{ display: 'inline', marginRight: '8px', verticalAlign: 'middle' }}/>
          Detected Violations ({violations.length})
        </div>
        {selectedId && (
          <button className="reset-button" onClick={() => onSelect(null)}>
            Reset View
          </button>
        )}
      </div>
      
      {violations.length === 0 ? (
        <div style={{ color: "var(--text-muted)", textAlign: "center", padding: "2rem 0" }}>
          No violations detected.
        </div>
      ) : (
        violations.map(v => (
          <ViolationCard 
            key={v.id} 
            violation={v} 
            isActive={selectedId === v.id}
            onClick={() => onSelect(selectedId === v.id ? null : v.id)}
          />
        ))
      )}
    </div>
  );
}
