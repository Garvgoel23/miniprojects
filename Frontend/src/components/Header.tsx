import { useEffect, useState } from "react";
import { Activity, ShieldAlert, ShieldCheck } from "lucide-react";
import { api } from "../api";
import type { HealthResponse } from "../types";

export default function Header() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.checkHealth()
      .then(setHealth)
      .catch(err => setError(err.message));
  }, []);

  return (
    <header className="header">
      <div className="header-title">
        <Activity size={24} color="var(--accent)" />
        <span>Traffic Violation Dashboard</span>
      </div>
      
      <div className={`health-badge ${error ? "error" : ""}`}>
        <div className="status-dot" />
        {error ? (
          <>
            <ShieldAlert size={14} />
            <span>Backend Offline</span>
          </>
        ) : health ? (
          <>
            <ShieldCheck size={14} />
            <span>Systems Online ({health.device})</span>
          </>
        ) : (
          <span>Checking status...</span>
        )}
      </div>
    </header>
  );
}
