import React, { useState } from 'react';
import './index.css';

const API_BASE = 'http://localhost:8000';

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<any[] | null>(null); // array of violations

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (!selected) return;

    // Reset state
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResults(null);
    setError(null);
    setLoading(true);

    const formData = new FormData();
    formData.append('file', selected);

    const isVideo = selected.type.startsWith('video/');

    try {
      if (isVideo) {
        const res = await fetch(`${API_BASE}/api/analyze/video`, {
          method: 'POST',
          body: formData,
        });
        if (!res.ok) throw new Error(`API Error: ${res.statusText}`);
        const data = await res.json();
        
        // Extract violations from all frames
        const allViolations: any[] = [];
        data.frames.forEach((f: any) => {
          if (f.violations) {
            allViolations.push(...f.violations);
          }
        });
        setResults(allViolations);
      } else {
        const res = await fetch(`${API_BASE}/api/analyze/image`, {
          method: 'POST',
          body: formData,
        });
        if (!res.ok) throw new Error(`API Error: ${res.statusText}`);
        const data = await res.json();
        setResults(data.report.violations || []);
      }
    } catch (err: any) {
      setError(err.message || "Failed to connect to backend.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResults(null);
    setError(null);
  };

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', padding: '2rem', fontFamily: 'sans-serif', color: '#fff', backgroundColor: '#111', minHeight: '100vh' }}>
      <h1 style={{ textAlign: 'center', marginBottom: '2rem', borderBottom: '1px solid #333', paddingBottom: '1rem' }}>
        Traffic Violation Detector
      </h1>

      {!preview && (
        <div style={{ border: '2px dashed #444', padding: '4rem 2rem', textAlign: 'center', borderRadius: '8px' }}>
          <h3>Upload Image or Video</h3>
          <input type="file" accept="image/*,video/*" onChange={handleFileChange} style={{ marginTop: '1rem' }} />
        </div>
      )}

      {preview && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h2>Input Media</h2>
            <button onClick={handleReset} style={{ padding: '0.5rem 1rem', background: '#333', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>
              Upload Another
            </button>
          </div>

          <div style={{ background: '#000', padding: '1rem', borderRadius: '8px', textAlign: 'center' }}>
            {file?.type.startsWith('video/') ? (
              <video src={preview} controls style={{ maxWidth: '100%', maxHeight: '400px' }} />
            ) : (
              <img src={preview} alt="Preview" style={{ maxWidth: '100%', maxHeight: '400px' }} />
            )}
          </div>

          {loading && (
            <div style={{ background: '#222', padding: '2rem', textAlign: 'center', borderRadius: '8px', border: '1px solid #444' }}>
              <h2 style={{ color: '#4ade80', margin: 0 }}>⚙️ Processing Input...</h2>
              <p style={{ color: '#888', marginTop: '0.5rem' }}>Running dual-model parallel inference on backend.</p>
            </div>
          )}

          {error && (
            <div style={{ background: '#7f1d1d', padding: '1rem', borderRadius: '8px', border: '1px solid #ef4444' }}>
              <strong>Error:</strong> {error}
            </div>
          )}

          {results && (
            <div>
              <h2>Violations Detected: {results.length}</h2>
              {results.length === 0 ? (
                <div style={{ background: '#222', padding: '2rem', textAlign: 'center', borderRadius: '8px', color: '#888' }}>
                  No violations were found in this media.
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {results.map((v, idx) => (
                    <div key={idx} style={{ 
                      background: '#1f2937', 
                      padding: '1.5rem', 
                      borderRadius: '8px', 
                      borderLeft: `4px solid ${v.severity === 'HIGH' ? '#ef4444' : v.severity === 'MEDIUM' ? '#f59e0b' : '#10b981'}` 
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem' }}>
                        <strong style={{ fontSize: '1.2rem', textTransform: 'capitalize' }}>{v.type.replace(/_/g, ' ')}</strong>
                        <span style={{ 
                          background: v.severity === 'HIGH' ? '#7f1d1d' : v.severity === 'MEDIUM' ? '#78350f' : '#064e3b', 
                          padding: '0.25rem 0.5rem', 
                          borderRadius: '4px',
                          fontSize: '0.8rem',
                          fontWeight: 'bold'
                        }}>
                          {v.severity}
                        </span>
                      </div>
                      
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', fontSize: '0.9rem', color: '#ccc' }}>
                        <div><strong>Vehicle:</strong> {v.vehicle?.class}</div>
                        <div><strong>Confidence:</strong> {(v.confidence * 100).toFixed(1)}%</div>
                        {v.plate && (
                          <div style={{ gridColumn: 'span 2', marginTop: '0.5rem', background: '#000', padding: '0.5rem', borderRadius: '4px' }}>
                            <strong>License Plate:</strong> <span style={{ fontFamily: 'monospace', color: '#fff', fontSize: '1.1rem' }}>{v.plate.text}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
