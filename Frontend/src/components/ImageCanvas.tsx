import { useEffect, useRef, useState } from "react";
import type { FrameResultModel } from "../types";

interface ImageCanvasProps {
  result: FrameResultModel;
  mediaUrl: string;
  isVideo: boolean;
  selectedViolationId: string | null;
}

export default function ImageCanvas({ result, mediaUrl, isVideo, selectedViolationId }: ImageCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mediaRef = useRef<HTMLImageElement | HTMLVideoElement>(null);
  const [mediaLoaded, setMediaLoaded] = useState(false);

  useEffect(() => {
    if (!mediaLoaded || !canvasRef.current || !mediaRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // The rendered size of the image/video element
    const rect = mediaRef.current.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    // The original dimensions from the backend
    const origW = result.dimensions.width;
    const origH = result.dimensions.height;

    // Scale factors
    const scaleX = rect.width / origW;
    const scaleY = rect.height / origH;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const getSeverityColor = (severity: string) => {
      switch (severity) {
        case "HIGH": return "#ef4444";
        case "MEDIUM": return "#f59e0b";
        case "LOW": return "#10b981";
        default: return "#3b82f6";
      }
    };

    // Draw all violations
    result.violations.forEach(v => {
      const isSelected = selectedViolationId === v.id;
      const isFaded = selectedViolationId !== null && !isSelected;
      
      const color = getSeverityColor(v.severity);
      
      // Draw main vehicle bbox
      drawBox(ctx, v.vehicle.bbox, scaleX, scaleY, color, isSelected, isFaded, v.type);

      // Draw evidence bboxes (e.g. helmet, traffic light)
      if (isSelected || selectedViolationId === null) {
        v.evidence.forEach(e => {
          drawBox(ctx, e.bbox, scaleX, scaleY, "#3b82f6", false, false, e.class);
        });
      }

      // Draw plate bbox if present
      if (v.plate && (isSelected || selectedViolationId === null)) {
        drawBox(ctx, v.plate.bbox, scaleX, scaleY, "#a855f7", false, false, v.plate.text);
      }
    });

  }, [result, mediaLoaded, selectedViolationId]);

  const drawBox = (
    ctx: CanvasRenderingContext2D, 
    bbox: number[], 
    scaleX: number, 
    scaleY: number, 
    color: string, 
    isSelected: boolean,
    isFaded: boolean,
    label: string
  ) => {
    const [x1, y1, x2, y2] = bbox;
    const sx = x1 * scaleX;
    const sy = y1 * scaleY;
    const sw = (x2 - x1) * scaleX;
    const sh = (y2 - y1) * scaleY;

    ctx.globalAlpha = isFaded ? 0.2 : 1.0;
    
    // Box
    ctx.strokeStyle = color;
    ctx.lineWidth = isSelected ? 4 : 2;
    ctx.strokeRect(sx, sy, sw, sh);

    // Label background
    ctx.fillStyle = color;
    const labelHeight = 20;
    ctx.fillRect(sx, sy - labelHeight, ctx.measureText(label).width + 8, labelHeight);

    // Label text
    ctx.fillStyle = "#ffffff";
    ctx.font = "12px Inter";
    ctx.fillText(label, sx + 4, sy - 5);
    
    ctx.globalAlpha = 1.0;
  };

  return (
    <div className="canvas-container">
      <div className="canvas-wrapper">
        {isVideo ? (
          <video
            ref={mediaRef as any}
            src={mediaUrl}
            className="image-layer"
            controls
            onLoadedData={() => setMediaLoaded(true)}
          />
        ) : (
          <img
            ref={mediaRef as any}
            src={mediaUrl}
            className="image-layer"
            alt="Analyzed frame"
            onLoad={() => setMediaLoaded(true)}
          />
        )}
        <canvas
          ref={canvasRef}
          className="canvas-layer"
        />
      </div>
    </div>
  );
}
