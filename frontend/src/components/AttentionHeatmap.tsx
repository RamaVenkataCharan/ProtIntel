import React, { useRef, useEffect, useState } from 'react';

interface AttentionHeatmapProps {
  attentionMap: number[][];
  sequence: string;
}

export const AttentionHeatmap: React.FC<AttentionHeatmapProps> = ({ attentionMap, sequence }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number; val: number } | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

  const L = sequence.length;
  const cellSize = Math.max(4, Math.min(24, Math.floor(400 / L))); // Dynamically size cells
  const width = L * cellSize;
  const height = L * cellSize;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw heatmap
    for (let r = 0; r < L; r++) {
      for (let c = 0; c < L; c++) {
        const val = attentionMap[r]?.[c] || 0;
        
        // Premium color mapping: HSL from dark slate (0 attention) to bright cyan/indigo (high attention)
        // Hue 260 (Indigo) for low attention, shifting to Hue 180 (Cyan) for high attention
        const intensity = Math.min(1, val * 10); // Amplify for better visibility
        ctx.fillStyle = `hsla(${260 - intensity * 80}, 90%, ${10 + intensity * 60}%, ${0.2 + intensity * 0.8})`;
        ctx.fillRect(c * cellSize, r * cellSize, cellSize - 0.5, cellSize - 0.5);
      }
    }
  }, [attentionMap, L, cellSize]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const col = Math.floor(x / cellSize);
    const row = Math.floor(y / cellSize);

    if (row >= 0 && row < L && col >= 0 && col < L) {
      const val = attentionMap[row]?.[col] || 0;
      setHoveredCell({ row, col, val });
      setTooltipPos({ x: e.clientX - rect.left + 15, y: e.clientY - rect.top + 15 });
    } else {
      setHoveredCell(null);
    }
  };

  const handleMouseLeave = () => {
    setHoveredCell(null);
  };

  return (
    <div className="flex flex-col items-center bg-slate-900/30 border border-slate-800/80 p-6 rounded-3xl relative">
      <div className="relative overflow-auto max-w-full">
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          className="border border-slate-800 rounded-lg cursor-crosshair"
        />

        {hoveredCell && (
          <div
            style={{ left: tooltipPos.x, top: tooltipPos.y }}
            className="absolute z-10 pointer-events-none bg-slate-950/95 border border-purple-500/30 px-3 py-2 rounded-xl text-xs shadow-2xl flex flex-col gap-1 backdrop-blur-md"
          >
            <span className="text-purple-400 font-bold">Residue Connection</span>
            <div className="text-slate-300 font-mono">
              Row {hoveredCell.row + 1} ({sequence[hoveredCell.row]}) &rarr; Col {hoveredCell.col + 1} ({sequence[hoveredCell.col]})
            </div>
            <div className="text-slate-400">
              Weight: <span className="font-semibold text-white">{hoveredCell.val.toFixed(5)}</span>
            </div>
          </div>
        )}
      </div>

      <div className="flex justify-between w-full mt-4 text-[10px] text-slate-400 px-2 font-medium">
        <span>Residue 1 (Col) &rarr;</span>
        <span>&larr; Residue {L} (Col)</span>
      </div>
    </div>
  );
};
