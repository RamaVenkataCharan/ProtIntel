import React, { useMemo } from 'react';
import { STRUCTURE_COLORS, Q8_STRUCTURE_COLORS } from '../utils/colors';
import { BarChart2, PieChart, AlertTriangle } from 'lucide-react';

interface StatisticalBreakdownPanelsProps {
  q3Prediction: string[];
  q8Prediction: string[];
  confidence: number[];
}

const MINORITY_Q8_CLASSES = new Set(['I', 'B', 'S']);

export const StatisticalBreakdownPanels: React.FC<StatisticalBreakdownPanelsProps> = ({
  q3Prediction,
  q8Prediction,
  confidence,
}) => {
  const safeQ3 = q3Prediction || [];
  const safeQ8 = q8Prediction || safeQ3;
  const safeConf = confidence || [];
  const totalLength = safeQ3.length || 1;

  // 1. Q3 Statistical Breakdown Calculation
  const q3Stats = useMemo(() => {
    const counts: Record<'H' | 'E' | 'C', number> = { H: 0, E: 0, C: 0 };
    const confSums: Record<'H' | 'E' | 'C', number> = { H: 0, E: 0, C: 0 };

    safeQ3.forEach((cls, i) => {
      const validCls = (cls === 'H' || cls === 'E' || cls === 'C') ? cls : 'C';
      counts[validCls]++;
      confSums[validCls] += safeConf[i] ?? 1.0;
    });

    const keys: Array<'H' | 'E' | 'C'> = ['H', 'E', 'C'];
    
    const rawPcts = keys.map(k => (counts[k] / totalLength) * 100);
    const floorPcts = rawPcts.map(p => Math.floor(p));
    let currentSum = floorPcts.reduce((a, b) => a + b, 0);
    const remainders = rawPcts.map((p, idx) => ({ idx, rem: p - floorPcts[idx] }));
    remainders.sort((a, b) => b.rem - a.rem);

    let rIdx = 0;
    while (currentSum < 100 && rIdx < remainders.length) {
      floorPcts[remainders[rIdx].idx]++;
      currentSum++;
      rIdx++;
    }

    return keys.map((k, idx) => {
      const count = counts[k];
      const pct = totalLength > 0 ? (count / totalLength) * 100 : 0;
      const avgConf = count > 0 ? (confSums[k] / count) * 100 : 0;
      return {
        key: k,
        label: STRUCTURE_COLORS[k].label,
        hex: STRUCTURE_COLORS[k].hex,
        count,
        pct: pct.toFixed(1),
        displayPct: floorPcts[idx],
        avgConf: avgConf.toFixed(1),
      };
    });
  }, [safeQ3, safeConf, totalLength]);

  // 2. Q8 Statistical Breakdown Calculation
  const q8Stats = useMemo(() => {
    const q8Keys: Array<'H' | 'G' | 'I' | 'E' | 'B' | 'T' | 'S' | 'C'> = [
      'H', 'G', 'I', 'E', 'B', 'T', 'S', 'C'
    ];

    const counts: Record<string, number> = {};
    const confSums: Record<string, number> = {};
    q8Keys.forEach(k => { counts[k] = 0; confSums[k] = 0; });

    safeQ8.forEach((cls, i) => {
      const key = cls in counts ? cls : 'C';
      counts[key]++;
      confSums[key] += safeConf[i] ?? 1.0;
    });

    const rawPcts = q8Keys.map(k => (counts[k] / totalLength) * 100);
    const floorPcts = rawPcts.map(p => Math.floor(p));
    let currentSum = floorPcts.reduce((a, b) => a + b, 0);
    const remainders = rawPcts.map((p, idx) => ({ idx, rem: p - floorPcts[idx] }));
    remainders.sort((a, b) => b.rem - a.rem);

    let rIdx = 0;
    while (currentSum < 100 && rIdx < remainders.length) {
      floorPcts[remainders[rIdx].idx]++;
      currentSum++;
      rIdx++;
    }

    return q8Keys.map((k, idx) => {
      const colorDef = Q8_STRUCTURE_COLORS[k] || Q8_STRUCTURE_COLORS['C'];
      const count = counts[k];
      const pct = totalLength > 0 ? (count / totalLength) * 100 : 0;
      const avgConf = count > 0 ? (confSums[k] / count) * 100 : 0;
      const isMinority = MINORITY_Q8_CLASSES.has(k);

      return {
        key: k,
        label: colorDef.label,
        hex: colorDef.hex,
        count,
        pct: pct.toFixed(1),
        displayPct: floorPcts[idx],
        avgConf: avgConf.toFixed(1),
        isMinority,
      };
    });
  }, [q8Prediction, confidence, totalLength]);

  return (
    <div className="flex flex-col gap-5 animate-spring-up mt-2">
      {/* SECTION HEADER */}
      <div className="flex items-center justify-between pb-2.5 border-b border-[var(--border-subtle)]">
        <div className="flex items-center gap-2.5">
          <div className="p-1.5 rounded-xl bg-teal-500/10 border border-teal-500/30">
            <PieChart className="h-4 w-4 text-teal-400" strokeWidth={1.8} />
          </div>
          <div>
            <h4 className="text-sm font-bold text-[var(--text-primary)]" style={{ fontFamily: 'var(--font-heading)' }}>
              Statistical Structure Breakdown
            </h4>
            <p className="text-[10px] font-mono text-[var(--text-muted)]">
              Aggregate distribution across {totalLength} amino acid residues
            </p>
          </div>
        </div>
        <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-md bg-violet-500/15 text-violet-300 border border-violet-500/30">
          Q3 & Q8 AGGREGATE
        </span>
      </div>

      {/* Q3 BREAKDOWN PANEL */}
      <div className="surface-tier-2 p-5 rounded-2xl flex flex-col gap-4 border border-[var(--aurora-violet)]/25">
        <div className="flex items-center justify-between">
          <h5 className="text-xs font-mono font-bold text-[var(--text-primary)] flex items-center gap-2 uppercase tracking-wider">
            <BarChart2 className="h-3.5 w-3.5 text-violet-400" strokeWidth={2} />
            <span>Q3 Class Breakdown (Helix / Sheet / Coil)</span>
          </h5>
          <span className="text-[10px] font-mono text-[var(--text-muted)] font-bold">3 CLASSES</span>
        </div>

        {/* Progress Bar */}
        <div className="h-3.5 w-full bg-black/40 rounded-xl overflow-hidden flex p-0.5 gap-0.5 border border-white/10">
          {q3Stats.map((item) => {
            const widthPct = parseFloat(item.pct);
            if (widthPct === 0) return null;
            return (
              <div
                key={item.key}
                className="h-full rounded-sm transition-all duration-500 relative group cursor-pointer"
                style={{
                  width: `${widthPct}%`,
                  backgroundColor: item.hex,
                  boxShadow: `0 0 10px ${item.hex}40`,
                }}
                title={`${item.label}: ${item.count} residues (${item.pct}%), Avg Conf: ${item.avgConf}%`}
              />
            );
          })}
        </div>

        {/* Q3 Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {q3Stats.map((item) => (
            <div
              key={item.key}
              className="surface-tier-3 p-3.5 rounded-xl flex items-center justify-between border interactive-card"
              style={{ borderColor: `${item.hex}40` }}
            >
              <div className="flex items-center gap-2.5">
                <span className="w-3 h-3 rounded-md shrink-0 shadow-sm" style={{ backgroundColor: item.hex }} />
                <div>
                  <span className="text-xs font-bold font-mono text-[var(--text-primary)] block">
                    {item.label}
                  </span>
                  <span className="text-[10px] font-mono text-[var(--text-muted)] block">
                    {item.count} res ({item.pct}%)
                  </span>
                </div>
              </div>

              <div className="text-right font-mono">
                <span className="text-[9px] text-[var(--text-muted)] font-bold uppercase block">Avg Conf</span>
                <span className="text-xs font-bold text-teal-400">{item.avgConf}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Q8 BREAKDOWN PANEL */}
      <div className="surface-tier-2 p-5 rounded-2xl flex flex-col gap-4 border border-teal-500/25">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <h5 className="text-xs font-mono font-bold text-[var(--text-primary)] flex items-center gap-2 uppercase tracking-wider">
              <BarChart2 className="h-3.5 w-3.5 text-teal-400" strokeWidth={2} />
              <span>Q8 DSSP Class Breakdown</span>
            </h5>
            <span className="text-[10px] font-mono text-[var(--text-muted)] font-bold">8 DETAILED CLASSES</span>
          </div>

          <div className="flex items-center gap-1.5 text-[10px] font-mono text-amber-400/90">
            <AlertTriangle className="h-3 w-3" strokeWidth={2} />
            <span>Muted = Minority DSSP Classes (I, B, S)</span>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="h-3.5 w-full bg-black/40 rounded-xl overflow-hidden flex p-0.5 gap-0.5 border border-white/10">
          {q8Stats.map((item) => {
            const widthPct = parseFloat(item.pct);
            if (widthPct === 0) return null;
            return (
              <div
                key={item.key}
                className="h-full rounded-sm transition-all duration-500 relative group cursor-pointer"
                style={{
                  width: `${widthPct}%`,
                  backgroundColor: item.hex,
                  opacity: item.isMinority ? 0.5 : 1,
                  boxShadow: `0 0 8px ${item.hex}30`,
                }}
                title={`${item.label}: ${item.count} residues (${item.pct}%), Avg Conf: ${item.avgConf}%${item.isMinority ? ' [Minority Class]' : ''}`}
              />
            );
          })}
        </div>

        {/* Q8 Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
          {q8Stats.map((item) => {
            const isZero = item.count === 0;
            return (
              <div
                key={item.key}
                className={`p-3 rounded-xl flex flex-col justify-between border transition-all ${
                  item.isMinority
                    ? 'bg-black/30 border-amber-400/20 opacity-80'
                    : 'surface-tier-3'
                }`}
                style={{
                  borderColor: isZero ? 'var(--border-subtle)' : `${item.hex}40`,
                  opacity: item.isMinority ? (isZero ? 0.45 : 0.7) : (isZero ? 0.55 : 1),
                }}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-sm shrink-0 shadow-sm" style={{ backgroundColor: item.hex }} />
                    <span className="text-xs font-bold font-mono text-[var(--text-primary)]">
                      {item.key}
                    </span>
                  </div>
                  {item.isMinority && (
                    <span className="text-[8px] font-mono font-bold px-1 rounded bg-amber-400/10 text-amber-400 border border-amber-400/20" title="Minority Class">
                      MIN
                    </span>
                  )}
                </div>

                <div className="text-[10px] font-mono leading-tight">
                  <span className="text-[var(--text-muted)] block truncate" title={item.label}>
                    {item.label}
                  </span>
                  <div className="flex items-baseline justify-between mt-1 pt-1 border-t border-white/5">
                    <span className={`font-bold ${isZero ? 'text-slate-500' : 'text-slate-200'}`}>
                      {item.count} res ({item.pct}%)
                    </span>
                    {!isZero && (
                      <span className="text-teal-400 text-[9px] font-bold">
                        {item.avgConf}%
                      </span>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
