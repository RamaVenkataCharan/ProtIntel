import React, { useState, lazy, Suspense, useRef, useCallback, useMemo } from 'react';
import { usePredictionStore } from '../store/usePredictionStore';
import { useModelStore } from '../store/useModelStore';
import { StatisticalBreakdownPanels } from '../components/StatisticalBreakdownPanels';

const ProteinStructure3D = lazy(() => import('../components/ProteinStructure3D').then(m => ({ default: m.ProteinStructure3D })));
const AssemblyLoader = lazy(() => import('../components/AssemblyLoader'));

import { STRUCTURE_COLORS, Q8_STRUCTURE_COLORS, getXAIColorHex } from '../utils/colors';
import { Play, AlertCircle, Sparkles, Cpu, Download, Search, FileSpreadsheet, Activity } from 'lucide-react';

const MINORITY_Q8 = new Set(['I', 'B', 'S']);

// Sample sequences for quick scientific exploration
const SAMPLE_SEQUENCES = [
  { label: 'Ubiquitin (76 aa)', seq: 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG' },
  { label: 'Crambin (46 aa)', seq: 'TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN' },
  { label: 'Myoglobin Frag (50 aa)', seq: 'VLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLK' }
];

// =============================================================================
// PHYSICOCHEMICAL PROFILE CALCULATOR
// =============================================================================
const computePhysicochemicalProfile = (seq: string) => {
  if (!seq) return { mw: 0, pI: 7.0, netCharge: 0, hydrophobicPct: 0 };
  const len = seq.length;
  
  const masses: Record<string, number> = {
    A: 71.08, R: 156.19, N: 114.10, D: 115.09, C: 103.14, E: 129.12, Q: 128.13,
    G: 57.05, H: 137.14, I: 131.17, L: 131.17, K: 128.17, M: 131.19, F: 147.18,
    P: 97.12, S: 87.08, T: 101.11, W: 186.21, Y: 163.18, V: 99.13
  };
  let mw = 18.02; // H2O terminal
  let pos = 0;
  let neg = 0;
  let hydrophobic = 0;
  const hydrophobics = new Set(['A', 'V', 'I', 'L', 'M', 'F', 'W', 'P']);

  for (const aa of seq) {
    mw += masses[aa] || 110.0;
    if (aa === 'K' || aa === 'R' || aa === 'H') pos++;
    if (aa === 'D' || aa === 'E') neg++;
    if (hydrophobics.has(aa)) hydrophobic++;
  }

  const netCharge = pos - neg;
  const pI = Math.min(11.5, Math.max(3.5, 7.0 + (netCharge / len) * 4.5));
  const hydrophobicPct = (hydrophobic / len) * 100;

  return {
    mw: mw / 1000, // kDa
    pI,
    netCharge,
    hydrophobicPct
  };
};

// =============================================================================
// STACKED SEQUENCE STRIP (Q3 above, Q8 below, Motif Highlight support)
// =============================================================================
interface StackedStripProps {
  sequence: string;
  q3Prediction: string[];
  q8Prediction: string[];
  confidence: number[];
  residueImportance?: number[] | null;
  activeTab: string;
  activeClassification?: 'q3' | 'q8';
  hoveredIndex: number | null;
  searchPattern: string;
  onHover: (idx: number | null) => void;
  onSelectClassification?: (mode: 'q3' | 'q8') => void;
}

const StackedSequenceStrip: React.FC<StackedStripProps> = ({
  sequence, q3Prediction, q8Prediction, confidence, residueImportance,
  activeTab, activeClassification = 'q3', hoveredIndex, searchPattern, onHover, onSelectClassification,
}) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const chars = sequence.split('');
  const len = chars.length;

  const matchedIndices = useMemo(() => {
    if (!searchPattern || searchPattern.length < 1) return new Set<number>();
    const query = searchPattern.toUpperCase();
    const matches = new Set<number>();
    for (let i = 0; i <= sequence.length - query.length; i++) {
      if (sequence.substring(i, i + query.length) === query) {
        for (let k = 0; k < query.length; k++) matches.add(i + k);
      }
    }
    return matches;
  }, [sequence, searchPattern]);

  const { minImp, impRange } = useMemo(() => {
    if (!residueImportance || residueImportance.length === 0) {
      return { minImp: 0, impRange: 1 };
    }
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < residueImportance.length; i++) {
      const val = residueImportance[i];
      if (val < min) min = val;
      if (val > max) max = val;
    }
    const range = (max - min) > 1e-6 ? (max - min) : 1.0;
    return { minImp: min, impRange: range };
  }, [residueImportance]);

  return (
    <div className="flex flex-col gap-0 animate-spring-up">
      <div className="flex flex-wrap items-center justify-between mb-2.5 gap-2">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-[var(--aurora-amber)] animate-pulse" />
          <h5 className="text-xs font-bold uppercase tracking-widest text-[var(--text-primary)] font-mono">
            Sequence Structure Map
          </h5>
          {matchedIndices.size > 0 && (
            <span className="text-[10px] font-mono px-2 py-0.5 rounded-md bg-teal-500/20 text-teal-300 font-bold border border-teal-500/40">
              {matchedIndices.size} Residues Matched ({searchPattern.toUpperCase()})
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {Object.entries(STRUCTURE_COLORS).map(([k, cfg]) => (
            <span key={k} className="flex items-center gap-1.5 text-[10px] font-mono font-semibold text-[var(--text-muted)]">
              <span className="w-2.5 h-2.5 rounded-sm inline-block shadow-sm" style={{ backgroundColor: cfg.hex }} />
              {cfg.label}
            </span>
          ))}
        </div>
      </div>

      <div
        ref={scrollRef}
        className="residue-strip rounded-2xl relative surface-tier-3"
        style={{ padding: '10px 10px 14px' }}
        onMouseLeave={() => onHover(null)}
      >
        {/* Q3 Track Header */}
        <div
          onClick={() => onSelectClassification?.('q3')}
          className={`flex items-center justify-between mb-1.5 px-1 cursor-pointer transition-all select-none ${
            activeClassification === 'q3' ? 'text-violet-300' : 'text-slate-400 hover:text-slate-200'
          }`}
          title="Click to mirror Q3 3-Class in 3D Viewer"
        >
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono font-bold uppercase tracking-wider">
              Q3 Primary Structure (3-Class: Helix / Sheet / Coil)
            </span>
            {activeClassification === 'q3' && (
              <span className="px-1.5 py-0.5 rounded text-[8px] font-mono font-extrabold bg-violet-500/20 text-violet-300 border border-violet-500/40 animate-pulse">
                3D VIEW ACTIVE
              </span>
            )}
          </div>
          <span className="text-[9px] font-mono text-slate-500">H = Helix, E = Beta-Sheet, C = Coil</span>
        </div>

        <div className={`flex gap-[2px] min-w-max p-1 rounded-xl transition-all ${
          activeClassification === 'q3' ? 'ring-1 ring-violet-500/40 bg-violet-950/20 shadow-[0_0_12px_rgba(139,92,246,0.15)]' : ''
        }`}>
          {chars.map((aa, idx) => {
            const q3 = q3Prediction[idx] || 'C';
            const conf = confidence[idx] ?? 1;
            const isHov = hoveredIndex === idx;
            const isMatch = matchedIndices.has(idx);

            const rawImp = residueImportance?.[idx] ?? minImp;
            const normImp = Math.max(0, Math.min(1, (rawImp - minImp) / impRange));
            const bgColor = activeTab === 'xai'
              ? getXAIColorHex(normImp)
              : (STRUCTURE_COLORS[q3 as keyof typeof STRUCTURE_COLORS]?.hex || '#94A3B8');

            return (
              <div
                key={idx}
                className={`residue-block flex flex-col items-center justify-between select-none${isHov ? ' hovered' : ''}`}
                style={{
                  height: 54, backgroundColor: bgColor,
                  opacity: conf < 0.5 ? 0.45 : conf < 0.7 ? 0.75 : 1,
                  outline: isMatch ? '2px solid #FFB347' : isHov ? '2px solid rgba(255,255,255,0.9)' : '2px solid transparent',
                  outlineOffset: '1px',
                  boxShadow: isMatch ? '0 0 10px #FFB347' : isHov ? `0 0 12px ${bgColor}` : 'none',
                }}
                onMouseEnter={() => onHover(idx)}
                onClick={() => onSelectClassification?.('q3')}
              >
                <span className="text-[8px] text-white/60 font-mono pt-[3px] leading-none">{idx + 1}</span>
                <span className="text-[11px] text-white font-bold font-mono leading-none">{aa}</span>
                <span className="text-[9px] font-bold leading-none pb-[3px] font-mono text-white/90">{q3}</span>
              </div>
            );
          })}
        </div>

        {/* Position Scrubber / Alignment Bar */}
        <div className="relative h-[6px] min-w-max my-[5px]" style={{ width: Math.max(0, len * 26 + (len - 1) * 2) }}>
          <div className="absolute inset-0 rounded-full" style={{ background: 'linear-gradient(90deg, rgba(123,47,247,0.25), rgba(0,217,192,0.25), rgba(255,179,71,0.25))' }} />
          {hoveredIndex !== null && (
            <div
              className="absolute top-0 bottom-0 rounded-full"
              style={{
                left: hoveredIndex * 26, width: 24,
                background: '#FFB347',
                boxShadow: '0 0 10px #FFB347',
                transition: 'left 0.05s linear'
              }}
            />
          )}
        </div>

        {/* Q8 Track Header */}
        <div
          onClick={() => onSelectClassification?.('q8')}
          className={`flex items-center justify-between mt-2.5 mb-1.5 px-1 cursor-pointer transition-all select-none ${
            activeClassification === 'q8' ? 'text-teal-300' : 'text-slate-400 hover:text-slate-200'
          }`}
          title="Click to mirror Q8 8-State DSSP in 3D Viewer"
        >
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono font-bold uppercase tracking-wider">
              Q8 Detailed DSSP States (8-State Granularity)
            </span>
            {activeClassification === 'q8' && (
              <span className="px-1.5 py-0.5 rounded text-[8px] font-mono font-extrabold bg-teal-500/20 text-teal-300 border border-teal-500/40 animate-pulse">
                3D VIEW ACTIVE
              </span>
            )}
          </div>
          <span className="text-[9px] font-mono text-slate-500">H, G, I, E, B, T, S, C</span>
        </div>

        <div className={`flex gap-[2px] min-w-max p-1 rounded-xl transition-all ${
          activeClassification === 'q8' ? 'ring-1 ring-teal-500/40 bg-teal-950/20 shadow-[0_0_12px_rgba(0,229,204,0.15)]' : ''
        }`}>
          {chars.map((_, idx) => {
            const q8 = q8Prediction[idx] || 'C';
            const isMinority = MINORITY_Q8.has(q8);
            const conf = confidence[idx] ?? 1;
            const isHov = hoveredIndex === idx;
            const colorDef = Q8_STRUCTURE_COLORS[q8 as keyof typeof Q8_STRUCTURE_COLORS] || Q8_STRUCTURE_COLORS['C'];
            return (
              <div
                key={idx}
                className={`residue-block flex flex-col items-center justify-between select-none${isHov ? ' hovered' : ''}`}
                style={{
                  height: 38, backgroundColor: colorDef.hex,
                  opacity: isMinority ? 0.35 : conf < 0.6 ? 0.55 : 0.85,
                  outline: isHov ? '2px solid rgba(255,255,255,0.8)' : '2px solid transparent',
                  outlineOffset: '1px',
                  filter: isMinority ? 'saturate(0.4)' : undefined,
                }}
                onMouseEnter={() => onHover(idx)}
                onClick={() => onSelectClassification?.('q8')}
              >
                <span className="text-[9px] font-bold text-white/90 leading-none pt-[3px] font-mono">{q8}</span>
                {isMinority && <span className="text-[7px] font-bold leading-none pb-[2px] font-mono text-amber-300">!</span>}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

// =============================================================================
// COMPUTING ACTIVE LOADING STATE
// =============================================================================
const PredictionLoadingState: React.FC<{ jobStatus: string | null }> = ({ jobStatus }) => (
  <div className="flex-1 flex flex-col items-center justify-center min-h-[480px]">
    <Suspense fallback={
      <div className="flex-1 flex flex-col items-center justify-center surface-tier-2 p-12 text-center min-h-[480px] relative overflow-hidden">
        <div className="animate-laser-scan" />
        <div className="flex items-end gap-1 mb-8 p-4 rounded-2xl bg-black/40 border border-white/[0.08]">
          {Array.from({ length: 28 }, (_, i) => (
            <div
              key={i}
              className="rounded-sm"
              style={{
                width: 6,
                height: 16 + Math.sin(i * 0.6) * 14,
                backgroundColor: i < 9 ? '#7B2FF7' : i < 19 ? '#00D9C0' : '#FFB347',
                opacity: 0.3 + (i / 27) * 0.7,
                animation: `residue-pulse 1.1s ease-in-out ${i * 0.04}s infinite`,
              }}
            />
          ))}
        </div>
        <h5 className="text-sm font-bold text-[var(--text-primary)] font-mono" style={{ fontFamily: 'var(--font-heading)' }}>
          ESM-2 NEURAL INFERENCE ACTIVE
        </h5>
        <p className="text-xs font-mono text-[var(--text-muted)] mt-1">Executing BiLSTM & Self-Attention Heads</p>
      </div>
    }>
      <AssemblyLoader
        jobStatus={jobStatus as any}
        isComplete={false}
      />
    </Suspense>
  </div>
);

// =============================================================================
// MAIN PREDICT PAGE
// =============================================================================
export const Predict: React.FC = () => {
  const { runPredict, activePrediction, isPredicting, predictionError, jobStatus } = usePredictionStore();
  const { modelLoaded } = useModelStore();

  const [sequence, setSequence] = useState(activePrediction?.sequence || '');
  const [returnAttention, setReturnAttention] = useState(false);
  const [returnXai, setReturnXai] = useState(false);
  const [xaiMethod, setXaiMethod] = useState<'ig' | 'shap' | 'rollout'>('ig');
  const [validationError, setValidationError] = useState<string | null>(null);
  const [stripViewMode, setStripViewMode] = useState<'structure' | 'xai'>('structure');
  const [classificationMode, setClassificationMode] = useState<'q3' | 'q8'>('q3');
  const [searchPattern, setSearchPattern] = useState('');

  const [hoveredResidue, setHoveredResidue] = useState<{
    index: number; aa: string; q3: string; q8: string;
    q3_prob: number[]; q8_prob: number[]; conf: number; importance?: number;
  } | null>(null);

  const setHoveredResidueByIndex = useCallback((idx: number | null) => {
    if (idx === null || !activePrediction) { setHoveredResidue(null); return; }
    setHoveredResidue({
      index: idx,
      aa: activePrediction.sequence[idx],
      q3: activePrediction.q3_prediction[idx],
      q8: activePrediction.q8_prediction[idx],
      conf: activePrediction.confidence[idx],
      q3_prob: activePrediction.q3_probabilities[idx],
      q8_prob: activePrediction.q8_probabilities[idx],
      importance: activePrediction.residue_importance?.[idx] || 0,
    });
  }, [activePrediction]);

  const cleanInputSequence = (input: string): string => {
    let clean = input.trim();
    const lines = clean.split('\n');
    if (lines[0].startsWith('>')) clean = lines.slice(1).join('');
    else clean = lines.join('');
    return clean.toUpperCase().replace(/\s/g, '').replace(/[-*]/g, '');
  };

  const rawLength = useMemo(() => cleanInputSequence(sequence).length, [sequence]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setValidationError(null);
    const cleaned = cleanInputSequence(sequence);
    if (cleaned.length < 5) { setValidationError('Sequence must be at least 5 amino acids long.'); return; }
    if (cleaned.length > 2048) { setValidationError('Sequence exceeds maximum length of 2048 residues.'); return; }
    try { await runPredict({ sequence: cleaned, return_attention: returnAttention, return_xai: returnXai, xai_method: xaiMethod }); }
    catch { /* handled by store */ }
  };

  // Physicochemical Profile of Active Result
  const physProfile = useMemo(() => computePhysicochemicalProfile(activePrediction?.sequence || ''), [activePrediction]);

  // Multi-Format Prediction Exporter (JSON / CSV)
  const exportPredictionCSV = () => {
    if (!activePrediction) return;
    const len = activePrediction.sequence.length;

    const q3Counts: Record<string, number> = { H: 0, E: 0, C: 0 };
    activePrediction.q3_prediction.forEach(c => { if (c in q3Counts) q3Counts[c]++; });

    const q8Counts: Record<string, number> = { H: 0, G: 0, I: 0, E: 0, B: 0, T: 0, S: 0, C: 0 };
    activePrediction.q8_prediction.forEach(c => { if (c in q8Counts) q8Counts[c]++; });

    let csv = `# ProtIntel Prediction Report - ${activePrediction.protein_id}\n`;
    csv += `# Length: ${len} residues\n`;
    csv += `# Q3 Summary: Helix(H)=${q3Counts.H} (${(q3Counts.H/len*100).toFixed(1)}%), Sheet(E)=${q3Counts.E} (${(q3Counts.E/len*100).toFixed(1)}%), Coil(C)=${q3Counts.C} (${(q3Counts.C/len*100).toFixed(1)}%)\n`;
    csv += `# Q8 Summary: H=${q8Counts.H}, G=${q8Counts.G}, I=${q8Counts.I}, E=${q8Counts.E}, B=${q8Counts.B}, T=${q8Counts.T}, S=${q8Counts.S}, C=${q8Counts.C}\n\n`;

    csv += 'Index,Residue,Q3_Pred,Q8_Pred,Confidence,XAI_Importance\n';
    activePrediction.sequence.split('').forEach((aa, i) => {
      const q3 = activePrediction.q3_prediction[i];
      const q8 = activePrediction.q8_prediction[i];
      const conf = activePrediction.confidence[i].toFixed(4);
      const imp = activePrediction.residue_importance?.[i]?.toFixed(4) || '0.0000';
      csv += `${i + 1},${aa},${q3},${q8},${conf},${imp}\n`;
    });
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ProtIntel_Prediction_${activePrediction.protein_id}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportPredictionJSON = () => {
    if (!activePrediction) return;
    const len = activePrediction.sequence.length;
    const q3Counts: Record<string, number> = { H: 0, E: 0, C: 0 };
    activePrediction.q3_prediction.forEach(c => { if (c in q3Counts) q3Counts[c]++; });

    const q8Counts: Record<string, number> = { H: 0, G: 0, I: 0, E: 0, B: 0, T: 0, S: 0, C: 0 };
    activePrediction.q8_prediction.forEach(c => { if (c in q8Counts) q8Counts[c]++; });

    const payload = {
      ...activePrediction,
      statistical_breakdown: {
        total_length: len,
        q3_breakdown: {
          helix_H: { count: q3Counts.H, percentage: parseFloat((q3Counts.H / len * 100).toFixed(1)) },
          sheet_E: { count: q3Counts.E, percentage: parseFloat((q3Counts.E / len * 100).toFixed(1)) },
          coil_C: { count: q3Counts.C, percentage: parseFloat((q3Counts.C / len * 100).toFixed(1)) },
        },
        q8_breakdown: Object.fromEntries(
          Object.entries(q8Counts).map(([k, count]) => [
            k,
            { count, percentage: parseFloat((count / len * 100).toFixed(1)) }
          ])
        )
      }
    };

    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ProtIntel_Prediction_${activePrediction.protein_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col gap-6 animate-fade-in">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* LEFT 1 COL: Instrument Controls & Sequence Input */}
        <div className="lg:col-span-1">
          <section className="surface-tier-2 p-6 flex flex-col gap-5 relative overflow-hidden">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="p-2 rounded-xl bg-purple-500/15 border border-purple-500/30">
                  <Cpu className="h-4 w-4 text-violet-400" strokeWidth={1.8} />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-[var(--text-primary)]" style={{ fontFamily: 'var(--font-heading)' }}>
                    Inference Console
                  </h3>
                  <span className="text-[10px] font-mono text-[var(--text-muted)]">ESM2_650M // DUAL_HEAD</span>
                </div>
              </div>
              <span className="text-[10px] font-mono font-bold text-teal-400 bg-teal-400/10 px-2 py-0.5 rounded border border-teal-400/20">
                READY
              </span>
            </div>

            {/* Quick Sample Presets */}
            <div>
              <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-[var(--text-muted)] block mb-1.5">
                Load Sample Sequence:
              </span>
              <div className="flex flex-wrap gap-1.5">
                {SAMPLE_SEQUENCES.map((samp) => (
                  <button
                    key={samp.label}
                    type="button"
                    onClick={() => { setSequence(samp.seq); setValidationError(null); }}
                    className="text-[10px] font-mono px-2.5 py-1 rounded-lg bg-[var(--glass-tier3)] hover:bg-[var(--glass-interactive-hover)] border border-[var(--border-subtle)] hover:border-[var(--aurora-teal)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-all cursor-pointer"
                  >
                    {samp.label}
                  </button>
                ))}
              </div>
            </div>

            <form onSubmit={handleSubmit} className="flex flex-col gap-4">
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <label className="text-[10px] font-mono font-bold uppercase tracking-widest text-[var(--text-muted)]">
                    FASTA Sequence
                  </label>
                  <span className="text-[10px] font-mono text-[var(--text-muted)]">
                    Length: <strong className="text-slate-200">{rawLength}</strong> aa
                  </span>
                </div>
                <textarea
                  value={sequence}
                  onChange={(e) => { setSequence(e.target.value); setValidationError(null); }}
                  placeholder={">MyProtein\nMKFLILLFNILCLFPVLA..."}
                  className="instrument-input w-full h-36 p-3.5 text-xs font-mono resize-none"
                  required
                />
              </div>

              {/* Toggles Panel */}
              <div className="surface-tier-3 p-4 flex flex-col gap-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-[var(--text-secondary)] flex items-center gap-2">
                    <Activity className="h-3.5 w-3.5 text-teal-400" />
                    Compute Self-Attention
                  </span>
                  <button
                    type="button"
                    role="switch"
                    aria-checked={returnAttention}
                    onClick={() => setReturnAttention(!returnAttention)}
                    className="relative flex-shrink-0 transition-colors"
                    style={{
                      width: 36, height: 20, borderRadius: 10, cursor: 'pointer',
                      background: returnAttention ? 'linear-gradient(135deg, var(--aurora-violet), var(--aurora-teal))' : 'rgba(255,255,255,0.08)',
                      border: `1px solid ${returnAttention ? 'rgba(0,217,192,0.5)' : 'var(--border-muted)'}`,
                    }}
                  >
                    <span className="absolute top-[2px] transition-all duration-200" style={{ width: 14, height: 14, borderRadius: '50%', background: '#fff', left: returnAttention ? 18 : 2 }} />
                  </button>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-[var(--text-secondary)] flex items-center gap-2">
                    <Sparkles className="h-3.5 w-3.5 text-purple-400" />
                    Compute XAI Attributions
                  </span>
                  <button
                    type="button"
                    role="switch"
                    aria-checked={returnXai}
                    onClick={() => setReturnXai(!returnXai)}
                    className="relative flex-shrink-0 transition-colors"
                    style={{
                      width: 36, height: 20, borderRadius: 10, cursor: 'pointer',
                      background: returnXai ? 'linear-gradient(135deg, var(--aurora-violet), var(--aurora-teal))' : 'rgba(255,255,255,0.08)',
                      border: `1px solid ${returnXai ? 'rgba(0,217,192,0.5)' : 'var(--border-muted)'}`,
                    }}
                  >
                    <span className="absolute top-[2px] transition-all duration-200" style={{ width: 14, height: 14, borderRadius: '50%', background: '#fff', left: returnXai ? 18 : 2 }} />
                  </button>
                </div>

                {returnXai && (
                  <div className="pt-2 border-t border-[var(--border-subtle)]">
                    <label className="block text-[10px] font-mono font-bold text-amber-400 uppercase tracking-widest mb-1.5">
                      XAI Method
                    </label>
                    <select
                      value={xaiMethod}
                      onChange={(e: any) => setXaiMethod(e.target.value)}
                      className="instrument-select w-full px-3 py-1.5"
                    >
                      <option value="ig">Integrated Gradients (IG)</option>
                      <option value="shap">Gradient SHAP</option>
                      <option value="rollout">Attention Rollout</option>
                    </select>
                  </div>
                )}
              </div>

              {(validationError || predictionError) && (
                <div className="bg-rose-500/10 border border-rose-500/25 text-rose-300 p-3.5 rounded-xl text-xs flex items-start gap-2 backdrop-blur-sm">
                  <AlertCircle className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />
                  <span>{validationError || predictionError}</span>
                </div>
              )}

              <button
                type="submit"
                disabled={isPredicting || !modelLoaded}
                className="btn-primary-action w-full py-3.5 flex items-center justify-center gap-2.5 text-sm cursor-pointer shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isPredicting ? (
                  <span className="font-mono text-xs animate-pulse">PROCESSING_FORWARD_PASS...</span>
                ) : (
                  <>
                    <Play className="h-4 w-4 fill-white" />
                    <span>Run Inference</span>
                  </>
                )}
              </button>
            </form>
          </section>
        </div>

        {/* RIGHT 2 COLS: 3D Visualization, Sequence Strip & Results */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          {isPredicting ? (
            <PredictionLoadingState jobStatus={jobStatus} />
          ) : activePrediction ? (
            <section className="surface-tier-1 p-6 flex flex-col gap-6 relative overflow-hidden animate-spring-up">
              
              {/* Header Bar + Export Buttons */}
              <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-[var(--border-muted)]">
                <div>
                  <div className="flex items-center gap-2 mb-0.5">
                    <Sparkles className="h-3.5 w-3.5 text-amber-400" />
                    <span className="text-[10px] font-mono font-bold uppercase tracking-widest text-amber-400">
                      PREDICTION COMPLETE
                    </span>
                  </div>
                  <p className="text-xs font-mono text-[var(--text-muted)] truncate max-w-[280px]">
                    ID: <strong className="text-slate-200">{activePrediction?.protein_id || '--'}</strong>
                  </p>
                </div>

                <div className="flex items-center gap-2.5">
                  {/* Motif Search */}
                  <div className="relative flex items-center">
                    <Search className="h-3.5 w-3.5 text-[var(--text-muted)] absolute left-3" />
                    <input
                      type="text"
                      placeholder="Search motif (e.g. HHH)..."
                      value={searchPattern}
                      onChange={(e) => setSearchPattern(e.target.value)}
                      className="instrument-input pl-8 pr-3 py-1 text-xs w-44"
                    />
                  </div>

                  {/* Exporters */}
                  <button
                    onClick={exportPredictionCSV}
                    className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-teal-500/10 border border-teal-500/30 text-teal-400 text-xs font-mono font-bold hover:bg-teal-500/20 transition-all cursor-pointer"
                    title="Export CSV dataset report"
                  >
                    <FileSpreadsheet className="h-3.5 w-3.5" />
                    <span>CSV</span>
                  </button>

                  <button
                    onClick={exportPredictionJSON}
                    className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-purple-500/10 border border-purple-500/30 text-violet-400 text-xs font-mono font-bold hover:bg-purple-500/20 transition-all cursor-pointer"
                    title="Export JSON response"
                  >
                    <Download className="h-3.5 w-3.5" />
                    <span>JSON</span>
                  </button>
                </div>
              </div>

              {/* Physicochemical Profile Bar */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 p-4 rounded-xl surface-tier-3 text-xs font-mono">
                <div>
                  <span className="text-[9px] text-[var(--text-muted)] font-bold uppercase tracking-wider block">Mol. Weight</span>
                  <span className="text-slate-200 font-bold">{physProfile.mw.toFixed(2)} kDa</span>
                </div>
                <div>
                  <span className="text-[9px] text-[var(--text-muted)] font-bold uppercase tracking-wider block">Isoelectric Point</span>
                  <span className="text-teal-400 font-bold">pI {(physProfile.pI ?? 7.0).toFixed(2)}</span>
                </div>
                <div>
                  <span className="text-[9px] text-[var(--text-muted)] font-bold uppercase tracking-wider block">Net Charge (pH 7)</span>
                  <span className="text-amber-400 font-bold">{physProfile.netCharge > 0 ? `+${physProfile.netCharge}` : physProfile.netCharge}</span>
                </div>
                <div>
                  <span className="text-[9px] text-[var(--text-muted)] font-bold uppercase tracking-wider block">Hydrophobicity</span>
                  <span className="text-violet-400 font-bold">{physProfile.hydrophobicPct.toFixed(1)}%</span>
                </div>
              </div>

              {/* 3D Structure Viewer Viewport */}
              <Suspense fallback={<div className="h-[380px] w-full flex items-center justify-center surface-tier-3">Loading 3D Visualizer...</div>}>
                <ProteinStructure3D
                  sequence={activePrediction?.sequence ?? null}
                  q3Prediction={activePrediction?.q3_prediction ?? null}
                  q8Prediction={activePrediction?.q8_prediction ?? null}
                  confidence={activePrediction?.confidence ?? null}
                  residueImportance={activePrediction?.residue_importance ?? null}
                  hoveredIndex={hoveredResidue?.index ?? null}
                  isPredicting={isPredicting}
                  onHoverResidue={setHoveredResidueByIndex}
                  viewMode={stripViewMode as any}
                  onViewModeChange={(m) => setStripViewMode(m as any)}
                  classificationMode={classificationMode}
                  onClassificationModeChange={setClassificationMode}
                />
              </Suspense>

              {/* Sequence Strip Controls */}
              <div className="flex items-center justify-between mt-1">
                <span className="text-xs font-bold text-[var(--text-primary)] font-mono uppercase tracking-wider">
                  Interactive Sequence Strip
                </span>
                <div className="flex items-center gap-1 bg-black/40 p-1 rounded-xl border border-white/10 text-[10px] font-mono">
                  <button
                    onClick={() => setStripViewMode('structure')}
                    className={`px-2.5 py-0.5 rounded-lg font-bold transition-all cursor-pointer ${
                      stripViewMode === 'structure' ? 'bg-violet-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    Structure Colors
                  </button>
                  <button
                    onClick={() => setStripViewMode('xai')}
                    disabled={!activePrediction.residue_importance}
                    className={`px-2.5 py-0.5 rounded-lg font-bold transition-all cursor-pointer ${
                      stripViewMode === 'xai' ? 'bg-teal-500 text-black shadow' : 'text-slate-400 hover:text-slate-200 disabled:opacity-30'
                    }`}
                  >
                    XAI Attributions
                  </button>
                </div>
              </div>

              <StackedSequenceStrip
                sequence={activePrediction.sequence}
                q3Prediction={activePrediction.q3_prediction}
                q8Prediction={activePrediction.q8_prediction}
                confidence={activePrediction.confidence}
                residueImportance={activePrediction.residue_importance}
                activeTab={stripViewMode}
                activeClassification={classificationMode}
                hoveredIndex={hoveredResidue?.index ?? null}
                searchPattern={searchPattern}
                onHover={setHoveredResidueByIndex}
                onSelectClassification={setClassificationMode}
              />

              {/* Statistical Breakdown Panels */}
              <StatisticalBreakdownPanels
                q3Prediction={activePrediction.q3_prediction}
                q8Prediction={activePrediction.q8_prediction}
                confidence={activePrediction.confidence}
              />

              {/* Per-Residue Probability Distribution Inspector */}
              {hoveredResidue && (
                <div className="p-4 rounded-2xl surface-tier-3 border border-teal-400/30 flex flex-wrap justify-between gap-4 animate-spring-up">
                  <div>
                    <h6 className="text-xs font-bold text-[var(--text-primary)] font-mono mb-1">
                      Residue {hoveredResidue.aa}{hoveredResidue.index + 1} Probability Distribution
                    </h6>
                    <div className="flex gap-4 text-[10px] font-mono text-[var(--text-muted)]">
                      <span>Conf: <span className="text-teal-400 font-bold">{(hoveredResidue.conf * 100).toFixed(1)}%</span></span>
                      <span>Q3: <span className="text-violet-400 font-bold">{hoveredResidue.q3}</span></span>
                      <span>Q8: <span className="text-amber-400 font-bold">{hoveredResidue.q8}</span></span>
                    </div>
                  </div>

                  <div className="flex gap-6 text-[10px] font-mono">
                    <div>
                      <span className="text-[var(--text-muted)] font-bold block mb-1">Q3 PROBABILITIES</span>
                      <div className="flex gap-2">
                        {['H', 'E', 'C'].map((cls, i) => {
                          const probVal = (hoveredResidue.q3_prob?.[i] ?? 0) * 100;
                          return (
                            <div key={cls} className="flex flex-col items-center">
                              <span className="text-slate-400 text-[8px] font-bold">{cls}</span>
                              <div className="w-3 h-8 bg-black/40 rounded overflow-hidden flex flex-col-reverse">
                                <div className="w-full bg-teal-400" style={{ height: `${probVal}%` }} />
                              </div>
                              <span className="text-[8px] text-teal-300 font-bold mt-0.5">{probVal.toFixed(0)}%</span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </section>
          ) : predictionError ? (
            <section className="surface-tier-2 p-8 flex flex-col items-center justify-center text-center gap-4 min-h-[400px]">
              <div className="p-4 rounded-full bg-rose-500/10 border border-rose-500/30">
                <AlertCircle className="h-10 w-10 text-rose-400" strokeWidth={1.8} />
              </div>
              <div>
                <h4 className="text-base font-bold text-[var(--text-primary)] font-mono">
                  Prediction Job Failed
                </h4>
                <p className="text-xs text-rose-300 font-mono mt-1 max-w-md">
                  {predictionError}
                </p>
              </div>
            </section>
          ) : null}
        </div>
      </div>
    </div>
  );
};
