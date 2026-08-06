import React, { useState, lazy, Suspense, useRef, useCallback } from 'react';
import { usePredictionStore } from '../store/usePredictionStore';
import { useModelStore } from '../store/useModelStore';
import { AttentionHeatmap } from '../components/AttentionHeatmap';
import { AnimatedCounter } from '../components/AnimatedCounter';

const ProteinStructure3D = lazy(() => import('../components/ProteinStructure3D').then(m => ({ default: m.ProteinStructure3D })));

import { STRUCTURE_COLORS, Q8_STRUCTURE_COLORS } from '../utils/colors';
import { ResponsiveContainer, LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip as ChartTooltip, CartesianGrid } from 'recharts';
import { Play, AlertCircle, Info, Dna, Sparkles, ChevronRight, Cpu, Zap } from 'lucide-react';

// Minority Q8 classes = lower reliability signal
const MINORITY_Q8 = new Set(['I', 'B', 'S']);

// Q3 hex colors for strip rendering
const Q3_HEX: Record<string, string> = {
  H: '#7B2FF7',
  E: '#00D9C0',
  C: '#64748B',
};

// =============================================================================
// STACKED SEQUENCE STRIP — Q3 (primary) above, Q8 (secondary) below
// =============================================================================
interface StackedStripProps {
  sequence: string;
  q3Prediction: string[];
  q8Prediction: string[];
  confidence: number[];
  residueImportance?: number[] | null;
  activeTab: string;
  hoveredIndex: number | null;
  onHover: (idx: number | null) => void;
}

const StackedSequenceStrip: React.FC<StackedStripProps> = ({
  sequence, q3Prediction, q8Prediction, confidence, residueImportance,
  activeTab, hoveredIndex, onHover,
}) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const chars = sequence.split('');
  const len = chars.length;

  return (
    <div className="flex flex-col gap-0 animate-spring-up">
      {/* Header bar */}
      <div className="flex items-center justify-between mb-2.5">
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
          <h5 className="text-xs font-bold uppercase tracking-widest text-slate-300" style={{ fontFamily: 'var(--font-heading)' }}>
            Sequence Structure Map
          </h5>
        </div>
        <div className="flex items-center gap-3">
          {Object.entries(STRUCTURE_COLORS).map(([k, cfg]) => (
            <span key={k} className="flex items-center gap-1.5 text-[10px] font-semibold text-slate-400">
              <span className="w-2 h-2 rounded-sm inline-block" style={{ backgroundColor: cfg.hex }} />
              {cfg.label}
            </span>
          ))}
        </div>
      </div>

      {/* Shared scroll container */}
      <div
        ref={scrollRef}
        className="residue-strip rounded-2xl relative"
        style={{ padding: '6px 6px 10px', background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.06)' }}
        onMouseLeave={() => onHover(null)}
      >
        {/* Q3 Primary Strip */}
        <div className="flex gap-[2px] min-w-max">
          {chars.map((aa, idx) => {
            const q3 = q3Prediction[idx] || 'C';
            const conf = confidence[idx] ?? 1;
            const importance = residueImportance?.[idx] ?? 0;
            const isHov = hoveredIndex === idx;
            const bgColor = activeTab === 'xai'
              ? `rgba(0, 217, 192, ${0.08 + importance * 0.5})`
              : (Q3_HEX[q3] || '#64748B');
            return (
              <div
                key={idx}
                className={`residue-block flex flex-col items-center justify-between select-none${isHov ? ' hovered' : ''}`}
                style={{
                  height: 54, backgroundColor: bgColor,
                  opacity: conf < 0.5 ? 0.45 : conf < 0.7 ? 0.75 : 1,
                  outline: isHov ? '2px solid rgba(255,255,255,0.9)' : '2px solid transparent',
                  outlineOffset: '1px',
                  boxShadow: isHov ? `0 0 12px ${bgColor}` : 'none',
                }}
                onMouseEnter={() => onHover(idx)}
                title={`${aa}${idx + 1} - Q3: ${q3} - Conf: ${(conf * 100).toFixed(0)}%`}
              >
                <span className="text-[8px] text-white/50 font-mono pt-[3px] leading-none">{idx + 1}</span>
                <span className="text-[11px] text-white font-bold font-mono leading-none">{aa}</span>
                <span className="text-[9px] font-bold leading-none pb-[3px]" style={{ color: 'rgba(255,255,255,0.8)' }}>{q3}</span>
              </div>
            );
          })}
        </div>

        {/* Connecting bridge Q3 -> Q8 with Amber accent */}
        <div className="relative h-[6px] min-w-max my-[2px]" style={{ width: len * 26 + (len - 1) * 2 }}>
          <div className="absolute inset-0 rounded-none" style={{ background: 'linear-gradient(90deg, rgba(123,47,247,0.3), rgba(0,217,192,0.3), rgba(255,179,71,0.3))' }} />
          {hoveredIndex !== null && (
            <div
              className="absolute top-0 bottom-0 rounded"
              style={{
                left: hoveredIndex * 26, width: 24,
                background: '#FFB347',
                boxShadow: '0 0 10px #FFB347',
                transition: 'left 0.05s linear'
              }}
            />
          )}
        </div>

        {/* Q8 Secondary Strip */}
        <div className="flex gap-[2px] min-w-max">
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
                  height: 40, backgroundColor: colorDef.hex,
                  opacity: isMinority ? 0.35 : conf < 0.6 ? 0.55 : 0.85,
                  outline: isHov ? '2px solid rgba(255,255,255,0.8)' : '2px solid transparent',
                  outlineOffset: '1px',
                  filter: isMinority ? 'saturate(0.4)' : undefined,
                }}
                onMouseEnter={() => onHover(idx)}
                title={`Q8: ${q8}${isMinority ? ' - Minority class (low reliability)' : ''}`}
              >
                <span className="text-[9px] font-bold text-white/90 leading-none pt-[4px] font-mono">{q8}</span>
                {isMinority && (
                  <span className="text-[7px] font-bold leading-none pb-[3px]" style={{ color: '#FFB347' }}>!</span>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Q8 legend */}
      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 px-1">
        {Object.entries(Q8_STRUCTURE_COLORS).map(([k, cfg]) => {
          const isMinority = MINORITY_Q8.has(k);
          return (
            <span key={k} className="flex items-center gap-1 text-[9px] font-semibold font-mono" style={{ color: isMinority ? '#475569' : '#94A3B8' }}>
              <span className="w-1.5 h-1.5 rounded-sm inline-block" style={{ backgroundColor: cfg.hex, opacity: isMinority ? 0.4 : 1 }} />
              {cfg.label}{isMinority ? ' *' : ''}
            </span>
          );
        })}
        <span className="text-[9px] text-amber-400/80 font-mono ml-1">* Minority class — desaturated (lower confidence)</span>
      </div>
    </div>
  );
};

// =============================================================================
// COMPUTING ACTIVE LOADING STATE — Scan line + HUD progress
// =============================================================================
const PredictionLoadingState: React.FC<{ jobStatus: string | null }> = ({ jobStatus }) => (
  <div
    className="flex-1 flex flex-col items-center justify-center rounded-3xl p-12 text-center min-h-[500px] relative overflow-hidden bg-mesh-panel"
    style={{ border: '1px solid rgba(0, 217, 192, 0.25)' }}
  >
    {/* Active Laser Computing Bar */}
    <div className="animate-laser-scan" />

    <div className="flex items-end gap-[4px] mb-8 p-4 rounded-2xl bg-black/40 border border-white/[0.08]">
      {Array.from({ length: 32 }, (_, i) => (
        <div
          key={i}
          className="rounded-sm"
          style={{
            width: 6,
            height: 16 + Math.sin(i * 0.6) * 14,
            backgroundColor: i < 10 ? '#7B2FF7' : i < 22 ? '#00D9C0' : '#FFB347',
            opacity: 0.3 + (i / 31) * 0.7,
            animation: `residue-pulse 1.1s ease-in-out ${i * 0.04}s infinite`,
            boxShadow: `0 0 6px ${i < 10 ? '#7B2FF7' : i < 22 ? '#00D9C0' : '#FFB347'}60`,
          }}
        />
      ))}
    </div>

    <div className="flex items-center gap-2 mb-2">
      <Zap className="h-4 w-4 text-amber-400 animate-bounce" />
      <h5 className="text-base font-bold text-slate-100" style={{ fontFamily: 'var(--font-heading)' }}>
        {jobStatus === 'pending' && 'Job Queued — Waiting for Worker Slot'}
        {jobStatus === 'processing' && 'Computing Integrated Gradients Attribution'}
        {!jobStatus && 'ESM-2 Neural Inference Active'}
      </h5>
    </div>

    <p className="text-xs font-mono text-slate-400 max-w-md mx-auto leading-relaxed mb-6">
      {jobStatus === 'pending' && 'Sequence submitted. Waiting for backend inference worker slot...'}
      {jobStatus === 'processing' && 'Running 50 interpolation steps across ESM-2 hidden layers (~13s)...'}
      {!jobStatus && 'Tokenizing sequence -> ESM-2 (650M) -> Multi-Scale CNN -> BiLSTM -> 8-Head Attention...'}
    </p>

    <div className="flex items-center gap-2">
      {['Tokenizing', 'ESM-2 (650M)', 'BiLSTM', jobStatus === 'processing' ? 'XAI (IG)' : 'Dual Softmax'].map((step, i) => (
        <React.Fragment key={step}>
          <span
            className="text-[10px] font-mono font-bold px-3 py-1 rounded-xl"
            style={{
              background: i === 0 ? 'rgba(0, 217, 192, 0.15)' : 'rgba(255,255,255,0.04)',
              color: i === 0 ? '#00D9C0' : '#64748B',
              border: `1px solid ${i === 0 ? 'rgba(0, 217, 192, 0.4)' : 'rgba(255,255,255,0.06)'}`,
            }}
          >
            {step}
          </span>
          {i < 3 && <ChevronRight className="h-3 w-3 text-slate-700" />}
        </React.Fragment>
      ))}
    </div>
  </div>
);

// =============================================================================
// EMPTY STATE
// =============================================================================
const PredictionEmptyState: React.FC = () => (
  <div
    className="flex-1 flex flex-col items-center justify-center rounded-3xl p-14 text-center min-h-[500px] relative overflow-hidden bg-mesh-panel"
    style={{ border: '1px solid rgba(255,255,255,0.06)' }}
  >
    <div
      className="absolute"
      style={{
        width: 400, height: 400, borderRadius: '50%',
        background: 'radial-gradient(circle, rgba(123,47,247,0.1) 0%, rgba(0,217,192,0.05) 45%, transparent 70%)',
        top: '50%', left: '50%', transform: 'translate(-50%,-50%)', pointerEvents: 'none',
      }}
    />
    <div
      className="relative mb-6 p-5 rounded-2xl"
      style={{
        background: 'linear-gradient(135deg, rgba(123,47,247,0.15), rgba(0,217,192,0.08))',
        border: '1px solid rgba(123,47,247,0.3)',
        boxShadow: '0 0 40px rgba(123,47,247,0.15)',
      }}
    >
      <Dna className="h-9 w-9" style={{ color: '#00D9C0' }} />
    </div>
    <h5 className="text-base font-bold text-slate-200 mb-2" style={{ fontFamily: 'var(--font-heading)' }}>
      Awaiting Protein Sequence
    </h5>
    <p className="text-xs text-slate-400 max-w-sm mx-auto leading-relaxed font-sans mb-6">
      Paste an amino acid sequence or FASTA input to launch ESM-2 powered secondary structure prediction with residue-level XAI attributions.
    </p>
    <div className="flex flex-wrap gap-2 justify-center">
      {['Q3 Structure', 'Q8 Detailed', '3D Cartoon Ribbon', 'XAI Attributions', '8-Head Attention'].map(chip => (
        <span
          key={chip}
          className="text-[10px] font-mono font-semibold px-3 py-1 rounded-xl"
          style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)', color: '#94A3B8' }}
        >
          {chip}
        </span>
      ))}
    </div>
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
  const [activeTab, setActiveTab] = useState<'overview' | 'confidence' | 'attention' | 'xai'>('overview');
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

  const handleSequenceChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setSequence(e.target.value);
    setValidationError(null);
  };

  const cleanInputSequence = (input: string): string => {
    let clean = input.trim();
    const lines = clean.split('\n');
    if (lines[0].startsWith('>')) clean = lines.slice(1).join('');
    else clean = lines.join('');
    return clean.toUpperCase().replace(/\s/g, '').replace(/[-*]/g, '');
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setValidationError(null);
    const cleaned = cleanInputSequence(sequence);
    if (cleaned.length < 5) { setValidationError('Sequence must be at least 5 amino acids long.'); return; }
    if (cleaned.length > 2048) { setValidationError('Sequence exceeds maximum length of 2048 residues.'); return; }
    const validChars = new Set('ACDEFGHIKLMNPQRSTVWYBZXJOU');
    const invalid = [...cleaned].filter(c => !validChars.has(c));
    if (invalid.length > 0) {
      setValidationError(`Invalid amino acid characters: ${Array.from(new Set(invalid)).join(', ')}`);
      return;
    }
    try { await runPredict({ sequence: cleaned, return_attention: returnAttention, return_xai: returnXai, xai_method: xaiMethod }); }
    catch { /* handled by store */ }
  };

  const chartData = activePrediction
    ? activePrediction.sequence.split('').map((char, i) => ({
        index: i + 1,
        residue: `${char}${i + 1}`,
        confidence: activePrediction.confidence[i],
        importance: activePrediction.residue_importance?.[i] || 0,
      }))
    : [];

  const getQ3Breakdown = () => {
    if (!activePrediction) return { H: 0, E: 0, C: 0 };
    const counts = { H: 0, E: 0, C: 0 };
    activePrediction.q3_prediction.forEach(c => { if (c in counts) counts[c as 'H' | 'E' | 'C']++; });
    const len = activePrediction.length;
    return {
      H: (counts.H / len * 100),
      E: (counts.E / len * 100),
      C: (counts.C / len * 100),
    };
  };
  const q3Breakdown = getQ3Breakdown();

  const tabStyle = (isActive: boolean): React.CSSProperties => ({
    fontFamily: 'var(--font-heading)',
    background: isActive ? 'rgba(0, 217, 192, 0.15)' : 'transparent',
    border: `1px solid ${isActive ? 'rgba(0, 217, 192, 0.35)' : 'transparent'}`,
    color: isActive ? '#66E8D5' : '#64748B',
    borderRadius: 10,
    padding: '6px 16px',
    fontSize: 11,
    fontWeight: 700,
    cursor: 'pointer',
    transition: 'all 0.15s var(--ease-spring)',
    letterSpacing: '0.02em',
  });

  const avgConfidence = activePrediction
    ? (activePrediction.confidence.reduce((a, b) => a + b, 0) / activePrediction.length * 100)
    : 0;

  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* LEFT: Scientific Controls Panel */}
        <div className="lg:col-span-1">
          <section className="glass-instrument-card rounded-3xl p-6 flex flex-col gap-5 relative overflow-hidden">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl" style={{ background: 'rgba(123,47,247,0.15)', border: '1px solid rgba(123,47,247,0.3)' }}>
                <Cpu className="h-4 w-4 text-violet-400" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-slate-100" style={{ fontFamily: 'var(--font-heading)' }}>
                  Inference Controls
                </h3>
                <span className="text-[10px] font-mono text-slate-500">INPUT_PARAM // ESM2_650M</span>
              </div>
            </div>

            <form onSubmit={handleSubmit} className="flex flex-col gap-4">
              <div>
                <label className="block text-[10px] font-mono font-bold uppercase tracking-widest mb-2 text-slate-400">
                  Amino Acid Sequence (FASTA)
                </label>
                <textarea
                  value={sequence}
                  onChange={handleSequenceChange}
                  placeholder={">MyProtein\nMKFLILLFNILCLFPVLA..."}
                  className="w-full h-44 p-4 text-xs font-mono text-slate-100 placeholder-slate-700 resize-none rounded-2xl transition-all duration-200"
                  style={{ background: 'rgba(0,0,0,0.4)', border: '1px solid rgba(255,255,255,0.08)', outline: 'none', lineHeight: 1.6 }}
                  onFocus={e => { e.target.style.borderColor = 'rgba(0,217,192,0.5)'; e.target.style.boxShadow = '0 0 0 3px rgba(0,217,192,0.1)'; }}
                  onBlur={e => { e.target.style.borderColor = 'rgba(255,255,255,0.08)'; e.target.style.boxShadow = 'none'; }}
                  required
                />
              </div>

              {/* Toggle Options */}
              <div className="p-4 rounded-2xl flex flex-col gap-4 bg-black/40 border border-white/[0.06]">
                {[
                  { label: 'Compute Self-Attention Matrix', id: 'attention', checked: returnAttention, onChange: setReturnAttention },
                  { label: 'Compute XAI Attributions (IG)', id: 'xai-attrs', checked: returnXai, onChange: setReturnXai },
                ].map(({ label, id, checked, onChange }) => (
                  <div key={id} className="flex items-center justify-between gap-3">
                    <span className="text-xs font-semibold text-slate-300">{label}</span>
                    <button
                      type="button"
                      role="switch"
                      aria-checked={checked}
                      id={`toggle-${id}`}
                      onClick={() => onChange(!checked)}
                      className="relative flex-shrink-0"
                      style={{
                        width: 38, height: 22, borderRadius: 11, cursor: 'pointer',
                        background: checked ? 'linear-gradient(135deg, #7B2FF7, #00D9C0)' : 'rgba(255,255,255,0.08)',
                        border: `1px solid ${checked ? 'rgba(0,217,192,0.5)' : 'rgba(255,255,255,0.12)'}`,
                        boxShadow: checked ? '0 0 10px rgba(0,217,192,0.3)' : 'none',
                        transition: 'all 0.2s var(--ease-spring)',
                      }}
                    >
                      <span
                        className="absolute top-[2px]"
                        style={{
                          width: 16, height: 16, borderRadius: '50%', background: '#fff',
                          left: checked ? 19 : 3, transition: 'left 0.2s var(--ease-spring)',
                          boxShadow: '0 1px 3px rgba(0,0,0,0.4)',
                        }}
                      />
                    </button>
                  </div>
                ))}

                {returnXai && (
                  <div className="pt-3" style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                    <label className="block text-[10px] font-mono font-bold text-amber-400 uppercase tracking-widest mb-2">
                      XAI Method Selection
                    </label>
                    <select
                      value={xaiMethod}
                      onChange={(e: any) => setXaiMethod(e.target.value)}
                      className="w-full px-3 py-2 text-xs font-mono text-slate-200 rounded-xl"
                      style={{ background: 'rgba(0,0,0,0.5)', border: '1px solid rgba(255,255,255,0.1)', outline: 'none' }}
                    >
                      <option value="ig">Integrated Gradients (IG)</option>
                      <option value="shap">Gradient SHAP</option>
                      <option value="rollout">Attention Rollout</option>
                    </select>
                  </div>
                )}
              </div>

              {/* Error */}
              {(validationError || predictionError) && (
                <div
                  className="p-3.5 rounded-2xl flex items-start gap-2.5 animate-spring-up"
                  style={{ background: 'rgba(255,77,109,0.08)', border: '1px solid rgba(255,77,109,0.25)' }}
                >
                  <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" style={{ color: '#FF4D6D' }} />
                  <div>
                    <p className="text-[11px] font-bold mb-0.5" style={{ color: '#FF4D6D', fontFamily: 'var(--font-heading)' }}>Input Error</p>
                    <p className="text-xs text-rose-300">{validationError || predictionError}</p>
                  </div>
                </div>
              )}

              {/* Run Inference Button */}
              <button
                type="submit"
                id="submit-prediction"
                disabled={isPredicting || !modelLoaded}
                className="w-full font-bold text-sm py-3.5 rounded-2xl flex items-center justify-center gap-2.5 transition-all duration-200"
                style={{
                  fontFamily: 'var(--font-heading)',
                  background: isPredicting || !modelLoaded ? 'rgba(255,255,255,0.04)' : 'linear-gradient(135deg, #7B2FF7 0%, #00D9C0 100%)',
                  boxShadow: isPredicting || !modelLoaded ? 'none' : '0 4px 24px rgba(123,47,247,0.4), 0 0 0 1px rgba(0,217,192,0.3)',
                  color: isPredicting || !modelLoaded ? '#475569' : '#fff',
                  cursor: isPredicting || !modelLoaded ? 'not-allowed' : 'pointer',
                }}
              >
                {isPredicting ? (
                  <>
                    <div className="rounded-full border-2 border-white/20 border-t-amber-400" style={{ width: 16, height: 16, animation: 'spin 0.7s linear infinite' }} />
                    <span className="font-mono text-xs">
                      {jobStatus === 'pending' && 'QUEUE_PENDING...'}
                      {jobStatus === 'processing' && 'COMPUTING_XAI...'}
                      {!jobStatus && 'FORWARD_PASS...'}
                    </span>
                  </>
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

        {/* RIGHT: Instrument Output Panel */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          {isPredicting ? (
            <PredictionLoadingState jobStatus={jobStatus} />
          ) : activePrediction ? (
            <section className="glass-instrument-card rounded-3xl p-6 flex flex-col gap-6 relative overflow-hidden animate-spring-up">
              {/* Output Header with Animated Counters */}
              <div className="flex flex-wrap items-start justify-between gap-4 pb-4 border-b border-white/[0.08]">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <Sparkles className="h-3.5 w-3.5 text-amber-400" />
                    <span className="text-[10px] font-mono font-bold uppercase tracking-widest text-amber-400">
                      PREDICTION_COMPLETE
                    </span>
                  </div>
                  <p className="text-xs font-mono text-slate-400 truncate max-w-[260px]">
                    ID: {activePrediction?.protein_id || '--'}
                  </p>
                </div>
                <div className="flex gap-6">
                  <div className="text-right">
                    <span className="text-[9px] font-mono font-bold uppercase tracking-widest text-slate-500 block">Length</span>
                    <span className="text-sm font-bold font-mono text-slate-100 mt-0.5 block">
                      <AnimatedCounter value={activePrediction.length} suffix=" aa" />
                    </span>
                  </div>
                  <div className="text-right">
                    <span className="text-[9px] font-mono font-bold uppercase tracking-widest text-slate-500 block">Latency</span>
                    <span className="text-sm font-bold font-mono text-amber-400 mt-0.5 block">
                      <AnimatedCounter value={activePrediction.processing_time_ms} suffix=" ms" />
                    </span>
                  </div>
                  <div className="text-right">
                    <span className="text-[9px] font-mono font-bold uppercase tracking-widest text-slate-500 block">Avg Conf</span>
                    <span className="text-sm font-bold font-mono text-teal-300 mt-0.5 block">
                      <AnimatedCounter value={avgConfidence} decimals={1} suffix="%" />
                    </span>
                  </div>
                </div>
              </div>

              {/* 3D Viewer */}
              <Suspense fallback={
                <div className="h-[380px] w-full flex flex-col items-center justify-center rounded-2xl gap-3 bg-black/40 border border-white/[0.06]">
                  <div className="flex items-end gap-[3px]">
                    {[...Array(8)].map((_, i) => (
                      <div key={i} className="w-[5px] rounded-sm" style={{ height: 20, backgroundColor: '#7B2FF7', animation: `residue-pulse 1s ease-in-out ${i * 0.1}s infinite` }} />
                    ))}
                  </div>
                  <span className="text-xs font-mono text-slate-400">Loading 3D Visualizer...</span>
                </div>
              }>
                <ProteinStructure3D
                  sequence={activePrediction?.sequence ?? null}
                  q3Prediction={activePrediction?.q3_prediction ?? null}
                  q8Prediction={activePrediction?.q8_prediction ?? null}
                  confidence={activePrediction?.confidence ?? null}
                  residueImportance={activePrediction?.residue_importance ?? null}
                  hoveredIndex={hoveredResidue?.index ?? null}
                  isPredicting={isPredicting}
                  onHoverResidue={setHoveredResidueByIndex}
                />
              </Suspense>

              {/* STACKED Q3 / Q8 STRIPS */}
              <StackedSequenceStrip
                sequence={activePrediction.sequence}
                q3Prediction={activePrediction.q3_prediction}
                q8Prediction={activePrediction.q8_prediction}
                confidence={activePrediction.confidence}
                residueImportance={activePrediction.residue_importance}
                activeTab={activeTab}
                hoveredIndex={hoveredResidue?.index ?? null}
                onHover={setHoveredResidueByIndex}
              />

              {/* Hover residue detail banner */}
              {hoveredResidue ? (
                <div
                  className="p-4 rounded-2xl flex flex-wrap items-center justify-between gap-4 animate-spring-up bg-black/50"
                  style={{ border: '1px solid rgba(0, 217, 192, 0.3)', boxShadow: '0 0 20px rgba(0, 217, 192, 0.1)' }}
                >
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-xl text-center w-12 font-mono bg-teal-500/10 border border-teal-500/30">
                      <span className="text-[9px] font-bold text-teal-300 block leading-none mb-1">AA</span>
                      <span className="text-base font-bold text-white leading-none block">{hoveredResidue.aa}{hoveredResidue.index + 1}</span>
                    </div>
                    <div>
                      <h6 className="text-xs font-bold text-slate-100 flex items-center gap-2 flex-wrap">
                        Conf: {(hoveredResidue.conf * 100).toFixed(1)}%
                        {MINORITY_Q8.has(hoveredResidue.q8) && (
                          <span className="text-[9px] px-2 py-0.5 rounded font-mono font-bold uppercase text-amber-400 bg-amber-400/10 border border-amber-400/30">
                            Minority Class (Low Conf)
                          </span>
                        )}
                      </h6>
                      <div className="flex gap-4 text-[10px] font-mono text-slate-400 mt-1">
                        <span>Q3: <span className="text-violet-400 font-bold">{hoveredResidue.q3}</span></span>
                        <span>Q8: <span className="text-teal-300 font-bold">{hoveredResidue.q8}</span></span>
                        {hoveredResidue.importance !== undefined && activePrediction.residue_importance && (
                          <span>XAI: <span className="text-emerald-400 font-bold">{hoveredResidue.importance.toFixed(4)}</span></span>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="flex gap-6 text-[10px] font-mono text-slate-400">
                    <div>
                      <span className="text-slate-500 font-bold block mb-0.5">Q3 PROBS</span>
                      <span>H:{hoveredResidue.q3_prob[0].toFixed(2)} | E:{hoveredResidue.q3_prob[1].toFixed(2)} | C:{hoveredResidue.q3_prob[2].toFixed(2)}</span>
                    </div>
                    <div>
                      <span className="text-slate-500 font-bold block mb-0.5">Q8 PROBS</span>
                      <span>H:{hoveredResidue.q8_prob[0].toFixed(2)} | E:{hoveredResidue.q8_prob[1].toFixed(2)} | C:{hoveredResidue.q8_prob[7].toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="p-3 rounded-2xl flex items-center justify-center gap-2 bg-black/30 border border-white/[0.04]">
                  <Info className="h-3.5 w-3.5 text-slate-500" />
                  <span className="text-xs text-slate-500 font-mono">Hover residue blocks for structural probability readouts</span>
                </div>
              )}

              {/* Visualization Tabs */}
              <div>
                <div className="flex gap-2 mb-4 pb-3 border-b border-white/[0.06]">
                  {[
                    { key: 'overview', label: 'Structure Breakdown' },
                    { key: 'confidence', label: 'Confidence Curve' },
                    ...(activePrediction.attention_map ? [{ key: 'attention', label: 'Attention Matrix' }] : []),
                    ...(activePrediction.residue_importance ? [{ key: 'xai', label: 'XAI Attributions' }] : []),
                  ].map(({ key, label }) => (
                    <button key={key} onClick={() => setActiveTab(key as any)} style={tabStyle(activeTab === key)}>{label}</button>
                  ))}
                </div>

                <div className="min-h-[240px] flex flex-col justify-center">
                  {activeTab === 'overview' && (
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      {[
                        { key: 'H', label: 'Alpha Helix', val: q3Breakdown.H, color: '#7B2FF7', border: 'rgba(123,47,247,0.3)' },
                        { key: 'E', label: 'Beta Sheet',  val: q3Breakdown.E, color: '#00D9C0', border: 'rgba(0,217,192,0.3)' },
                        { key: 'C', label: 'Coil / Loop', val: q3Breakdown.C, color: '#64748B', border: 'rgba(100,116,139,0.3)' },
                      ].map(({ key, label, val, color, border }) => (
                        <div key={key} className="p-5 rounded-2xl text-center bg-black/40" style={{ border: `1px solid ${border}` }}>
                          <span className="text-[10px] font-mono font-bold uppercase tracking-widest block" style={{ color }}>{label}</span>
                          <p className="text-4xl font-extrabold font-mono mt-2" style={{ color }}>
                            <AnimatedCounter value={val} decimals={0} suffix="%" />
                          </p>
                        </div>
                      ))}
                    </div>
                  )}

                  {activeTab === 'confidence' && (
                    <div className="h-[240px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData} margin={{ top: 8, right: 8, left: -24, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                          <XAxis dataKey="index" stroke="#475569" fontSize={9} />
                          <YAxis domain={[0, 1]} stroke="#475569" fontSize={9} />
                          <ChartTooltip contentStyle={{ backgroundColor: '#0A0E17', borderColor: 'rgba(123,47,247,0.4)', borderRadius: 12, fontSize: 11 }} labelStyle={{ color: '#94A3B8' }} itemStyle={{ color: '#00D9C0' }} />
                          <Line type="monotone" dataKey="confidence" stroke="#00D9C0" strokeWidth={2} dot={false} activeDot={{ r: 5, fill: '#FFB347' }} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  )}

                  {activeTab === 'attention' && activePrediction.attention_map && (
                    <AttentionHeatmap attentionMap={activePrediction.attention_map} sequence={activePrediction.sequence} onHoverResidue={setHoveredResidueByIndex} />
                  )}

                  {activeTab === 'xai' && activePrediction.residue_importance && (
                    <div className="h-[240px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={chartData} margin={{ top: 8, right: 8, left: -24, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                          <XAxis dataKey="index" stroke="#475569" fontSize={9} />
                          <YAxis stroke="#475569" fontSize={9} />
                          <ChartTooltip contentStyle={{ backgroundColor: '#0A0E17', borderColor: 'rgba(0,217,192,0.4)', borderRadius: 12, fontSize: 11 }} labelStyle={{ color: '#94A3B8' }} itemStyle={{ color: '#FFB347' }} />
                          <Bar dataKey="importance" fill="#FFB347" radius={[3, 3, 0, 0]} opacity={0.85} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>
              </div>
            </section>
          ) : (
            <PredictionEmptyState />
          )}
        </div>
      </div>
    </div>
  );
};
