import React, { useState, lazy, Suspense } from 'react';
import { usePredictionStore } from '../store/usePredictionStore';
import { useModelStore } from '../store/useModelStore';
import { AttentionHeatmap } from '../components/AttentionHeatmap';

const ProteinStructure3D = lazy(() => import('../components/ProteinStructure3D').then(m => ({ default: m.ProteinStructure3D })));

import { STRUCTURE_COLORS } from '../utils/colors';
import { ResponsiveContainer, LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip as ChartTooltip, CartesianGrid } from 'recharts';
import { Activity, Play, AlertCircle, Info, HelpCircle } from 'lucide-react';

export const Predict: React.FC = () => {
  const { runPredict, activePrediction, isPredicting, predictionError, jobStatus } = usePredictionStore();
  const { modelLoaded } = useModelStore();

  const [sequence, setSequence] = useState(activePrediction?.sequence || '');
  const [returnAttention, setReturnAttention] = useState(false);
  const [returnXai, setReturnXai] = useState(false);
  const [xaiMethod, setXaiMethod] = useState<'ig' | 'shap' | 'rollout'>('ig');
  const [validationError, setValidationError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'confidence' | 'attention' | 'xai'>('overview');
  const [hoveredResidue, setHoveredResidue] = useState<{ index: number; aa: string; q3: string; q8: string; q3_prob: number[]; q8_prob: number[]; conf: number; importance?: number } | null>(null);

  // Helper to set hovered residue details based on index (used by 3D and Heatmap)
  const setHoveredResidueByIndex = (idx: number | null) => {
    if (idx === null || !activePrediction) {
      setHoveredResidue(null);
      return;
    }
    const char = activePrediction.sequence[idx];
    const q3 = activePrediction.q3_prediction[idx];
    const q8 = activePrediction.q8_prediction[idx];
    const conf = activePrediction.confidence[idx];
    const q3_prob = activePrediction.q3_probabilities[idx];
    const q8_prob = activePrediction.q8_probabilities[idx];
    const importance = activePrediction.residue_importance?.[idx] || 0;
    setHoveredResidue({ index: idx, aa: char, q3, q8, q3_prob, q8_prob, conf, importance });
  };

  // Validate sequence character and length constraints
  const handleSequenceChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const val = e.target.value;
    setSequence(val);
    setValidationError(null);
  };

  const cleanInputSequence = (input: string): string => {
    let clean = input.trim();
    // Strip FASTA header if present
    const lines = clean.split('\n');
    if (lines[0].startsWith('>')) {
      clean = lines.slice(1).join('');
    } else {
      clean = lines.join('');
    }
    return clean.toUpperCase().replace(/\s/g, '').replace(/[-*]/g, '');
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setValidationError(null);

    const cleaned = cleanInputSequence(sequence);
    
    if (cleaned.length < 5) {
      setValidationError('Sequence must be at least 5 amino acids long.');
      return;
    }
    if (cleaned.length > 2048) {
      setValidationError('Sequence exceeds maximum length of 2048 residues.');
      return;
    }

    const validChars = new Set('ACDEFGHIKLMNPQRSTVWYBZXJOU');
    const invalid = [...cleaned].filter(char => !validChars.has(char));
    if (invalid.length > 0) {
      setValidationError(`Invalid amino acid characters: ${Array.from(new Set(invalid)).join(', ')}`);
      return;
    }

    try {
      await runPredict({
        sequence: cleaned,
        return_attention: returnAttention,
        return_xai: returnXai,
        xai_method: xaiMethod,
      });
    } catch {
      // Handled by store
    }
  };

  // Helper: map secondary structure labels to Tailwind bg colors (driven by colors.ts)
  const getQ3ColorClass = (char: string) => {
    const q3 = (char === 'H' || char === 'E' || char === 'C') ? char : 'C';
    return STRUCTURE_COLORS[q3].bgClass;
  };

  const getQ8ColorClass = (char: string) => {
    switch (char) {
      case 'H': return 'bg-blue-500/20 border-blue-500/40 text-blue-400'; // Alpha helix (Blue)
      case 'G': return 'bg-cyan-500/20 border-cyan-500/40 text-cyan-400';  // 3-10 helix (Cyan)
      case 'I': return 'bg-indigo-950/40 border-indigo-900/10 text-indigo-500/50 opacity-60'; // Pi helix (Muted)
      
      case 'E': return 'bg-orange-500/20 border-orange-500/40 text-orange-400'; // Beta strand (Orange)
      case 'B': return 'bg-amber-950/40 border-amber-900/10 text-amber-500/50 opacity-60';  // Beta bridge (Muted)
      
      case 'T': return 'bg-slate-500/20 border-slate-500/40 text-slate-400'; // Turn (Light Grey)
      case 'S': return 'bg-zinc-900/40 border-zinc-800/10 text-zinc-550 opacity-60';   // Bend (Muted)
      case 'C':
      default: return 'bg-slate-700/20 border-slate-700/40 text-slate-400'; // Coil (Slate)
    }
  };

  // Prepare chart data
  const chartData = activePrediction
    ? activePrediction.sequence.split('').map((char, i) => ({
        index: i + 1,
        residue: `${char}${i + 1}`,
        confidence: activePrediction.confidence[i],
        importance: activePrediction.residue_importance?.[i] || 0,
      }))
    : [];

  // Q3 Breakdown calculation
  const getQ3Breakdown = () => {
    if (!activePrediction) return { H: 0, E: 0, C: 0 };
    const counts = { H: 0, E: 0, C: 0 };
    activePrediction.q3_prediction.forEach(c => {
      if (c in counts) counts[c as 'H'|'E'|'C']++;
    });
    const len = activePrediction.length;
    return {
      H: (counts.H / len * 100).toFixed(0),
      E: (counts.E / len * 100).toFixed(0),
      C: (counts.C / len * 100).toFixed(0),
    };
  };

  const q3Breakdown = getQ3Breakdown();

  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Form Panel */}
        <div className="lg:col-span-1">
          <section className="bg-[#0f172a]/30 border border-slate-900 rounded-3xl p-6 relative overflow-hidden">
            <h3 className="text-lg font-bold text-slate-200 mb-4 flex items-center gap-2">
              <Activity className="h-5 w-5 text-purple-400" />
              Prediction Settings
            </h3>

            <form onSubmit={handleSubmit} className="flex flex-col gap-4">
              <div>
                <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                  Protein Sequence (FASTA or Raw AA)
                </label>
                <textarea
                  value={sequence}
                  onChange={handleSequenceChange}
                  placeholder="Paste amino acids (e.g. MKFLILLFNILCLFPVL...)"
                  className="w-full h-40 bg-slate-950/60 border border-slate-800 rounded-2xl p-4 text-sm font-mono text-slate-100 placeholder-slate-600 focus:outline-none focus:border-purple-500/50 resize-none transition-all"
                  required
                />
              </div>

              {/* Checkboxes */}
              <div className="bg-slate-900/40 border border-slate-850/50 p-4 rounded-2xl flex flex-col gap-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-300">Compute Self-Attention</span>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={returnAttention}
                      onChange={(e) => setReturnAttention(e.target.checked)}
                      className="sr-only peer"
                    />
                    <div className="w-9 h-5 bg-slate-800 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-slate-300 after:border-slate-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-purple-600"></div>
                  </label>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-300">Compute XAI Attributions</span>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={returnXai}
                      onChange={(e) => setReturnXai(e.target.checked)}
                      className="sr-only peer"
                    />
                    <div className="w-9 h-5 bg-slate-800 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-slate-300 after:border-slate-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-purple-600"></div>
                  </label>
                </div>

                {returnXai && (
                  <div className="mt-2 pt-2 border-t border-slate-800/80">
                    <label className="block text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
                      XAI Method
                    </label>
                    <select
                      value={xaiMethod}
                      onChange={(e: any) => setXaiMethod(e.target.value)}
                      className="w-full bg-slate-950/60 border border-slate-800 rounded-xl px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-purple-500/50"
                    >
                      <option value="ig">Integrated Gradients (IG)</option>
                      <option value="shap">Gradient SHAP</option>
                      <option value="rollout">Attention Rollout</option>
                    </select>
                  </div>
                )}
              </div>

              {(validationError || predictionError) && (
                <div className="bg-rose-500/10 border border-rose-500/20 text-rose-300 p-3.5 rounded-2xl text-xs flex items-start gap-2">
                  <AlertCircle className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />
                  <span>{validationError || predictionError}</span>
                </div>
              )}

              <button
                type="submit"
                disabled={isPredicting || !modelLoaded}
                className="w-full bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 disabled:from-slate-800 disabled:to-slate-800 text-white font-semibold text-sm py-3 rounded-2xl shadow-lg shadow-purple-500/10 disabled:shadow-none hover:scale-[1.01] active:scale-[0.99] disabled:pointer-events-none transition-all flex items-center justify-center gap-2"
              >
                {isPredicting ? (
                  <>
                    <div className="h-4 w-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    <span>
                      {jobStatus === 'pending' && 'Job Queued...'}
                      {jobStatus === 'processing' && 'Calculating XAI...'}
                      {!jobStatus && 'Processing...'}
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

        {/* Right Columns: Inference Results */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          {isPredicting ? (
            <div className="flex-1 flex flex-col items-center justify-center border border-slate-800 bg-[#0f172a]/20 rounded-3xl p-12 text-center text-slate-500 min-h-[500px]">
              <div className="h-10 w-10 border-4 border-purple-500/30 border-t-purple-500 rounded-full animate-spin mb-4 animate-duration-1000"></div>
              <h5 className="text-sm font-bold text-slate-200">
                {jobStatus === 'pending' && 'Job in Queue'}
                {jobStatus === 'processing' && 'Calculating Attribution Map'}
                {!jobStatus && 'Predicting Secondary Structure'}
              </h5>
              <p className="text-xs text-slate-400 mt-2 max-w-sm mx-auto leading-relaxed">
                {jobStatus === 'pending' && 'Your prediction job is queued. Live ESM-2 inference and attribution calculation run asynchronously to prevent API blockage.'}
                {jobStatus === 'processing' && 'Running Integrated Gradients (50 interpolation steps). This process takes ~13 seconds on CPU.'}
                {!jobStatus && 'Tokenizing amino acid sequence and running forward pass...'}
              </p>
            </div>
          ) : activePrediction ? (
            <section className="bg-[#0f172a]/30 border border-slate-900 rounded-3xl p-6 flex flex-col gap-6 relative overflow-hidden">
              {/* Header Stats */}
              <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800/80 pb-4">
                <div>
                  <span className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider">Prediction Results</span>
                  <h4 className="text-sm font-bold text-slate-300 font-mono mt-0.5 truncate max-w-[280px]">
                    ID: {activePrediction?.protein_id || 'Generating ID...'}
                  </h4>
                </div>

                <div className="flex gap-4">
                  <div className="text-right">
                    <span className="text-[10px] text-slate-400 font-semibold uppercase">Length</span>
                    <p className="text-sm font-bold text-slate-200 mt-0.5">{activePrediction?.length || '-'} AAs</p>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] text-slate-400 font-semibold uppercase">Latency</span>
                    <p className="text-sm font-bold text-slate-200 mt-0.5">{activePrediction?.processing_time_ms || '-'} ms</p>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] text-slate-400 font-semibold uppercase">Avg Confidence</span>
                    <p className="text-sm font-bold text-slate-200 mt-0.5">
                      {activePrediction
                        ? `${(activePrediction.confidence.reduce((a, b) => a + b, 0) / activePrediction.length * 100).toFixed(0)}%`
                        : '-'}
                    </p>
                  </div>
                </div>
              </div>

              {/* 3D Protein Structure Viewer */}
              <Suspense fallback={
                <div className="h-[400px] w-full flex flex-col items-center justify-center border border-slate-800 bg-slate-950/20 rounded-2xl text-slate-400">
                  <div className="h-8 w-8 animate-spin rounded-full border-4 border-[#3A64E8] border-t-transparent mb-2"></div>
                  <span className="text-xs font-semibold">Loading 3D Visualizer...</span>
                </div>
              }>
                <ProteinStructure3D
                  sequence={activePrediction?.sequence ?? null}
                  q3Prediction={activePrediction?.q3_prediction ?? null}
                  confidence={activePrediction?.confidence ?? null}
                  hoveredIndex={hoveredResidue?.index ?? null}
                  isPredicting={isPredicting}
                />
              </Suspense>

              {activePrediction && (
                <>
                  {/* Grid Residue Map */}
              <div>
                <h5 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
                  Sequence Structure Layout
                </h5>
                <div className="flex flex-wrap gap-1.5 max-h-56 overflow-y-auto p-1 border border-slate-850 rounded-2xl bg-slate-950/20">
                  {activePrediction.sequence.split('').map((char, idx) => {
                    const q3 = activePrediction.q3_prediction[idx];
                    const q8 = activePrediction.q8_prediction[idx];
                    const conf = activePrediction.confidence[idx];
                    const q3_prob = activePrediction.q3_probabilities[idx];
                    const q8_prob = activePrediction.q8_probabilities[idx];
                    const importance = activePrediction.residue_importance?.[idx] || 0;
                    const isHovered = hoveredResidue?.index === idx;

                    const cellBg = activeTab === 'xai'
                      ? `rgba(16, 185, 129, ${0.05 + importance * 0.45})`
                      : undefined;
                    const cellBorder = activeTab === 'xai'
                      ? `rgba(16, 185, 129, ${0.15 + importance * 0.35})`
                      : undefined;

                    return (
                      <div
                        key={idx}
                        onMouseEnter={() => setHoveredResidue({ index: idx, aa: char, q3, q8, q3_prob, q8_prob, conf, importance })}
                        className={`flex flex-col items-center justify-center p-1.5 w-11 rounded-lg border text-center transition-all cursor-default select-none ${
                          isHovered ? 'scale-110 shadow-lg border-purple-500 ring-2 ring-purple-500/20' : 'border-slate-800'
                        }`}
                        style={activeTab === 'xai' ? { backgroundColor: cellBg, borderColor: cellBorder } : {}}
                      >
                        <span className="text-[10px] text-slate-500 font-bold leading-none">{idx + 1}</span>
                        <span className="text-sm font-bold text-white font-mono my-0.5">{char}</span>
                        <div className="flex gap-0.5 mt-0.5">
                          <span className={`w-3.5 h-3.5 text-[8px] font-bold rounded flex items-center justify-center border ${getQ3ColorClass(q3)}`}>
                            {q3}
                          </span>
                          <span className={`w-3.5 h-3.5 text-[8px] font-bold rounded flex items-center justify-center border ${getQ8ColorClass(q8)}`}>
                            {q8}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Hover residue detail banner */}
              {hoveredResidue ? (
                <div className="bg-purple-950/15 border border-purple-500/20 p-4 rounded-2xl flex flex-wrap items-center justify-between gap-4 backdrop-blur-sm animate-fadeIn">
                  <div className="flex items-center gap-3">
                    <div className="bg-purple-600/10 border border-purple-500/30 p-2 rounded-xl text-center w-12 font-mono">
                      <span className="text-[10px] text-purple-400 font-bold block leading-none">Residue</span>
                      <span className="text-base font-bold text-white leading-none mt-1 block">
                        {hoveredResidue.aa}{hoveredResidue.index + 1}
                      </span>
                    </div>

                    <div>
                      <h6 className="text-xs font-bold text-slate-200 flex items-center">
                        Confidence: {(hoveredResidue.conf * 100).toFixed(1)}%
                        {['I', 'B', 'S'].includes(hoveredResidue.q8) && (
                          <span className="ml-2 text-[8px] bg-rose-500/10 border border-rose-500/20 text-rose-400 px-1.5 py-0.5 rounded font-bold uppercase tracking-wider animate-pulse">
                            Minority Class (Low Reliability)
                          </span>
                        )}
                      </h6>
                      <div className="flex gap-3 text-[10px] font-semibold text-slate-400 mt-1">
                        <span>Q3 Class: <span className="text-amber-400">{hoveredResidue.q3}</span></span>
                        <span>Q8 Class: <span className="text-sky-400">{hoveredResidue.q8}</span></span>
                      </div>
                      {hoveredResidue.importance !== undefined && activePrediction.residue_importance && (
                        <div className="text-[10px] font-semibold text-slate-400 mt-1.5 flex items-center gap-1.5">
                          <span>Attribution Score:</span>
                          <span className="text-emerald-400 font-bold bg-emerald-500/10 border border-emerald-500/20 px-1.5 py-0.5 rounded font-mono">
                            {hoveredResidue.importance.toFixed(4)}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="flex gap-6 text-[10px] font-mono text-slate-400">
                    <div>
                      <span className="text-slate-500 font-semibold block">Q3 PROBS</span>
                      <span>H: {hoveredResidue.q3_prob[0].toFixed(2)} | E: {hoveredResidue.q3_prob[1].toFixed(2)} | C: {hoveredResidue.q3_prob[2].toFixed(2)}</span>
                    </div>
                    <div>
                      <span className="text-slate-500 font-semibold block">Q8 PROBS</span>
                      <span>H: {hoveredResidue.q8_prob[0].toFixed(2)} | E: {hoveredResidue.q8_prob[1].toFixed(2)} | C: {hoveredResidue.q8_prob[7].toFixed(2)} (Coil)</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="bg-slate-900/35 border border-slate-850 p-4 rounded-2xl flex items-center justify-center gap-2 text-slate-500 text-xs font-semibold">
                  <Info className="h-4 w-4" />
                  <span>Hover over any residue position above to view structural detail.</span>
                </div>
              )}

              {/* Visualization Tabs */}
              <div>
                <div className="flex gap-1.5 border-b border-slate-800 pb-3 mb-4">
                  <button
                    onClick={() => setActiveTab('overview')}
                    className={`px-4 py-1.5 text-xs font-bold rounded-xl transition-all ${
                      activeTab === 'overview'
                        ? 'bg-slate-800 text-slate-100'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    Structure Breakdown
                  </button>
                  <button
                    onClick={() => setActiveTab('confidence')}
                    className={`px-4 py-1.5 text-xs font-bold rounded-xl transition-all ${
                      activeTab === 'confidence'
                        ? 'bg-slate-800 text-slate-100'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    Confidence Curve
                  </button>
                  {activePrediction.attention_map && (
                    <button
                      onClick={() => setActiveTab('attention')}
                      className={`px-4 py-1.5 text-xs font-bold rounded-xl transition-all ${
                        activeTab === 'attention'
                          ? 'bg-slate-800 text-slate-100'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      Attention Matrix
                    </button>
                  )}
                  {activePrediction.residue_importance && (
                    <button
                      onClick={() => setActiveTab('xai')}
                      className={`px-4 py-1.5 text-xs font-bold rounded-xl transition-all ${
                        activeTab === 'xai'
                          ? 'bg-slate-800 text-slate-100'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      XAI Attributions
                    </button>
                  )}
                </div>

                {/* Tab content */}
                <div className="min-h-[260px] flex flex-col justify-center">
                  {activeTab === 'overview' && (
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      {Object.entries(STRUCTURE_COLORS).map(([key, config]) => {
                        const countsKey = key as 'H' | 'E' | 'C';
                        let cardColorClass = '';
                        let textClass = '';
                        let desc = '';
                        
                        if (key === 'H') {
                          cardColorClass = 'bg-blue-500/5 border-blue-500/10';
                          textClass = 'text-blue-400';
                          desc = 'Alpha/310/Pi helix structures';
                        } else if (key === 'E') {
                          cardColorClass = 'bg-orange-500/5 border-orange-500/10';
                          textClass = 'text-orange-400';
                          desc = 'Beta strand/bridges';
                        } else {
                          cardColorClass = 'bg-slate-700/5 border-slate-700/10';
                          textClass = 'text-slate-400';
                          desc = 'Turns, bends, random coils';
                        }
                        
                        return (
                          <div key={key} className={`${cardColorClass} border p-5 rounded-2xl text-center`}>
                            <span className={`text-[10px] font-bold ${textClass} uppercase tracking-widest block`}>
                              {config.label}
                            </span>
                            <p className={`text-4xl font-extrabold ${textClass} mt-2`}>
                              {q3Breakdown[countsKey]}%
                            </p>
                            <span className="text-[10px] text-slate-500 mt-1 block">{desc}</span>
                          </div>
                        );
                      })}
                    </div>
                  )}

                  {activeTab === 'confidence' && (
                    <div className="h-[260px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                          <XAxis dataKey="index" stroke="#64748b" fontSize={10} />
                          <YAxis domain={[0, 1]} stroke="#64748b" fontSize={10} />
                          <ChartTooltip
                            contentStyle={{ backgroundColor: '#020617', borderColor: '#334155', borderRadius: '12px' }}
                            labelStyle={{ color: '#94a3b8', fontWeight: 'bold' }}
                            itemStyle={{ color: '#f8fafc' }}
                          />
                          <Line type="monotone" dataKey="confidence" stroke="#8b5cf6" strokeWidth={2.5} dot={false} activeDot={{ r: 6 }} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  )}

                  {activeTab === 'attention' && activePrediction.attention_map && (
                    <AttentionHeatmap
                      attentionMap={activePrediction.attention_map}
                      sequence={activePrediction.sequence}
                      onHoverResidue={setHoveredResidueByIndex}
                    />
                  )}

                  {activeTab === 'xai' && activePrediction.residue_importance && (
                    <div className="h-[260px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                          <XAxis dataKey="index" stroke="#64748b" fontSize={10} />
                          <YAxis stroke="#64748b" fontSize={10} />
                          <ChartTooltip
                            contentStyle={{ backgroundColor: '#020617', borderColor: '#334155', borderRadius: '12px' }}
                            labelStyle={{ color: '#94a3b8', fontWeight: 'bold' }}
                            itemStyle={{ color: '#10b981' }}
                          />
                          <Bar dataKey="importance" fill="#10b981" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>
              </div>
            </>
          )}
            </section>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center border border-dashed border-slate-800 rounded-3xl p-12 text-center text-slate-500">
              <HelpCircle className="h-10 w-10 text-slate-700 mb-2 animate-bounce" />
              <h5 className="text-sm font-bold text-slate-400">Awaiting Prediction Request</h5>
              <p className="text-xs text-slate-600 mt-1 max-w-sm mx-auto">
                Paste a protein sequence string on the settings form panel to launch explainable sequence structural analysis.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
