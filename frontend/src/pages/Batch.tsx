import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { usePredictionStore } from '../store/usePredictionStore';
import { useModelStore } from '../store/useModelStore';
import { Database, FileText, Play, AlertCircle, ExternalLink, Download } from 'lucide-react';
import { AnimatedCounter } from '../components/AnimatedCounter';

export const Batch: React.FC = () => {
  const { runBatchPredict, batchResults, isBatchPredicting, batchError, setActivePrediction } = usePredictionStore();
  const { modelLoaded } = useModelStore();
  const navigate = useNavigate();

  const [inputData, setInputData] = useState('');
  const [returnAttention, setReturnAttention] = useState(false);
  const [returnXai, setReturnXai] = useState(false);
  const [xaiMethod, setXaiMethod] = useState<'ig' | 'shap' | 'rollout'>('ig');
  const [validationError, setValidationError] = useState<string | null>(null);

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputData(e.target.value);
    setValidationError(null);
  };

  const parseInputToSequences = (input: string): string[] => {
    const rawLines = input.split('\n').map(l => l.trim());
    const seqs: string[] = [];
    
    if (rawLines[0]?.startsWith('>')) {
      let currentSeq = '';
      for (const line of rawLines) {
        if (line.startsWith('>')) {
          if (currentSeq) {
            seqs.push(currentSeq.toUpperCase().replace(/\s/g, '').replace(/[-*]/g, ''));
            currentSeq = '';
          }
        } else {
          currentSeq += line;
        }
      }
      if (currentSeq) {
        seqs.push(currentSeq.toUpperCase().replace(/\s/g, '').replace(/[-*]/g, ''));
      }
    } else {
      for (const line of rawLines) {
        if (line) {
          seqs.push(line.toUpperCase().replace(/\s/g, '').replace(/[-*]/g, ''));
        }
      }
    }
    return seqs.filter(s => s.length >= 5 && s.length <= 2048);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setValidationError(null);

    const parsed = parseInputToSequences(inputData);
    if (parsed.length === 0) {
      setValidationError('No valid sequences found. Ensure sequences are 5-2048 residues long.');
      return;
    }
    if (parsed.length > 50) {
      setValidationError(`Maximum batch size is 50 sequences. Found ${parsed.length}.`);
      return;
    }

    try {
      await runBatchPredict({
        sequences: parsed,
        return_attention: returnAttention,
        return_xai: returnXai,
        xai_method: xaiMethod,
      });
    } catch {
      // Handled by store
    }
  };

  const handleOpenInPredictor = (item: any) => {
    setActivePrediction(item);
    navigate('/predict');
  };

  const exportResultsToJSON = () => {
    if (!batchResults) return;
    const blob = new Blob([JSON.stringify(batchResults, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `protintel_batch_results_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const getQ3CountsStr = (q3_pred: string[]) => {
    const counts = { H: 0, E: 0, C: 0 };
    (q3_pred || []).forEach(c => {
      if (c in counts) counts[c as 'H'|'E'|'C']++;
    });
    const len = q3_pred?.length || 1;
    const h = (counts.H / len * 100).toFixed(0);
    const e = (counts.E / len * 100).toFixed(0);
    const c = (counts.C / len * 100).toFixed(0);
    return `H:${h}% | E:${e}% | C:${c}%`;
  };

  return (
    <div className="flex flex-col gap-6 animate-fade-in">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Configuration Form */}
        <div className="lg:col-span-1">
          <section className="surface-tier-2 p-6 flex flex-col gap-5 relative overflow-hidden">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-[var(--text-primary)] flex items-center gap-2.5" style={{ fontFamily: 'var(--font-heading)' }}>
                <div className="p-2 rounded-xl bg-purple-500/15 border border-purple-500/30">
                  <Database className="h-4 w-4 text-violet-400" strokeWidth={1.8} />
                </div>
                <span>Batch Configuration</span>
              </h3>
              <span className="text-[10px] font-mono font-bold text-violet-400 bg-violet-400/10 px-2 py-0.5 rounded border border-violet-400/20">
                MAX 50 SEQS
              </span>
            </div>

            <form onSubmit={handleSubmit} className="flex flex-col gap-4">
              <div>
                <label className="block text-[10px] font-mono font-bold text-[var(--text-muted)] uppercase tracking-widest mb-1.5">
                  Multi-Sequence Input (FASTA or Line List)
                </label>
                <textarea
                  value={inputData}
                  onChange={handleTextareaChange}
                  placeholder={">Seq1\nMKFLILLFNILCLFPVLA\n>Seq2\nMGGKFVLLASILFP"}
                  className="instrument-input w-full h-52 p-3.5 text-xs font-mono resize-none"
                  required
                />
              </div>

              {/* Toggles */}
              <div className="surface-tier-3 p-4 flex flex-col gap-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-[var(--text-secondary)]">Compute Self-Attention</span>
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
                    <span
                      className="absolute top-[2px] transition-all duration-200"
                      style={{
                        width: 14, height: 14, borderRadius: '50%', background: '#fff',
                        left: returnAttention ? 18 : 2,
                      }}
                    />
                  </button>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-[var(--text-secondary)]">Compute XAI Attributions</span>
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
                    <span
                      className="absolute top-[2px] transition-all duration-200"
                      style={{
                        width: 14, height: 14, borderRadius: '50%', background: '#fff',
                        left: returnXai ? 18 : 2,
                      }}
                    />
                  </button>
                </div>

                {returnXai && (
                  <div className="mt-2 pt-2 border-t border-[var(--border-subtle)]">
                    <label className="block text-[10px] font-mono font-bold text-amber-400 uppercase tracking-wider mb-1.5">
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

              {(validationError || batchError) && (
                <div className="bg-rose-500/10 border border-rose-500/25 text-rose-300 p-3.5 rounded-xl text-xs flex items-start gap-2 backdrop-blur-sm">
                  <AlertCircle className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />
                  <span>{validationError || batchError}</span>
                </div>
              )}

              <button
                type="submit"
                disabled={isBatchPredicting || !modelLoaded}
                className="btn-primary-action w-full py-3.5 flex items-center justify-center gap-2.5 text-sm cursor-pointer shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isBatchPredicting ? (
                  <>
                    <div className="h-4 w-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    <span className="font-mono text-xs animate-pulse">Processing Batch...</span>
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 fill-white" />
                    <span>Run Batch Predictions</span>
                  </>
                )}
              </button>
            </form>
          </section>
        </div>

        {/* Right: Output Results Table */}
        <div className="lg:col-span-2">
          <section className="surface-tier-2 p-6 flex flex-col h-full min-h-[520px]">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
              <div>
                <h3 className="text-sm font-bold text-[var(--text-primary)] flex items-center gap-2" style={{ fontFamily: 'var(--font-heading)' }}>
                  <FileText className="h-4 w-4 text-teal-400" strokeWidth={2} />
                  <span>Batch Output Results</span>
                </h3>
                <p className="text-xs text-[var(--text-secondary)] mt-0.5">High-throughput sequence predictions and structural ratios</p>
              </div>

              {batchResults && (
                <button
                  onClick={exportResultsToJSON}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-teal-500/10 border border-teal-500/30 text-teal-400 font-mono font-bold text-xs hover:bg-teal-500/20 transition-all cursor-pointer"
                >
                  <Download className="h-3.5 w-3.5" />
                  <span>Export JSON</span>
                </button>
              )}
            </div>

            {batchResults ? (
              <div className="flex flex-col gap-4">
                {/* Stats Header Bar */}
                <div className="grid grid-cols-3 gap-3 p-4 rounded-xl surface-tier-3 font-mono text-xs border border-teal-500/20">
                  <div>
                    <span className="text-[9px] text-[var(--text-muted)] font-bold uppercase tracking-wider block">Total Batch</span>
                    <span className="text-lg font-black text-teal-400">
                      <AnimatedCounter value={batchResults.total_sequences} />
                    </span>
                  </div>
                  <div>
                    <span className="text-[9px] text-[var(--text-muted)] font-bold uppercase tracking-wider block">Successful</span>
                    <span className="text-lg font-black text-emerald-400">
                      <AnimatedCounter value={batchResults.results.length} />
                    </span>
                  </div>
                  <div>
                    <span className="text-[9px] text-[var(--text-muted)] font-bold uppercase tracking-wider block">Total Time</span>
                    <span className="text-lg font-black text-amber-400">
                      <AnimatedCounter value={batchResults.total_processing_time_ms} suffix=" ms" />
                    </span>
                  </div>
                </div>

                {/* Table */}
                <div className="overflow-x-auto rounded-xl border border-[var(--border-muted)] surface-tier-3">
                  <table className="w-full text-left text-xs font-mono">
                    <thead className="bg-[var(--glass-tier3)] text-[var(--text-muted)] font-bold uppercase border-b border-[var(--border-muted)] text-[10px] tracking-wider">
                      <tr>
                        <th className="p-3">ID</th>
                        <th className="p-3">Length</th>
                        <th className="p-3">Q3 Composition</th>
                        <th className="p-3">Avg Conf</th>
                        <th className="p-3 text-right">Action</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[var(--border-subtle)]">
                      {batchResults.results.map((res: any) => {
                        const avgConf = (res.confidence.reduce((a: number, b: number) => a + b, 0) / res.length * 100);
                        return (
                          <tr key={res.protein_id} className="hover:bg-[var(--glass-interactive-hover)] transition-colors">
                            <td className="p-3 font-bold text-[var(--text-primary)]">{res.protein_id}</td>
                            <td className="p-3 text-[var(--text-secondary)]">{res.length} aa</td>
                            <td className="p-3 text-teal-400 font-semibold">{getQ3CountsStr(res.q3_prediction)}</td>
                            <td className="p-3 text-amber-400 font-bold">{avgConf.toFixed(1)}%</td>
                            <td className="p-3 text-right">
                              <button
                                onClick={() => handleOpenInPredictor(res)}
                                className="inline-flex items-center gap-1 text-xs text-violet-400 hover:text-violet-300 font-bold cursor-pointer transition-colors"
                              >
                                <span>Inspect</span>
                                <ExternalLink className="h-3 w-3" />
                              </button>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center text-center p-8 surface-tier-3 rounded-2xl border-dashed">
                <Database className="h-10 w-10 text-[var(--text-muted)] mb-3 opacity-40 animate-pulse" strokeWidth={1.5} />
                <h4 className="text-sm font-bold text-[var(--text-secondary)]" style={{ fontFamily: 'var(--font-heading)' }}>
                  Awaiting Batch Submission
                </h4>
                <p className="text-xs text-[var(--text-muted)] font-mono mt-1 max-w-xs">
                  Enter multi-sequence FASTA in the configuration console to run batch inference.
                </p>
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
};
