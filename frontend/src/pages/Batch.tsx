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
    q3_pred.forEach(c => {
      if (c in counts) counts[c as 'H'|'E'|'C']++;
    });
    const len = q3_pred.length;
    const h = (counts.H / len * 100).toFixed(0);
    const e = (counts.E / len * 100).toFixed(0);
    const c = (counts.C / len * 100).toFixed(0);
    return `H:${h}% | E:${e}% | C:${c}%`;
  };

  return (
    <div className="flex flex-col gap-6 animate-fade-in">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Form */}
        <div className="lg:col-span-1">
          <section className="surface-tier-2 p-6">
            <h3 className="text-base font-bold text-[var(--text-primary)] mb-4 flex items-center gap-2" style={{ fontFamily: 'var(--font-heading)' }}>
              <Database className="h-5 w-5 text-violet-400" />
              <span>Batch Configuration</span>
            </h3>

            <form onSubmit={handleSubmit} className="flex flex-col gap-4">
              <div>
                <label className="block text-[10px] font-mono font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">
                  Multi-Sequence Input (FASTA or Line List)
                </label>
                <textarea
                  value={inputData}
                  onChange={handleTextareaChange}
                  placeholder={">Seq1\nMKFLILLFNILCLFPVLA\n>Seq2\nMGGKFVLLASILFP"}
                  className="w-full h-56 bg-[var(--bg-card-tier3)] border border-[var(--border-subtle)] rounded-2xl p-4 text-xs font-mono text-[var(--text-primary)] placeholder-slate-600 focus:outline-none focus:border-teal-400 resize-none transition-all"
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
                    className="relative flex-shrink-0"
                    style={{
                      width: 36, height: 20, borderRadius: 10, cursor: 'pointer',
                      background: returnAttention ? 'linear-gradient(135deg, #7B2FF7, #00D9C0)' : 'rgba(255,255,255,0.08)',
                      border: `1px solid ${returnAttention ? 'rgba(0,217,192,0.5)' : 'var(--border-muted)'}`,
                    }}
                  >
                    <span
                      className="absolute top-[2px]"
                      style={{
                        width: 14, height: 14, borderRadius: '50%', background: '#fff',
                        left: returnAttention ? 18 : 2, transition: 'left 0.2s ease',
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
                    className="relative flex-shrink-0"
                    style={{
                      width: 36, height: 20, borderRadius: 10, cursor: 'pointer',
                      background: returnXai ? 'linear-gradient(135deg, #7B2FF7, #00D9C0)' : 'rgba(255,255,255,0.08)',
                      border: `1px solid ${returnXai ? 'rgba(0,217,192,0.5)' : 'var(--border-muted)'}`,
                    }}
                  >
                    <span
                      className="absolute top-[2px]"
                      style={{
                        width: 14, height: 14, borderRadius: '50%', background: '#fff',
                        left: returnXai ? 18 : 2, transition: 'left 0.2s ease',
                      }}
                    />
                  </button>
                </div>

                {returnXai && (
                  <div className="mt-2 pt-2 border-t border-[var(--border-subtle)]">
                    <label className="block text-[10px] font-mono font-bold text-amber-400 uppercase tracking-wider mb-1">
                      XAI Method
                    </label>
                    <select
                      value={xaiMethod}
                      onChange={(e: any) => setXaiMethod(e.target.value)}
                      className="w-full bg-[var(--bg-card-tier3)] border border-[var(--border-subtle)] rounded-xl px-3 py-1.5 text-xs font-mono text-[var(--text-primary)] focus:outline-none"
                    >
                      <option value="ig">Integrated Gradients (IG)</option>
                      <option value="shap">Gradient SHAP</option>
                      <option value="rollout">Attention Rollout</option>
                    </select>
                  </div>
                )}
              </div>

              {(validationError || batchError) && (
                <div className="bg-rose-500/10 border border-rose-500/20 text-rose-300 p-3.5 rounded-2xl text-xs flex items-start gap-2">
                  <AlertCircle className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />
                  <span>{validationError || batchError}</span>
                </div>
              )}

              <button
                type="submit"
                disabled={isBatchPredicting || !modelLoaded}
                className="w-full font-bold text-sm py-3 rounded-2xl text-white flex items-center justify-center gap-2 transition-all cursor-pointer"
                style={{
                  fontFamily: 'var(--font-heading)',
                  background: isBatchPredicting || !modelLoaded ? 'var(--bg-card-tier3)' : 'linear-gradient(135deg, var(--aurora-violet) 0%, var(--aurora-teal) 100%)',
                  boxShadow: isBatchPredicting || !modelLoaded ? 'none' : '0 4px 20px rgba(123,47,247,0.3)',
                  color: isBatchPredicting || !modelLoaded ? 'var(--text-muted)' : '#fff',
                }}
              >
                {isBatchPredicting ? (
                  <>
                    <div className="h-4 w-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    <span className="font-mono text-xs">Processing Batch...</span>
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

        {/* Right Output Table */}
        <div className="lg:col-span-2">
          <section className="surface-tier-2 p-6 flex flex-col h-full min-h-[500px]">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-base font-bold text-[var(--text-primary)] flex items-center gap-2" style={{ fontFamily: 'var(--font-heading)' }}>
                  <FileText className="h-5 w-5 text-teal-400" />
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
                <div className="flex gap-4 p-4 rounded-2xl surface-tier-3 font-mono text-xs">
                  <div>
                    <span className="text-[10px] text-[var(--text-muted)] uppercase block">Total Batch</span>
                    <span className="text-base font-bold text-teal-400">
                      <AnimatedCounter value={batchResults.total_sequences} />
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-[var(--text-muted)] uppercase block">Successful</span>
                    <span className="text-base font-bold text-emerald-400">
                      <AnimatedCounter value={batchResults.results.length} />
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-[var(--text-muted)] uppercase block">Total Time</span>
                    <span className="text-base font-bold text-amber-400">
                      <AnimatedCounter value={batchResults.total_processing_time_ms} suffix=" ms" />
                    </span>
                  </div>
                </div>

                <div className="overflow-x-auto rounded-2xl border border-[var(--border-subtle)]">
                  <table className="w-full text-left text-xs font-mono">
                    <thead className="bg-[var(--bg-card-tier3)] text-[var(--text-muted)] font-bold uppercase border-b border-[var(--border-subtle)]">
                      <tr>
                        <th className="p-3">ID</th>
                        <th className="p-3">Length</th>
                        <th className="p-3">Q3 Ratio</th>
                        <th className="p-3">Avg Conf</th>
                        <th className="p-3 text-right">Action</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[var(--border-subtle)]">
                      {batchResults.results.map((res: any) => {
                        const avgConf = (res.confidence.reduce((a: number, b: number) => a + b, 0) / res.length * 100);
                        return (
                          <tr key={res.protein_id} className="hover:bg-[var(--border-subtle)]/50 transition-colors">
                            <td className="p-3 font-bold text-[var(--text-primary)]">{res.protein_id}</td>
                            <td className="p-3 text-[var(--text-secondary)]">{res.length} aa</td>
                            <td className="p-3 text-teal-400">{getQ3CountsStr(res.q3_prediction)}</td>
                            <td className="p-3 text-amber-400">{avgConf.toFixed(1)}%</td>
                            <td className="p-3 text-right">
                              <button
                                onClick={() => handleOpenInPredictor(res)}
                                className="inline-flex items-center gap-1 text-xs text-violet-400 hover:text-violet-300 font-bold cursor-pointer"
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
              <div className="flex-1 flex flex-col items-center justify-center text-center p-8">
                <Database className="h-10 w-10 text-[var(--text-muted)] mb-3 opacity-50" />
                <h4 className="text-sm font-bold text-[var(--text-secondary)]" style={{ fontFamily: 'var(--font-heading)' }}>
                  Awaiting Batch Submission
                </h4>
                <p className="text-xs text-[var(--text-muted)] mt-1 max-w-xs">
                  Enter multi-sequence input in the configuration panel to execute batch inference.
                </p>
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
};
