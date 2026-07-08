import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { usePredictionStore } from '../store/usePredictionStore';
import { useModelStore } from '../store/useModelStore';
import { Database, FileText, Play, AlertCircle, ExternalLink, Download } from 'lucide-react';

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
    
    // Check if FASTA format
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
      // Just plain list, one sequence per line
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
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Form */}
        <div className="lg:col-span-1">
          <section className="bg-[#0f172a]/30 border border-slate-900 rounded-3xl p-6">
            <h3 className="text-lg font-bold text-slate-200 mb-4 flex items-center gap-2">
              <Database className="h-5 w-5 text-purple-400" />
              Batch Configuration
            </h3>

            <form onSubmit={handleSubmit} className="flex flex-col gap-4">
              <div>
                <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                  Multi-Sequence Input (FASTA or List)
                </label>
                <textarea
                  value={inputData}
                  onChange={handleTextareaChange}
                  placeholder="Paste multiple sequences (one per line, or standard multi-record FASTA)&#10;&#10;Example:&#10;>Seq1&#10;MKFLILLFNILCLFPVLA&#10;>Seq2&#10;MGGKFVLLASILFP"
                  className="w-full h-56 bg-slate-950/60 border border-slate-800 rounded-2xl p-4 text-sm font-mono text-slate-100 placeholder-slate-600 focus:outline-none focus:border-purple-500/50 resize-none transition-all"
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

              {(validationError || batchError) && (
                <div className="bg-rose-500/10 border border-rose-500/20 text-rose-300 p-3.5 rounded-2xl text-xs flex items-start gap-2">
                  <AlertCircle className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />
                  <span>{validationError || batchError}</span>
                </div>
              )}

              <button
                type="submit"
                disabled={isBatchPredicting || !modelLoaded}
                className="w-full bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 disabled:from-slate-800 disabled:to-slate-800 text-white font-semibold text-sm py-3 rounded-2xl shadow-lg shadow-purple-500/10 disabled:shadow-none hover:scale-[1.01] active:scale-[0.99] disabled:pointer-events-none transition-all flex items-center justify-center gap-2"
              >
                {isBatchPredicting ? (
                  <>
                    <div className="h-4 w-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    <span>Processing Batch...</span>
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

        {/* Right Panel: Batch Results */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          {batchResults ? (
            <section className="bg-[#0f172a]/30 border border-slate-900 rounded-3xl p-6 flex flex-col gap-6">
              <div className="flex items-center justify-between gap-4 border-b border-slate-800/80 pb-4">
                <div>
                  <span className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider">Batch Summary</span>
                  <h4 className="text-sm font-bold text-slate-300 mt-0.5">
                    Processed {batchResults.total_sequences} sequences
                  </h4>
                </div>

                <div className="flex gap-4 items-center">
                  <div className="text-right">
                    <span className="text-[10px] text-slate-400 font-semibold uppercase">Total Latency</span>
                    <p className="text-sm font-bold text-slate-200 mt-0.5">{batchResults.total_processing_time_ms.toFixed(0)} ms</p>
                  </div>
                  <div className="text-right border-l border-slate-800 pl-4">
                    <span className="text-[10px] text-slate-400 font-semibold uppercase">Throughput</span>
                    <p className="text-sm font-bold text-slate-200 mt-0.5">
                      {(batchResults.total_sequences / (batchResults.total_processing_time_ms / 1000)).toFixed(1)} seq/sec
                    </p>
                  </div>

                  <button
                    onClick={exportResultsToJSON}
                    className="ml-2 bg-slate-850 hover:bg-slate-800 text-slate-300 border border-slate-800 hover:border-slate-700 p-2 rounded-xl transition-all"
                    title="Export Results JSON"
                  >
                    <Download className="h-4 w-4" />
                  </button>
                </div>
              </div>

              {/* Table */}
              <div className="overflow-x-auto rounded-2xl border border-slate-850 bg-slate-950/20">
                <table className="w-full border-collapse text-left text-xs text-slate-300">
                  <thead className="bg-[#0f172a]/60 border-b border-slate-850 font-bold uppercase tracking-wider text-slate-400">
                    <tr>
                      <th className="p-4">#</th>
                      <th className="p-4">ID</th>
                      <th className="p-4">Length</th>
                      <th className="p-4">Q3 Composition</th>
                      <th className="p-4">Avg Conf</th>
                      <th className="p-4 text-center">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-850">
                    {batchResults.results.map((res, i) => (
                      <tr key={res.protein_id} className="hover:bg-slate-900/30">
                        <td className="p-4 font-semibold text-slate-500">{i + 1}</td>
                        <td className="p-4 font-mono font-bold truncate max-w-[120px]">{res.protein_id}</td>
                        <td className="p-4 font-medium">{res.length} AAs</td>
                        <td className="p-4 font-medium text-slate-400">{getQ3CountsStr(res.q3_prediction)}</td>
                        <td className="p-4 font-semibold text-emerald-400">
                          {(res.confidence.reduce((a, b) => a + b, 0) / res.length * 100).toFixed(0)}%
                        </td>
                        <td className="p-4 text-center">
                          <button
                            onClick={() => handleOpenInPredictor(res)}
                            className="bg-purple-600/10 hover:bg-purple-600/20 text-purple-400 border border-purple-500/20 px-2.5 py-1.5 rounded-lg font-semibold inline-flex items-center gap-1 hover:scale-[1.03] transition-all"
                          >
                            <ExternalLink className="h-3 w-3" />
                            <span>Visualize</span>
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center border border-dashed border-slate-800 rounded-3xl p-12 text-center text-slate-500">
              <FileText className="h-10 w-10 text-slate-700 mb-2 animate-pulse" />
              <h5 className="text-sm font-bold text-slate-400">No Batch Processed</h5>
              <p className="text-xs text-slate-600 mt-1 max-w-sm mx-auto">
                Configure sequence input in the left panel and click run to analyze multiple proteins in parallel.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
