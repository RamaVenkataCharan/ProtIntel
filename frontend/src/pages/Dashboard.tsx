import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useModelStore } from '../store/useModelStore';
import { usePredictionStore } from '../store/usePredictionStore';
import { Shield, Brain, BarChart, History, ChevronRight, HelpCircle, HardDrive, Cpu } from 'lucide-react';

export const Dashboard: React.FC = () => {
  const { modelInfo, metrics, modelLoaded } = useModelStore();
  const { history, setActivePrediction } = usePredictionStore();
  const navigate = useNavigate();

  const formatNumber = (num: number) => {
    if (num >= 1e6) return `${(num / 1e6).toFixed(1)}M`;
    if (num >= 1e3) return `${(num / 1e3).toFixed(1)}K`;
    return num.toString();
  };

  const handleHistoryClick = (item: any) => {
    setActivePrediction(item);
    navigate('/predict');
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Left 2 Cols: Main Info */}
      <div className="lg:col-span-2 flex flex-col gap-6">
        {/* Hero Section */}
        <section className="bg-gradient-to-br from-slate-900 via-[#0e1322] to-slate-900 border border-slate-800 rounded-3xl p-8 relative overflow-hidden shadow-2xl">
          <div className="absolute right-0 bottom-0 top-0 w-1/3 opacity-10 pointer-events-none bg-[radial-gradient(ellipse_at_bottom_right,_var(--tw-gradient-stops))] from-purple-500 via-indigo-500 to-transparent"></div>
          <div className="max-w-xl">
            <span className="bg-purple-500/10 text-purple-400 text-xs font-semibold px-3 py-1 rounded-full border border-purple-500/20 uppercase tracking-wider">
              Bioinformatics ML Platform
            </span>
            <h2 className="text-3xl font-extrabold mt-3 text-slate-100 leading-tight tracking-tight">
              Predict Protein Structure with Explanations
            </h2>
            <p className="text-sm text-slate-400 mt-2 leading-relaxed">
              ProtIntel leverages ESM-2 embeddings, convolutional feature extraction, bidirectional LSTMs, and self-attention heads to predict Q3 and Q8 structures with integrated residue-level XAI explanations.
            </p>
            <div className="flex gap-3 mt-6">
              <button
                onClick={() => navigate('/predict')}
                className="bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white font-semibold text-sm px-5 py-2.5 rounded-xl shadow-lg shadow-purple-500/20 hover:scale-[1.02] active:scale-[0.98] transition-all"
              >
                Launch Predictor
              </button>
              <button
                onClick={() => navigate('/batch')}
                className="bg-slate-800/80 hover:bg-slate-700/80 text-slate-200 border border-slate-700/60 font-semibold text-sm px-5 py-2.5 rounded-xl transition-all"
              >
                Batch Upload
              </button>
            </div>
          </div>
        </section>

        {/* Model Architecture Info */}
        <section className="bg-[#0f172a]/40 border border-slate-900 rounded-3xl p-6">
          <h3 className="text-lg font-bold text-slate-200 mb-4 flex items-center gap-2">
            <Brain className="h-5 w-5 text-indigo-400" />
            Model Specifications
          </h3>

          {modelLoaded && modelInfo ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-slate-900/40 border border-slate-800/50 p-4 rounded-2xl flex items-start gap-3">
                <Cpu className="h-5 w-5 text-purple-400 shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wide">Encoder Architecture</h4>
                  <p className="text-sm font-bold text-slate-200 mt-1">{modelInfo.architecture}</p>
                </div>
              </div>

              <div className="bg-slate-900/40 border border-slate-800/50 p-4 rounded-2xl flex items-start gap-3">
                <Brain className="h-5 w-5 text-pink-400 shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wide">ESM-2 Base Embeddings</h4>
                  <p className="text-sm font-bold text-slate-200 mt-1">{modelInfo.esm2_model}</p>
                </div>
              </div>

              <div className="bg-slate-900/40 border border-slate-800/50 p-4 rounded-2xl flex items-start gap-3">
                <HardDrive className="h-5 w-5 text-blue-400 shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wide">Total Parameters</h4>
                  <p className="text-sm font-bold text-slate-200 mt-1">{formatNumber(modelInfo.total_parameters)}</p>
                </div>
              </div>

              <div className="bg-slate-900/40 border border-slate-800/50 p-4 rounded-2xl flex items-start gap-3">
                <Shield className="h-5 w-5 text-emerald-400 shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wide">Trainable Parameters</h4>
                  <p className="text-sm font-bold text-slate-200 mt-1">{formatNumber(modelInfo.trainable_parameters)}</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-6 text-slate-500 text-sm font-medium">
              Model parameters unavailable. Please ensure checkpoint is loaded.
            </div>
          )}
        </section>

        {/* Model Accuracy Summary */}
        <section className="bg-[#0f172a]/40 border border-slate-900 rounded-3xl p-6">
          <h3 className="text-lg font-bold text-slate-200 mb-4 flex items-center gap-2">
            <BarChart className="h-5 w-5 text-emerald-400" />
            Evaluation Benchmark (CB513)
          </h3>

          {modelLoaded && metrics && metrics.q3_accuracy !== null ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-slate-900/40 border border-slate-800/50 p-5 rounded-2xl text-center">
                <h4 className="text-xs font-semibold text-slate-400 uppercase">Q3 Accuracy</h4>
                <p className="text-3xl font-extrabold text-emerald-400 mt-2">
                  {(metrics.q3_accuracy! * 100).toFixed(1)}%
                </p>
              </div>

              <div className="bg-slate-900/40 border border-slate-800/50 p-5 rounded-2xl text-center">
                <h4 className="text-xs font-semibold text-slate-400 uppercase">Q8 Accuracy</h4>
                <p className="text-3xl font-extrabold text-blue-400 mt-2">
                  {(metrics.q8_accuracy! * 100).toFixed(1)}%
                </p>
              </div>

              <div className="bg-slate-900/40 border border-slate-800/50 p-5 rounded-2xl text-center">
                <h4 className="text-xs font-semibold text-slate-400 uppercase">Q3 MCC</h4>
                <p className="text-3xl font-extrabold text-purple-400 mt-2">
                  {metrics.q3_mcc?.toFixed(3) || '0.000'}
                </p>
              </div>
            </div>
          ) : (
            <div className="text-center py-6 text-slate-500 text-sm font-medium">
              No evaluation results found. Run `python evaluate.py` to populate performance metrics.
            </div>
          )}
        </section>
      </div>

      {/* Right 1 Col: History Sidepanel */}
      <div className="flex flex-col gap-6">
        <section className="bg-[#0f172a]/30 border border-slate-900 rounded-3xl p-6 flex flex-col h-full min-h-[500px]">
          <h3 className="text-lg font-bold text-slate-200 mb-4 flex items-center gap-2">
            <History className="h-5 w-5 text-purple-400" />
            Recent History
          </h3>

          <div className="flex-1 overflow-y-auto flex flex-col gap-3 max-h-[520px] pr-1">
            {history.length > 0 ? (
              history.map((item) => (
                <button
                  key={item.protein_id}
                  onClick={() => handleHistoryClick(item)}
                  className="w-full text-left bg-[#0f172a]/80 hover:bg-slate-800/50 border border-slate-800/50 hover:border-slate-700/60 p-4 rounded-2xl transition-all duration-300 group flex items-center justify-between"
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between text-xs font-semibold text-slate-400 mb-1">
                      <span className="truncate max-w-[120px]">ID: {item.protein_id}</span>
                      <span>{item.length} residues</span>
                    </div>
                    <p className="text-sm font-bold text-slate-200 truncate font-mono">
                      {item.sequence}
                    </p>
                    <div className="flex gap-2 mt-2">
                      <span className="text-[10px] bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded-md font-semibold">
                        Q3: {(item.confidence.reduce((a, b) => a + b, 0) / item.length * 100).toFixed(0)}% Conf
                      </span>
                    </div>
                  </div>
                  <ChevronRight className="h-5 w-5 text-slate-500 group-hover:text-purple-400 transition-colors shrink-0 ml-2" />
                </button>
              ))
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center text-center p-6">
                <HelpCircle className="h-8 w-8 text-slate-600 mb-2 animate-pulse" />
                <p className="text-sm font-semibold text-slate-500">No predictions yet</p>
                <p className="text-xs text-slate-600 mt-1">Submit a sequence in the Predictor tab to start analysis.</p>
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
};
