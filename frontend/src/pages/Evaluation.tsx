import React from 'react';
import { useModelStore } from '../store/useModelStore';
import { BarChart3, TrendingUp, Grid, ShieldAlert } from 'lucide-react';

export const Evaluation: React.FC = () => {
  const { metrics, modelLoaded } = useModelStore();

  const isMetricsAvailable = modelLoaded && metrics && metrics.q3_accuracy !== null;

  return (
    <div className="flex flex-col gap-6">
      {/* Page Title */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-slate-100">Benchmark Performance</h2>
          <p className="text-xs text-slate-400 font-medium">Evaluation metrics on the CB513 test set</p>
        </div>
      </div>

      {isMetricsAvailable ? (
        <div className="flex flex-col gap-6">
          {/* Performance Overview Cards */}
          <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-[#0f172a]/30 border border-slate-900 p-6 rounded-3xl flex items-center justify-between">
              <div>
                <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider">Q3 Accuracy</span>
                <p className="text-3xl font-extrabold text-emerald-400 mt-1">
                  {(metrics.q3_accuracy! * 100).toFixed(1)}%
                </p>
              </div>
              <div className="bg-emerald-500/10 p-3 rounded-2xl">
                <TrendingUp className="h-6 w-6 text-emerald-400" />
              </div>
            </div>

            <div className="bg-[#0f172a]/30 border border-slate-900 p-6 rounded-3xl flex items-center justify-between">
              <div>
                <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider">Q8 Accuracy</span>
                <p className="text-3xl font-extrabold text-blue-400 mt-1">
                  {(metrics.q8_accuracy! * 100).toFixed(1)}%
                </p>
              </div>
              <div className="bg-blue-500/10 p-3 rounded-2xl">
                <Grid className="h-6 w-6 text-blue-400" />
              </div>
            </div>

            <div className="bg-[#0f172a]/30 border border-slate-900 p-6 rounded-3xl flex items-center justify-between">
              <div>
                <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider">Q3 MCC</span>
                <p className="text-3xl font-extrabold text-purple-400 mt-1">
                  {metrics.q3_mcc?.toFixed(3) || '0.000'}
                </p>
              </div>
              <div className="bg-purple-500/10 p-3 rounded-2xl">
                <BarChart3 className="h-6 w-6 text-purple-400" />
              </div>
            </div>
          </section>

          {/* Visualization Section */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Q3 Section */}
            <section className="bg-[#0f172a]/30 border border-slate-900 rounded-3xl p-6 flex flex-col gap-6">
              <div>
                <h3 className="text-lg font-bold text-slate-200">Q3 Classification Details</h3>
                <p className="text-xs text-slate-400 mt-0.5">Confusion matrix and accuracies for 3-class prediction (Helix, Sheet, Coil)</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-slate-950/40 border border-slate-850 p-4 rounded-2xl flex flex-col items-center">
                  <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider mb-2">Confusion Matrix</span>
                  <img
                    src="/api/evaluation-images/q3_confusion_cb513.png"
                    alt="Q3 Confusion Matrix"
                    onError={(e) => {
                      (e.target as HTMLImageElement).style.display = 'none';
                    }}
                    className="max-w-full rounded-lg border border-slate-800"
                  />
                </div>

                <div className="bg-slate-950/40 border border-slate-850 p-4 rounded-2xl flex flex-col items-center">
                  <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider mb-2">Per-Class Accuracy</span>
                  <img
                    src="/api/evaluation-images/cb513_per_class_q3.png"
                    alt="Q3 Per-Class Accuracy"
                    onError={(e) => {
                      (e.target as HTMLImageElement).style.display = 'none';
                    }}
                    className="max-w-full rounded-lg border border-slate-800"
                  />
                </div>
              </div>

              {metrics.per_class_q3 && (
                <div className="bg-slate-900/40 border border-slate-850/50 p-4 rounded-2xl flex flex-col gap-3 text-xs">
                  <h4 className="font-semibold text-slate-300">Q3 Class Accuracy Breakdown</h4>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="flex flex-col">
                      <span className="text-slate-500 font-bold">Helix (H)</span>
                      <span className="text-sm font-bold mt-0.5 text-amber-400">
                        {((metrics.per_class_q3['H'] || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-slate-500 font-bold">Sheet (E)</span>
                      <span className="text-sm font-bold mt-0.5 text-blue-400">
                        {((metrics.per_class_q3['E'] || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-slate-500 font-bold">Coil (C)</span>
                      <span className="text-sm font-bold mt-0.5 text-slate-300">
                        {((metrics.per_class_q3['C'] || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </section>

            {/* Q8 Section */}
            <section className="bg-[#0f172a]/30 border border-slate-900 rounded-3xl p-6 flex flex-col gap-6">
              <div>
                <h3 className="text-lg font-bold text-slate-200">Q8 Classification Details</h3>
                <p className="text-xs text-slate-400 mt-0.5">Confusion matrix and accuracies for 8-class prediction</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-slate-950/40 border border-slate-850 p-4 rounded-2xl flex flex-col items-center">
                  <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider mb-2">Confusion Matrix</span>
                  <img
                    src="/api/evaluation-images/q8_confusion_cb513.png"
                    alt="Q8 Confusion Matrix"
                    onError={(e) => {
                      (e.target as HTMLImageElement).style.display = 'none';
                    }}
                    className="max-w-full rounded-lg border border-slate-800"
                  />
                </div>

                <div className="bg-slate-950/40 border border-slate-850 p-4 rounded-2xl flex flex-col items-center">
                  <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider mb-2">Per-Class Accuracy</span>
                  <img
                    src="/api/evaluation-images/cb513_per_class_q8.png"
                    alt="Q8 Per-Class Accuracy"
                    onError={(e) => {
                      (e.target as HTMLImageElement).style.display = 'none';
                    }}
                    className="max-w-full rounded-lg border border-slate-800"
                  />
                </div>
              </div>
            </section>
          </div>
        </div>
      ) : (
        <section className="bg-slate-900/10 border border-dashed border-slate-800 rounded-3xl p-12 text-center text-slate-500 max-w-xl mx-auto mt-8 flex flex-col items-center gap-3">
          <ShieldAlert className="h-10 w-10 text-rose-500/80 animate-pulse" />
          <h4 className="text-sm font-bold text-slate-400">Benchmark Data Not Populated</h4>
          <p className="text-xs text-slate-600 leading-relaxed">
            Historical evaluation results are missing. Execute the benchmark evaluation suite in the terminal to view complete metrics:
          </p>
          <code className="bg-slate-950/80 border border-slate-800 rounded-xl px-4 py-2 font-mono text-purple-400 text-xs mt-2 select-all">
            python evaluate.py --checkpoint models/best_checkpoint.pt
          </code>
        </section>
      )}
    </div>
  );
};
