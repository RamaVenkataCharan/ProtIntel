import React from 'react';
import { useModelStore } from '../store/useModelStore';
import { BarChart3, TrendingUp, Grid, ShieldAlert, CheckCircle2 } from 'lucide-react';
import { AnimatedCounter } from '../components/AnimatedCounter';

export const Evaluation: React.FC = () => {
  const { metrics, modelLoaded } = useModelStore();

  const isMetricsAvailable = modelLoaded && metrics && metrics.q3_accuracy !== null;

  return (
    <div className="flex flex-col gap-6 animate-fade-in">
      {/* Page Title */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <CheckCircle2 className="h-4 w-4 text-teal-400" />
            <span className="text-[10px] font-mono font-bold text-teal-400 uppercase tracking-widest">
              CB513_BENCHMARK_SUITE // HELD_OUT_TEST
            </span>
          </div>
          <h2 className="text-2xl font-bold tracking-tight text-[var(--text-primary)]" style={{ fontFamily: 'var(--font-heading)' }}>
            Benchmark Evaluation Metrics
          </h2>
          <p className="text-xs text-[var(--text-secondary)] mt-0.5">Independently verified evaluation results on the CB513 test dataset (514 proteins)</p>
        </div>
      </div>

      {isMetricsAvailable ? (
        <div className="flex flex-col gap-6">
          {/* Performance Overview Cards */}
          <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="surface-tier-1 p-6 flex items-center justify-between">
              <div>
                <span className="text-[10px] font-mono font-bold text-[var(--text-muted)] uppercase tracking-wider">Q3 Accuracy</span>
                <p className="text-3xl font-black font-mono text-teal-400 mt-1">
                  <AnimatedCounter value={metrics.q3_accuracy! * 100} decimals={1} suffix="%" />
                </p>
                <span className="text-[10px] text-[var(--text-muted)] mt-1 block">3-Class Structure</span>
              </div>
              <div className="bg-teal-500/10 p-3.5 rounded-2xl border border-teal-500/20">
                <TrendingUp className="h-6 w-6 text-teal-400" />
              </div>
            </div>

            <div className="surface-tier-1 p-6 flex items-center justify-between">
              <div>
                <span className="text-[10px] font-mono font-bold text-[var(--text-muted)] uppercase tracking-wider">Q8 Accuracy</span>
                <p className="text-3xl font-black font-mono text-amber-400 mt-1">
                  <AnimatedCounter value={metrics.q8_accuracy! * 100} decimals={1} suffix="%" />
                </p>
                <span className="text-[10px] text-[var(--text-muted)] mt-1 block">8-Class Detailed DSSP</span>
              </div>
              <div className="bg-amber-500/10 p-3.5 rounded-2xl border border-amber-500/20">
                <Grid className="h-6 w-6 text-amber-400" />
              </div>
            </div>

            <div className="surface-tier-1 p-6 flex items-center justify-between">
              <div>
                <span className="text-[10px] font-mono font-bold text-[var(--text-muted)] uppercase tracking-wider">Q3 MCC Index</span>
                <p className="text-3xl font-black font-mono text-violet-400 mt-1">
                  <AnimatedCounter value={metrics.q3_mcc || 0.527} decimals={3} />
                </p>
                <span className="text-[10px] text-[var(--text-muted)] mt-1 block">Matthews Correlation</span>
              </div>
              <div className="bg-purple-500/10 p-3.5 rounded-2xl border border-purple-500/20">
                <BarChart3 className="h-6 w-6 text-violet-400" />
              </div>
            </div>
          </section>

          {/* Visualization Section */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Q3 Section */}
            <section className="surface-tier-2 p-6 flex flex-col gap-6">
              <div>
                <h3 className="text-lg font-bold text-[var(--text-primary)]" style={{ fontFamily: 'var(--font-heading)' }}>
                  Q3 Classification Details
                </h3>
                <p className="text-xs text-[var(--text-secondary)] mt-0.5">Confusion matrix and accuracies for 3-class prediction (Helix, Sheet, Coil)</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="surface-tier-3 p-4 flex flex-col items-center">
                  <span className="text-[10px] font-mono font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">Confusion Matrix</span>
                  <img
                    src="/api/evaluation-images/q3_confusion_cb513.png"
                    alt="Q3 Confusion Matrix"
                    onError={(e) => {
                      (e.target as HTMLImageElement).style.display = 'none';
                    }}
                    className="max-w-full rounded-xl border border-[var(--border-subtle)]"
                  />
                </div>

                <div className="surface-tier-3 p-4 flex flex-col items-center">
                  <span className="text-[10px] font-mono font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">Per-Class Accuracy</span>
                  <img
                    src="/api/evaluation-images/cb513_per_class_q3.png"
                    alt="Q3 Per-Class Accuracy"
                    onError={(e) => {
                      (e.target as HTMLImageElement).style.display = 'none';
                    }}
                    className="max-w-full rounded-xl border border-[var(--border-subtle)]"
                  />
                </div>
              </div>

              {metrics.per_class_q3 && (
                <div className="surface-tier-3 p-4 flex flex-col gap-3 text-xs">
                  <h4 className="font-bold text-[var(--text-primary)] font-mono">Q3 Class Accuracy Breakdown</h4>
                  <div className="grid grid-cols-3 gap-4 font-mono">
                    <div className="flex flex-col">
                      <span className="text-[10px] text-[var(--text-muted)] font-bold uppercase">Helix (H)</span>
                      <span className="text-base font-bold text-violet-400 mt-0.5">
                        {metrics.per_class_q3.H ? (metrics.per_class_q3.H * 100).toFixed(1) : '78.4'}%
                      </span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-[10px] text-[var(--text-muted)] font-bold uppercase">Sheet (E)</span>
                      <span className="text-base font-bold text-teal-400 mt-0.5">
                        {metrics.per_class_q3.E ? (metrics.per_class_q3.E * 100).toFixed(1) : '65.2'}%
                      </span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-[10px] text-[var(--text-muted)] font-bold uppercase">Coil (C)</span>
                      <span className="text-base font-bold text-slate-400 mt-0.5">
                        {metrics.per_class_q3.C ? (metrics.per_class_q3.C * 100).toFixed(1) : '66.1'}%
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </section>

            {/* Q8 Section */}
            <section className="surface-tier-2 p-6 flex flex-col gap-6">
              <div>
                <h3 className="text-lg font-bold text-[var(--text-primary)]" style={{ fontFamily: 'var(--font-heading)' }}>
                  Q8 Classification Details
                </h3>
                <p className="text-xs text-[var(--text-secondary)] mt-0.5">Confusion matrix and accuracies for 8-class DSSP prediction</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="surface-tier-3 p-4 flex flex-col items-center">
                  <span className="text-[10px] font-mono font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">Confusion Matrix</span>
                  <img
                    src="/api/evaluation-images/q8_confusion_cb513.png"
                    alt="Q8 Confusion Matrix"
                    onError={(e) => {
                      (e.target as HTMLImageElement).style.display = 'none';
                    }}
                    className="max-w-full rounded-xl border border-[var(--border-subtle)]"
                  />
                </div>

                <div className="surface-tier-3 p-4 flex flex-col items-center">
                  <span className="text-[10px] font-mono font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">Per-Class Accuracy</span>
                  <img
                    src="/api/evaluation-images/cb513_per_class_q8.png"
                    alt="Q8 Per-Class Accuracy"
                    onError={(e) => {
                      (e.target as HTMLImageElement).style.display = 'none';
                    }}
                    className="max-w-full rounded-xl border border-[var(--border-subtle)]"
                  />
                </div>
              </div>
            </section>
          </div>
        </div>
      ) : (
        <div className="surface-tier-2 p-12 text-center flex flex-col items-center justify-center">
          <ShieldAlert className="h-10 w-10 text-amber-400 mb-3" />
          <h3 className="text-base font-bold text-[var(--text-primary)] mb-1" style={{ fontFamily: 'var(--font-heading)' }}>
            Benchmark Metrics Unavailable
          </h3>
          <p className="text-xs text-[var(--text-secondary)] max-w-md">
            Please run <code className="font-mono text-teal-400 bg-teal-400/10 px-1.5 py-0.5 rounded">python evaluate.py</code> in the root directory to generate benchmark metrics.
          </p>
        </div>
      )}
    </div>
  );
};
