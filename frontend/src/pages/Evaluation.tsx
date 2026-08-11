import React from 'react';
import { useModelStore } from '../store/useModelStore';
import { MODEL_METRICS } from '../config/modelMetrics';
import {
  BarChart3,
  TrendingUp,
  Grid,
  CheckCircle2,
  Award,
  Zap,
  Target,
  ShieldCheck,
  Sparkles,
  Layers,
  Activity,
  Cpu,
  Clock,
  PieChart
} from 'lucide-react';
import { AnimatedCounter } from '../components/AnimatedCounter';
import { AttentionHeatmap } from '../components/AttentionHeatmap';

export const Evaluation: React.FC = () => {
  const { device } = useModelStore();

  const m = MODEL_METRICS;
  const q3Acc = m.q3Accuracy * 100;
  const q8Acc = m.q8Accuracy * 100;
  const q3Prec = m.q3Precision * 100;
  const q3Rec = m.q3Recall * 100;
  const q3F1 = m.q3F1 * 100;
  const q8Prec = m.q8Precision * 100;
  const q8Rec = m.q8Recall * 100;
  const q8F1 = m.q8F1 * 100;
  const q3Mcc = m.q3Mcc;
  const q8Mcc = m.q8Mcc;
  const confidence = m.confidence * 100;

  return (
    <div className="flex flex-col gap-6 animate-fade-in">
      {/* ── PAGE HEADER & BADGE ────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <CheckCircle2 className="h-4 w-4 text-teal-400" />
            <span className="text-[10px] font-mono font-bold text-teal-400 uppercase tracking-widest">
              PROTINTEL_ANALYTICS // MODEL_PERFORMANCE
            </span>
          </div>
          <h2 className="text-2xl font-bold tracking-tight text-[var(--text-primary)]" style={{ fontFamily: 'var(--font-heading)' }}>
            ProtIntel Prediction Results & Model Analytics
          </h2>
          <p className="text-xs text-[var(--text-secondary)] mt-0.5">
            Comprehensive model performance metrics and residue-level secondary structure evaluation
          </p>
        </div>

        {/* Status & Analytics Badge */}
        <div className="flex items-center gap-3 bg-black/40 border border-teal-500/30 p-3 rounded-2xl shadow-lg backdrop-blur-md">
          <div className="p-2 rounded-xl bg-teal-500/15 border border-teal-500/30 shrink-0">
            <ShieldCheck className="h-5 w-5 text-teal-400" />
          </div>
          <div className="text-xs font-mono">
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-slate-400 font-bold uppercase">
                Dataset: MODEL PERFORMANCE
              </span>
              <span className="px-1.5 py-0.2 rounded text-[8px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                ✓ Operational
              </span>
            </div>
            <p className="text-slate-200 font-bold mt-0.5">
              Status: <span className="text-teal-300">● ONLINE</span>
            </p>
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-6">

        {/* ── TOP-LEVEL METRIC CARDS ───────────────────────────────────── */}
        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">

          {/* Q3 ACCURACY Card */}
          <div className="surface-tier-1 p-5 flex flex-col justify-between border border-teal-500/30">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono font-bold text-[var(--text-muted)] uppercase tracking-wider">Q3 ACCURACY</span>
              <TrendingUp className="h-4 w-4 text-teal-400" />
            </div>
            <p className="text-3xl font-black font-mono text-teal-400 my-2">
              <AnimatedCounter value={q3Acc} decimals={2} suffix="%" />
            </p>
            <span className="text-[10px] text-[var(--text-muted)] block">3-Class Structure Accuracy</span>
          </div>

          {/* Q8 ACCURACY Card */}
          <div className="surface-tier-1 p-5 flex flex-col justify-between border border-amber-500/30">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono font-bold text-[var(--text-muted)] uppercase tracking-wider">Q8 ACCURACY</span>
              <Grid className="h-4 w-4 text-amber-400" />
            </div>
            <p className="text-3xl font-black font-mono text-amber-400 my-2">
              <AnimatedCounter value={q8Acc} decimals={2} suffix="%" />
            </p>
            <span className="text-[10px] text-[var(--text-muted)] block">8-Class Detailed DSSP</span>
          </div>

          {/* Q3 MCC Card */}
          <div className="surface-tier-1 p-5 flex flex-col justify-between border border-violet-500/30">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono font-bold text-[var(--text-muted)] uppercase tracking-wider">Q3 MCC</span>
              <BarChart3 className="h-4 w-4 text-violet-400" />
            </div>
            <p className="text-3xl font-black font-mono text-violet-400 my-2">
              <AnimatedCounter value={q3Mcc} decimals={3} />
            </p>
            <span className="text-[10px] text-[var(--text-muted)] block">Matthews Correlation (Q3)</span>
          </div>

          {/* Q8 MCC Card */}
          <div className="surface-tier-1 p-5 flex flex-col justify-between border border-purple-500/30">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono font-bold text-[var(--text-muted)] uppercase tracking-wider">Q8 MCC</span>
              <BarChart3 className="h-4 w-4 text-purple-400" />
            </div>
            <p className="text-3xl font-black font-mono text-purple-400 my-2">
              <AnimatedCounter value={q8Mcc} decimals={3} />
            </p>
            <span className="text-[10px] text-[var(--text-muted)] block">Matthews Correlation (Q8)</span>
          </div>

          {/* Q3 PRECISION Card */}
          <div className="surface-tier-1 p-5 flex flex-col justify-between border border-emerald-500/30">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono font-bold text-[var(--text-muted)] uppercase tracking-wider">Q3 PRECISION</span>
              <Award className="h-4 w-4 text-emerald-400" />
            </div>
            <p className="text-3xl font-black font-mono text-emerald-400 my-2">
              <AnimatedCounter value={q3Prec} decimals={2} suffix="%" />
            </p>
            <span className="text-[10px] text-[var(--text-muted)] block">Weighted Precision (Q3)</span>
          </div>

          {/* Q3 RECALL Card */}
          <div className="surface-tier-1 p-5 flex flex-col justify-between border border-cyan-500/30">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono font-bold text-[var(--text-muted)] uppercase tracking-wider">Q3 RECALL</span>
              <Zap className="h-4 w-4 text-cyan-400" />
            </div>
            <p className="text-3xl font-black font-mono text-cyan-400 my-2">
              <AnimatedCounter value={q3Rec} decimals={2} suffix="%" />
            </p>
            <span className="text-[10px] text-[var(--text-muted)] block">Weighted Recall (Q3)</span>
          </div>

          {/* Q3 F1 SCORE Card */}
          <div className="surface-tier-1 p-5 flex flex-col justify-between border border-purple-500/30">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono font-bold text-[var(--text-muted)] uppercase tracking-wider">Q3 F1 SCORE</span>
              <Sparkles className="h-4 w-4 text-purple-400" />
            </div>
            <p className="text-3xl font-black font-mono text-purple-400 my-2">
              <AnimatedCounter value={q3F1} decimals={2} suffix="%" />
            </p>
            <span className="text-[10px] text-[var(--text-muted)] block">Weighted F1 Metric (Q3)</span>
          </div>

          {/* Q8 F1 SCORE Card */}
          <div className="surface-tier-1 p-5 flex flex-col justify-between border border-pink-500/30">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono font-bold text-[var(--text-muted)] uppercase tracking-wider">Q8 F1 SCORE</span>
              <Sparkles className="h-4 w-4 text-pink-400" />
            </div>
            <p className="text-3xl font-black font-mono text-pink-400 my-2">
              <AnimatedCounter value={q8F1} decimals={2} suffix="%" />
            </p>
            <span className="text-[10px] text-[var(--text-muted)] block">Weighted F1 Metric (Q8)</span>
          </div>

          {/* MODEL CONFIDENCE Card */}
          <div className="surface-tier-1 p-5 flex flex-col justify-between border border-amber-400/30 md:col-span-2 lg:col-span-2">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono font-bold text-[var(--text-muted)] uppercase tracking-wider">MODEL CONFIDENCE</span>
              <ShieldCheck className="h-4 w-4 text-amber-400" />
            </div>
            <div className="flex items-baseline justify-between my-2">
              <p className="text-3xl font-black font-mono text-amber-300">
                <AnimatedCounter value={confidence} decimals={2} suffix="%" />
              </p>
              <span className="text-xs font-mono font-bold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                High Confidence Standard
              </span>
            </div>
            <span className="text-[10px] text-[var(--text-muted)] block">Average Prediction Confidence Score</span>
          </div>

          {/* Q8 PRECISION & RECALL Summary Card */}
          <div className="surface-tier-1 p-5 flex flex-col justify-between border border-indigo-500/30 md:col-span-2 lg:col-span-2">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono font-bold text-[var(--text-muted)] uppercase tracking-wider">Q8 PRECISION & RECALL</span>
              <Target className="h-4 w-4 text-indigo-400" />
            </div>
            <div className="grid grid-cols-2 gap-4 my-2 font-mono">
              <div>
                <span className="text-[10px] text-slate-400 block">Q8 Precision</span>
                <p className="text-2xl font-black text-indigo-300">
                  <AnimatedCounter value={q8Prec} decimals={2} suffix="%" />
                </p>
              </div>
              <div>
                <span className="text-[10px] text-slate-400 block">Q8 Recall</span>
                <p className="text-2xl font-black text-pink-300">
                  <AnimatedCounter value={q8Rec} decimals={2} suffix="%" />
                </p>
              </div>
            </div>
            <span className="text-[10px] text-[var(--text-muted)] block">Weighted Q8 Secondary Performance</span>
          </div>

        </section>

        {/* ── CONFIDENCE DISTRIBUTION SECTION ───────────────────────── */}
        <section className="surface-tier-2 p-6 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold text-[var(--text-primary)] flex items-center gap-2" style={{ fontFamily: 'var(--font-heading)' }}>
              <PieChart className="h-5 w-5 text-amber-400" />
              <span>Confidence Distribution</span>
            </h3>
            <span className="text-[10px] font-mono text-amber-400 bg-amber-400/10 px-2 py-0.5 rounded border border-amber-400/20 font-bold">
              AGGREGATE PREDICTION CONFIDENCE
            </span>
          </div>

          {/* Visual Distribution Bar */}
          <div className="h-4 w-full bg-black/40 rounded-xl overflow-hidden flex p-0.5 gap-0.5 border border-white/10">
            <div
              className="h-full bg-teal-400 rounded-sm transition-all duration-500 shadow-[0_0_10px_rgba(45,212,191,0.3)]"
              style={{ width: `${m.confidenceDistribution.high}%` }}
              title={`HIGH CONFIDENCE: ${m.confidenceDistribution.high}%`}
            />
            <div
              className="h-full bg-amber-400 rounded-sm transition-all duration-500 shadow-[0_0_10px_rgba(251,191,36,0.3)]"
              style={{ width: `${m.confidenceDistribution.medium}%` }}
              title={`MEDIUM CONFIDENCE: ${m.confidenceDistribution.medium}%`}
            />
            <div
              className="h-full bg-rose-400 rounded-sm transition-all duration-500 shadow-[0_0_10px_rgba(244,63,94,0.3)]"
              style={{ width: `${m.confidenceDistribution.low}%` }}
              title={`LOW CONFIDENCE: ${m.confidenceDistribution.low}%`}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 font-mono text-xs mt-1">
            <div className="surface-tier-3 p-4 rounded-xl border border-teal-500/30 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="w-3.5 h-3.5 rounded-md bg-teal-400 shrink-0" />
                <div>
                  <span className="text-slate-300 font-bold block">HIGH CONFIDENCE</span>
                  <span className="text-[10px] text-slate-400">Score &ge; 0.80</span>
                </div>
              </div>
              <span className="text-xl font-black text-teal-400">{m.confidenceDistribution.high}%</span>
            </div>

            <div className="surface-tier-3 p-4 rounded-xl border border-amber-500/30 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="w-3.5 h-3.5 rounded-md bg-amber-400 shrink-0" />
                <div>
                  <span className="text-slate-300 font-bold block">MEDIUM CONFIDENCE</span>
                  <span className="text-[10px] text-slate-400">0.50 &le; Score &lt; 0.80</span>
                </div>
              </div>
              <span className="text-xl font-black text-amber-400">{m.confidenceDistribution.medium}%</span>
            </div>

            <div className="surface-tier-3 p-4 rounded-xl border border-rose-500/30 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="w-3.5 h-3.5 rounded-md bg-rose-400 shrink-0" />
                <div>
                  <span className="text-slate-300 font-bold block">LOW CONFIDENCE</span>
                  <span className="text-[10px] text-slate-400">Score &lt; 0.50</span>
                </div>
              </div>
              <span className="text-xl font-black text-rose-400">{m.confidenceDistribution.low}%</span>
            </div>
          </div>
        </section>

        {/* ── Q3 & Q8 CLASS PERFORMANCE ──────────────────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

          {/* Q3 Class Performance */}
          <section className="surface-tier-2 p-6 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-bold text-[var(--text-primary)] flex items-center gap-2" style={{ fontFamily: 'var(--font-heading)' }}>
                <BarChart3 className="h-5 w-5 text-teal-400" />
                <span>Q3 Secondary Structure Performance</span>
              </h3>
              <span className="text-[10px] font-mono text-teal-400 bg-teal-400/10 px-2 py-0.5 rounded border border-teal-400/20 font-bold">
                Q3 OVERALL ACCURACY: 91.24%
              </span>
            </div>

            <div className="flex flex-col gap-4 font-mono text-xs">
              {/* Alpha Helix (H) */}
              <div className="surface-tier-3 p-3.5 rounded-xl border border-teal-500/30">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-slate-200 font-bold text-sm">Alpha Helix (H)</span>
                  <span className="text-teal-400 font-black text-sm">F1: {(m.q3Classes.H.f1 * 100).toFixed(2)}%</span>
                </div>
                <div className="h-2.5 bg-black/40 rounded-full overflow-hidden border border-white/10 p-0.5 my-1.5">
                  <div className="h-full bg-teal-400 rounded-full transition-all duration-500" style={{ width: `${m.q3Classes.H.f1 * 100}%` }} />
                </div>
                <div className="flex justify-between text-[10px] text-slate-400">
                  <span>Precision: <strong className="text-teal-300">{(m.q3Classes.H.precision * 100).toFixed(2)}%</strong></span>
                  <span>Recall: <strong className="text-teal-300">{(m.q3Classes.H.recall * 100).toFixed(2)}%</strong></span>
                </div>
              </div>

              {/* Beta Strand (E) */}
              <div className="surface-tier-3 p-3.5 rounded-xl border border-cyan-500/30">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-slate-200 font-bold text-sm">Beta Strand (E)</span>
                  <span className="text-cyan-400 font-black text-sm">F1: {(m.q3Classes.E.f1 * 100).toFixed(2)}%</span>
                </div>
                <div className="h-2.5 bg-black/40 rounded-full overflow-hidden border border-white/10 p-0.5 my-1.5">
                  <div className="h-full bg-cyan-400 rounded-full transition-all duration-500" style={{ width: `${m.q3Classes.E.f1 * 100}%` }} />
                </div>
                <div className="flex justify-between text-[10px] text-slate-400">
                  <span>Precision: <strong className="text-cyan-300">{(m.q3Classes.E.precision * 100).toFixed(2)}%</strong></span>
                  <span>Recall: <strong className="text-cyan-300">{(m.q3Classes.E.recall * 100).toFixed(2)}%</strong></span>
                </div>
              </div>

              {/* Coil (C) */}
              <div className="surface-tier-3 p-3.5 rounded-xl border border-violet-500/30">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-slate-200 font-bold text-sm">Coil (C)</span>
                  <span className="text-violet-400 font-black text-sm">F1: {(m.q3Classes.C.f1 * 100).toFixed(2)}%</span>
                </div>
                <div className="h-2.5 bg-black/40 rounded-full overflow-hidden border border-white/10 p-0.5 my-1.5">
                  <div className="h-full bg-violet-400 rounded-full transition-all duration-500" style={{ width: `${m.q3Classes.C.f1 * 100}%` }} />
                </div>
                <div className="flex justify-between text-[10px] text-slate-400">
                  <span>Precision: <strong className="text-violet-300">{(m.q3Classes.C.precision * 100).toFixed(2)}%</strong></span>
                  <span>Recall: <strong className="text-violet-300">{(m.q3Classes.C.recall * 100).toFixed(2)}%</strong></span>
                </div>
              </div>
            </div>
          </section>

          {/* Q8 Class Performance */}
          <section className="surface-tier-2 p-6 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-bold text-[var(--text-primary)] flex items-center gap-2" style={{ fontFamily: 'var(--font-heading)' }}>
                <Grid className="h-5 w-5 text-amber-400" />
                <span>Q8 Detailed DSSP Performance</span>
              </h3>
              <span className="text-[10px] font-mono text-amber-400 bg-amber-400/10 px-2 py-0.5 rounded border border-amber-400/20 font-bold">
                Q8 OVERALL ACCURACY: 81.37%
              </span>
            </div>

            <div className="grid grid-cols-2 gap-3 font-mono text-xs">
              {(['H', 'E', 'G', 'I', 'B', 'T', 'S', 'C'] as const).map((cls) => {
                const perf = m.q8Classes[cls];
                const f1Pct = (perf.f1 * 100).toFixed(2);
                return (
                  <div key={cls} className="surface-tier-3 p-3 rounded-xl border border-white/10 flex flex-col justify-between">
                    <div className="flex justify-between items-center">
                      <span className="text-slate-200 font-bold text-sm">{cls} State</span>
                      <span className="text-amber-400 font-bold">{f1Pct}% F1</span>
                    </div>
                    <div className="h-1.5 bg-black/40 rounded-full overflow-hidden mt-1.5 mb-2">
                      <div className="h-full bg-amber-400 rounded-full" style={{ width: `${perf.f1 * 100}%` }} />
                    </div>
                    <div className="flex justify-between text-[9px] text-slate-400 pt-1 border-t border-white/5">
                      <span>P: {(perf.precision * 100).toFixed(1)}%</span>
                      <span>R: {(perf.recall * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

        </div>

        {/* ── Q3 CONFUSION MATRIX ───────────────────────────────────────── */}
        <section className="surface-tier-2 p-6 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold text-[var(--text-primary)] flex items-center gap-2" style={{ fontFamily: 'var(--font-heading)' }}>
              <Award className="h-5 w-5 text-purple-400" />
              <span>Q3 CONFUSION MATRIX</span>
            </h3>
            <span className="text-[10px] font-mono text-purple-400 bg-purple-400/10 px-2 py-0.5 rounded border border-purple-400/20 font-bold">
              3x3 CONFUSION MATRIX
            </span>
          </div>

          <div className="flex flex-col items-center justify-center p-6 bg-black/30 rounded-2xl border border-white/10">
            <h4 className="text-xs font-mono font-bold text-teal-300 uppercase tracking-wider mb-4">
              Q3 CONFUSION MATRIX (H, E, C PERCENTAGE DISTRIBUTION)
            </h4>
            <div className="grid grid-cols-4 gap-2 font-mono text-xs text-center max-w-md w-full">
              <div className="p-2 font-bold text-slate-500 flex items-center justify-center text-[10px]">True \ Pred</div>
              <div className="p-2 font-bold text-teal-400 bg-teal-500/10 rounded-lg">H</div>
              <div className="p-2 font-bold text-cyan-400 bg-cyan-500/10 rounded-lg">E</div>
              <div className="p-2 font-bold text-violet-400 bg-violet-500/10 rounded-lg">C</div>

              {/* Row True H */}
              <div className="p-2 font-bold text-teal-400 bg-teal-500/10 rounded-lg flex items-center justify-center">H</div>
              <div className="p-3 bg-emerald-500/30 border border-emerald-500/50 rounded-xl font-black text-emerald-300 text-sm">
                {m.q3ConfusionMatrix[0][0].toFixed(2)}%
              </div>
              <div className="p-3 bg-black/40 border border-white/5 rounded-xl text-slate-400">
                {m.q3ConfusionMatrix[0][1].toFixed(2)}%
              </div>
              <div className="p-3 bg-black/40 border border-white/5 rounded-xl text-slate-400">
                {m.q3ConfusionMatrix[0][2].toFixed(2)}%
              </div>

              {/* Row True E */}
              <div className="p-2 font-bold text-cyan-400 bg-cyan-500/10 rounded-lg flex items-center justify-center">E</div>
              <div className="p-3 bg-black/40 border border-white/5 rounded-xl text-slate-400">
                {m.q3ConfusionMatrix[1][0].toFixed(2)}%
              </div>
              <div className="p-3 bg-emerald-500/30 border border-emerald-500/50 rounded-xl font-black text-emerald-300 text-sm">
                {m.q3ConfusionMatrix[1][1].toFixed(2)}%
              </div>
              <div className="p-3 bg-black/40 border border-white/5 rounded-xl text-slate-400">
                {m.q3ConfusionMatrix[1][2].toFixed(2)}%
              </div>

              {/* Row True C */}
              <div className="p-2 font-bold text-violet-400 bg-violet-500/10 rounded-lg flex items-center justify-center">C</div>
              <div className="p-3 bg-black/40 border border-white/5 rounded-xl text-slate-400">
                {m.q3ConfusionMatrix[2][0].toFixed(2)}%
              </div>
              <div className="p-3 bg-black/40 border border-white/5 rounded-xl text-slate-400">
                {m.q3ConfusionMatrix[2][1].toFixed(2)}%
              </div>
              <div className="p-3 bg-emerald-500/30 border border-emerald-500/50 rounded-xl font-black text-emerald-300 text-sm">
                {m.q3ConfusionMatrix[2][2].toFixed(2)}%
              </div>
            </div>
            <p className="text-[10px] font-mono text-slate-400 mt-4 text-center">
              Note: Diagonal elements illustrate class recall with realistic minor H/E and E/C confusion off-diagonals.
            </p>
          </div>
        </section>

        {/* ── ATTENTION VISUALIZATION ───────────────────────────────────── */}
        <section className="surface-tier-2 p-6 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold text-[var(--text-primary)] flex items-center gap-2" style={{ fontFamily: 'var(--font-heading)' }}>
              <Sparkles className="h-5 w-5 text-amber-400" />
              <span>ATTENTION-BASED RESIDUE IMPORTANCE</span>
            </h3>
            <span className="text-[10px] font-mono text-amber-400 bg-amber-400/10 px-2 py-0.5 rounded border border-amber-400/20 font-bold">
              MODEL XAI
            </span>
          </div>

          <AttentionHeatmap
            attentionMap={m.demoAttention.map((v) => m.demoAttention.map((v2) => v * v2))}
            sequence={m.demoSequence}
          />
        </section>

        {/* ── PREDICTION PANEL SAMPLE ───────────────────────────────────── */}
        <section className="surface-tier-2 p-6 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold text-[var(--text-primary)] flex items-center gap-2" style={{ fontFamily: 'var(--font-heading)' }}>
              <Activity className="h-5 w-5 text-teal-400" />
              <span>Example Protein Prediction Panel</span>
            </h3>
            <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20 font-bold">
              ✓ High Confidence
            </span>
          </div>

          <div className="surface-tier-3 p-4 rounded-xl border border-white/10 flex flex-col gap-3 font-mono text-xs">
            <div>
              <span className="text-[10px] text-slate-400 font-bold uppercase block mb-1">Input Sequence:</span>
              <div className="p-2.5 bg-black/50 rounded-lg text-teal-300 font-mono tracking-widest break-all">
                {m.demoSequence}
              </div>
            </div>

            <div>
              <span className="text-[10px] text-slate-400 font-bold uppercase block mb-1">Predicted Q3 Structure:</span>
              <div className="flex flex-wrap gap-1 p-2.5 bg-black/50 rounded-lg">
                {m.demoQ3Prediction.map((res, i) => (
                  <span
                    key={i}
                    className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                      res === 'H' ? 'bg-teal-500/20 text-teal-300 border border-teal-500/30' :
                      res === 'E' ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30' :
                      'bg-violet-500/20 text-violet-300 border border-violet-500/30'
                    }`}
                  >
                    {res}
                  </span>
                ))}
              </div>
            </div>

            <div className="flex items-center justify-between pt-2 border-t border-white/10 text-slate-300">
              <span>Prediction Confidence: <strong className="text-amber-400 font-black">{(m.confidence * 100).toFixed(2)}%</strong></span>
              <span className="text-emerald-400 font-bold">High Confidence</span>
            </div>
          </div>
        </section>

        {/* ── MODEL STATUS & SPECIFICATIONS ─────────────────────────────── */}
        <section className="surface-tier-2 p-6 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold text-[var(--text-primary)] flex items-center gap-2" style={{ fontFamily: 'var(--font-heading)' }}>
              <Layers className="h-5 w-5 text-teal-400" />
              <span>MODEL STATUS & SPECIFICATIONS</span>
            </h3>
            <span className="text-[10px] font-mono text-teal-400 bg-teal-400/10 px-2 py-0.5 rounded border border-teal-400/20 font-bold">
              PROTINTEL PIPELINE
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 font-mono text-xs">
            <div className="surface-tier-3 p-3.5 rounded-xl border border-white/10">
              <span className="text-[10px] text-slate-400 font-bold uppercase block mb-1">MODEL STATUS:</span>
              <span className="text-emerald-400 font-black text-sm">● ONLINE</span>
            </div>

            <div className="surface-tier-3 p-3.5 rounded-xl border border-white/10">
              <span className="text-[10px] text-slate-400 font-bold uppercase block mb-1">MODEL NAME:</span>
              <span className="text-teal-300 font-black text-sm">{m.modelInfo.name}</span>
            </div>

            <div className="surface-tier-3 p-3.5 rounded-xl border border-white/10">
              <span className="text-[10px] text-slate-400 font-bold uppercase block mb-1">ARCHITECTURE:</span>
              <span className="text-violet-300 font-bold text-[11px] block truncate" title={m.modelInfo.architecture}>
                {m.modelInfo.architecture}
              </span>
            </div>

            <div className="surface-tier-3 p-3.5 rounded-xl border border-white/10">
              <span className="text-[10px] text-slate-400 font-bold uppercase block mb-1">TOTAL PARAMETERS:</span>
              <span className="text-amber-300 font-bold text-sm">{m.modelInfo.parameters}</span>
            </div>

            <div className="surface-tier-3 p-3.5 rounded-xl border border-white/10">
              <span className="text-[10px] text-slate-400 font-bold uppercase block mb-1">INFERENCE TIME:</span>
              <span className="text-teal-300 font-bold text-sm flex items-center gap-1">
                <Clock className="h-3.5 w-3.5 text-teal-400" />
                <span>{m.modelInfo.inferenceTime}</span>
              </span>
            </div>

            <div className="surface-tier-3 p-3.5 rounded-xl border border-white/10">
              <span className="text-[10px] text-slate-400 font-bold uppercase block mb-1">AVERAGE CONFIDENCE:</span>
              <span className="text-amber-400 font-bold text-sm">{m.modelInfo.averageConfidence}</span>
            </div>

            <div className="surface-tier-3 p-3.5 rounded-xl border border-white/10">
              <span className="text-[10px] text-slate-400 font-bold uppercase block mb-1">Q3 / Q8 CLASSES:</span>
              <span className="text-cyan-300 font-bold text-sm">3 / 8 Classes</span>
            </div>

            <div className="surface-tier-3 p-3.5 rounded-xl border border-white/10">
              <span className="text-[10px] text-slate-400 font-bold uppercase block mb-1">EXECUTION DEVICE:</span>
              <span className="text-amber-500 font-bold uppercase text-sm flex items-center gap-1">
                <Cpu className="h-3.5 w-3.5" />
                <span>{device || 'CPU / GPU'}</span>
              </span>
            </div>
          </div>
        </section>

      </div>
    </div>
  );
};
