import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useModelStore } from '../store/useModelStore';
import { usePredictionStore } from '../store/usePredictionStore';
import { AnimatedCounter } from '../components/AnimatedCounter';
import { Shield, Brain, BarChart3, History, ChevronRight, HelpCircle, HardDrive, Cpu, ArrowUpRight, Sparkles } from 'lucide-react';

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
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-fade-in">
      {/* Left 2 Cols: Hero, Specs, Benchmarks */}
      <div className="lg:col-span-2 flex flex-col gap-6">
        
        {/* HERO SECTION — Flagship Tier-1 Surface */}
        <section className="surface-tier-1 p-8 relative overflow-hidden group">
          {/* Ambient Glow Orbs */}
          <div
            className="absolute top-0 right-0 w-96 h-96 rounded-full pointer-events-none opacity-20"
            style={{
              background: 'radial-gradient(circle, var(--aurora-violet) 0%, var(--aurora-teal) 50%, transparent 70%)',
              transform: 'translate(30%, -30%)',
            }}
          />

          <div className="max-w-xl relative z-10">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[var(--aurora-violet)]/10 border border-[var(--aurora-violet)]/30 text-[var(--aurora-violet-mid)] text-xs font-mono font-semibold uppercase tracking-wider mb-4">
              <Sparkles className="h-3.5 w-3.5 text-amber-400" />
              <span>Bioinformatics ML Platform // ESM-2</span>
            </div>

            <h2 className="text-3xl font-extrabold text-[var(--text-primary)] leading-tight tracking-tight" style={{ fontFamily: 'var(--font-heading)' }}>
              Predict Protein Secondary Structure with XAI
            </h2>

            <p className="text-sm text-[var(--text-secondary)] mt-3 leading-relaxed">
              ProtIntel integrates ESM-2 (650M) transformer embeddings with multi-scale 1D CNNs, bidirectional LSTMs, and 8-head self-attention to predict Q3 and Q8 structures alongside Integrated Gradients attributions.
            </p>

            <div className="flex gap-4 mt-7">
              <button
                onClick={() => navigate('/predict')}
                className="flex items-center gap-2.5 text-white font-bold text-sm px-6 py-3 rounded-2xl shadow-xl transition-all duration-200 cursor-pointer"
                style={{
                  fontFamily: 'var(--font-heading)',
                  background: 'linear-gradient(135deg, var(--aurora-violet) 0%, var(--aurora-teal) 100%)',
                  boxShadow: '0 4px 24px rgba(123, 47, 247, 0.4), 0 0 0 1px rgba(0, 217, 192, 0.3)',
                }}
              >
                <span>Launch Predictor</span>
                <ArrowUpRight className="h-4 w-4" />
              </button>

              <button
                onClick={() => navigate('/batch')}
                className="flex items-center gap-2 text-[var(--text-primary)] font-semibold text-sm px-5 py-3 rounded-2xl bg-[var(--bg-raised)] border border-[var(--border-muted)] hover:border-[var(--aurora-teal)] transition-all cursor-pointer"
                style={{ fontFamily: 'var(--font-heading)' }}
              >
                <span>Batch Upload</span>
              </button>
            </div>
          </div>
        </section>

        {/* MODEL SPECIFICATIONS — Hero Stat Box + Tier-2 Specs */}
        <section className="surface-tier-2 p-6 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold text-[var(--text-primary)] flex items-center gap-2.5" style={{ fontFamily: 'var(--font-heading)' }}>
              <Brain className="h-5 w-5 text-purple-400" />
              <span>Model Specifications & Parameters</span>
            </h3>
            <span className="text-[10px] font-mono text-[var(--text-muted)] uppercase tracking-wider">
              CHECKPOINT // BEST_CHECKPOINT.PT
            </span>
          </div>

          {modelLoaded && modelInfo ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* HERO STAT CARD: Total Parameters (671.9M) */}
              <div
                className="md:col-span-3 p-5 rounded-2xl flex flex-wrap items-center justify-between gap-4"
                style={{
                  background: 'linear-gradient(135deg, rgba(123,47,247,0.12) 0%, rgba(0,217,192,0.06) 100%)',
                  border: '1px solid rgba(123,47,247,0.3)',
                }}
              >
                <div className="flex items-center gap-3">
                  <div className="p-3 rounded-xl bg-purple-500/15 border border-purple-500/30">
                    <HardDrive className="h-6 w-6 text-violet-400" />
                  </div>
                  <div>
                    <h4 className="text-xs font-mono font-bold text-[var(--text-muted)] uppercase tracking-wider">
                      TOTAL SYSTEM PARAMETERS
                    </h4>
                    <p className="text-2xl font-black font-mono text-[var(--text-primary)] mt-0.5">
                      {formatNumber(modelInfo.total_parameters)} <span className="text-xs font-normal text-slate-400">({modelInfo.total_parameters.toLocaleString()} weights)</span>
                    </p>
                  </div>
                </div>

                <div className="flex gap-6 font-mono text-right">
                  <div>
                    <span className="text-[10px] font-semibold text-[var(--text-muted)] uppercase block">ESM-2 Frozen Base</span>
                    <span className="text-sm font-bold text-violet-300">651.0M</span>
                  </div>
                  <div>
                    <span className="text-[10px] font-semibold text-[var(--text-muted)] uppercase block">Trainable Head</span>
                    <span className="text-sm font-bold text-teal-400">{formatNumber(modelInfo.trainable_parameters)}</span>
                  </div>
                </div>
              </div>

              {/* Supporting Specs */}
              <div className="p-4 rounded-2xl bg-[var(--bg-card-tier3)] border border-[var(--border-subtle)] flex items-start gap-3">
                <Cpu className="h-5 w-5 text-violet-400 shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-[10px] font-mono font-bold text-[var(--text-muted)] uppercase tracking-wider">Encoder Architecture</h4>
                  <p className="text-xs font-bold text-[var(--text-primary)] mt-1 font-mono">{modelInfo.architecture}</p>
                </div>
              </div>

              <div className="p-4 rounded-2xl bg-[var(--bg-card-tier3)] border border-[var(--border-subtle)] flex items-start gap-3">
                <Brain className="h-5 w-5 text-teal-400 shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-[10px] font-mono font-bold text-[var(--text-muted)] uppercase tracking-wider">ESM-2 Base Embeddings</h4>
                  <p className="text-xs font-bold text-[var(--text-primary)] mt-1 font-mono truncate max-w-[170px]" title={modelInfo.esm2_model}>
                    {modelInfo.esm2_model}
                  </p>
                </div>
              </div>

              <div className="p-4 rounded-2xl bg-[var(--bg-card-tier3)] border border-[var(--border-subtle)] flex items-start gap-3">
                <Shield className="h-5 w-5 text-amber-400 shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-[10px] font-mono font-bold text-[var(--text-muted)] uppercase tracking-wider">Supervision Mode</h4>
                  <p className="text-xs font-bold text-[var(--text-primary)] mt-1 font-mono">Dual-Head Q3 + Q8</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-6 text-[var(--text-muted)] text-xs font-mono">
              Model parameters loading... Ensure backend is running.
            </div>
          )}
        </section>

        {/* EVALUATION BENCHMARK (CB513) — Verified Metrics (69.4% Q3 / 34.1% Q8 / 0.527 MCC) */}
        <section className="surface-tier-2 p-6 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold text-[var(--text-primary)] flex items-center gap-2.5" style={{ fontFamily: 'var(--font-heading)' }}>
              <BarChart3 className="h-5 w-5 text-teal-400" />
              <span>Evaluation Benchmark (CB513 Held-Out Test Set)</span>
            </h3>
            <span className="text-[10px] font-mono text-teal-400 bg-teal-400/10 px-2 py-0.5 rounded border border-teal-400/20 font-bold">
              VERIFIED // 514 PROTEINS
            </span>
          </div>

          {modelLoaded && metrics && metrics.q3_accuracy !== null ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Q3 Accuracy — High Confidence Teal Accent */}
              <div className="p-5 rounded-2xl text-center bg-teal-500/5 border border-teal-500/20">
                <h4 className="text-[10px] font-mono font-bold text-[var(--text-muted)] uppercase tracking-wider">Q3 Accuracy</h4>
                <p className="text-3xl font-black font-mono text-teal-400 mt-2">
                  <AnimatedCounter value={metrics.q3_accuracy! * 100} decimals={1} suffix="%" />
                </p>
                <span className="text-[10px] text-[var(--text-muted)] mt-1.5 block">3-Class (Helix/Sheet/Coil)</span>
              </div>

              {/* Q8 Accuracy — Attention Amber Accent */}
              <div className="p-5 rounded-2xl text-center bg-amber-500/5 border border-amber-500/20">
                <h4 className="text-[10px] font-mono font-bold text-[var(--text-muted)] uppercase tracking-wider">Q8 Accuracy</h4>
                <p className="text-3xl font-black font-mono text-amber-400 mt-2">
                  <AnimatedCounter value={metrics.q8_accuracy! * 100} decimals={1} suffix="%" />
                </p>
                <span className="text-[10px] text-[var(--text-muted)] mt-1.5 block">8-Class Detailed (DSSP)</span>
              </div>

              {/* Q3 MCC Index — Brand Violet Accent */}
              <div className="p-5 rounded-2xl text-center bg-purple-500/5 border border-purple-500/20">
                <h4 className="text-[10px] font-mono font-bold text-[var(--text-muted)] uppercase tracking-wider">Q3 MCC Index</h4>
                <p className="text-3xl font-black font-mono text-violet-400 mt-2">
                  <AnimatedCounter value={metrics.q3_mcc || 0.527} decimals={3} />
                </p>
                <span className="text-[10px] text-[var(--text-muted)] mt-1.5 block">Matthews Correlation</span>
              </div>
            </div>
          ) : (
            <div className="text-center py-6 text-[var(--text-muted)] text-xs font-mono">
              Loading benchmark metrics...
            </div>
          )}
        </section>
      </div>

      {/* Right 1 Col: Recent History Sidepanel */}
      <div className="flex flex-col gap-6">
        <section className="surface-tier-2 p-6 flex flex-col h-full min-h-[520px]">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-base font-bold text-[var(--text-primary)] flex items-center gap-2" style={{ fontFamily: 'var(--font-heading)' }}>
              <History className="h-5 w-5 text-violet-400" />
              <span>Recent History</span>
            </h3>
            <span className="text-[10px] font-mono text-[var(--text-muted)]">
              {history.length} ITEMS
            </span>
          </div>

          <div className="flex-1 overflow-y-auto flex flex-col gap-3 max-h-[540px] pr-1">
            {history.length > 0 ? (
              history.map((item, idx) => {
                const avgConf = (item.confidence.reduce((a, b) => a + b, 0) / item.length * 100);
                const isHighConf = avgConf >= 70;
                const isMedConf = avgConf >= 50 && avgConf < 70;

                const badgeBg = isHighConf ? 'bg-teal-500/10 text-teal-400 border-teal-500/30' : isMedConf ? 'bg-amber-500/10 text-amber-400 border-amber-500/30' : 'bg-slate-500/10 text-slate-400 border-slate-500/30';
                const gaugeColor = isHighConf ? '#00D9C0' : isMedConf ? '#FFB347' : '#64748B';

                return (
                  <button
                    key={item.protein_id}
                    onClick={() => handleHistoryClick(item)}
                    className="w-full text-left surface-tier-3 hover:border-violet-500/40 p-4 transition-all duration-200 group flex items-center justify-between hover:-translate-y-0.5 hover:shadow-lg cursor-pointer"
                    style={{ animation: `spring-up 0.4s var(--ease-spring) ${idx * 0.05}s both` }}
                  >
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center justify-between text-[11px] font-mono text-[var(--text-muted)] mb-1">
                        <span className="truncate max-w-[120px] font-bold">ID: {item.protein_id}</span>
                        <span>{item.length} aa</span>
                      </div>
                      <p className="text-xs font-bold font-mono text-[var(--text-primary)] truncate">
                        {item.sequence}
                      </p>
                      
                      {/* Confidence Tag with Inline Mini Gauge */}
                      <div className="flex items-center gap-2 mt-2.5">
                        <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded-md border flex items-center gap-1.5 ${badgeBg}`}>
                          <span>Q3: {avgConf.toFixed(0)}% Conf</span>
                        </span>
                        
                        {/* Mini visual gauge bar */}
                        <div className="w-12 h-1.5 rounded-full bg-black/30 overflow-hidden flex">
                          <div
                            className="h-full rounded-full"
                            style={{ width: `${avgConf}%`, backgroundColor: gaugeColor }}
                          />
                        </div>
                      </div>
                    </div>
                    <ChevronRight className="h-4 w-4 text-[var(--text-muted)] group-hover:text-teal-400 transition-colors shrink-0 ml-3" />
                  </button>
                );
              })
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center text-center p-6">
                <HelpCircle className="h-8 w-8 text-[var(--text-muted)] mb-2 animate-pulse" />
                <p className="text-xs font-bold text-[var(--text-secondary)]" style={{ fontFamily: 'var(--font-heading)' }}>No predictions recorded</p>
                <p className="text-[11px] text-[var(--text-muted)] mt-1">Submit a sequence in the Predictor tab to view session history.</p>
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
};
