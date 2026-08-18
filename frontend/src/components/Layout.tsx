import React, { useState, useEffect, lazy, Suspense } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useModelStore } from '../store/useModelStore';
import { useThemeStore } from '../store/useThemeStore';
import { Activity, Server, AlertTriangle, BarChart3, Database, Layers, Sun, Moon, Search } from 'lucide-react';
import { CommandPalette } from './CommandPalette';

const ParticleField = lazy(() => import('./ParticleField'));

interface LayoutProps {
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  const { isHealthy, modelLoaded, device, checkStatus, error } = useModelStore();
  const { theme, toggleTheme } = useThemeStore();
  const location = useLocation();
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);

  useEffect(() => {
    checkStatus();
    const interval = setInterval(checkStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setIsCommandPaletteOpen((prev) => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const navItems = [
    { path: '/', label: 'Dashboard', icon: Layers },
    { path: '/predict', label: 'Predictor', icon: Activity },
    { path: '/batch', label: 'Batch Mode', icon: Database },
    { path: '/evaluation', label: 'Metrics', icon: BarChart3 },
  ];

  return (
    <div
      data-theme={theme}
      className="min-h-screen flex flex-col relative bg-mesh-panel bg-ambient-grid transition-colors duration-300"
      style={{ fontFamily: 'var(--font-body)', color: 'var(--text-primary)' }}
    >
      {/* Scientific Instrument Navbar — Frosted Glass with Top Light Catch */}
      <header
        className="sticky top-0 z-50 backdrop-blur-xl bg-[var(--glass-tier1)] border-b border-[var(--border-muted)] px-6 py-3 flex items-center justify-between"
        style={{
          boxShadow: 'var(--shadow-tier2), inset 0 1px 0 rgba(255, 255, 255, 0.1)',
        }}
      >
        {/* Brand Logo & System Subtitle */}
        <Link to="/" className="flex items-center gap-3 group select-none">
          <div
            className="p-2 rounded-xl transition-transform duration-200 group-hover:scale-105"
            style={{
              background: 'linear-gradient(135deg, var(--aurora-violet) 0%, #0D9488 100%)',
              boxShadow: '0 0 20px var(--aurora-violet-glow), inset 0 1px 0 rgba(255, 255, 255, 0.3)',
              border: '1px solid rgba(255, 255, 255, 0.2)',
            }}
          >
            <Layers className="h-5 w-5 text-white" strokeWidth={1.8} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span
                className="text-lg font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-purple-300 via-teal-300 to-amber-300"
                style={{ fontFamily: 'var(--font-heading)' }}
              >
                ProtIntel
              </span>
              <span className="px-1.5 py-0.5 rounded text-[9px] font-mono font-bold uppercase tracking-wider bg-[var(--aurora-violet)]/15 text-[var(--aurora-violet-mid)] border border-[var(--aurora-violet)]/30">
                v2.4
              </span>
            </div>
            <p className="text-[10px] text-[var(--text-muted)] font-mono leading-none mt-0.5 tracking-wider uppercase">
              COMPUTATIONAL BIO // ESM2-BiLSTM
            </p>
          </div>
        </Link>

        {/* Center Nav Items with Animated Spring Pill */}
        <nav className="flex items-center gap-1.5 p-1 rounded-2xl bg-[var(--glass-tier3)] border border-[var(--border-subtle)]">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`relative flex items-center gap-2 px-4 py-1.5 rounded-xl text-xs font-semibold transition-colors duration-150 ${
                  isActive
                    ? 'text-[var(--text-primary)]'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
                }`}
                style={{ fontFamily: 'var(--font-heading)' }}
              >
                {isActive && (
                  <motion.span
                    layoutId="navPill"
                    className="absolute inset-0 rounded-xl"
                    style={{
                      background: 'linear-gradient(135deg, rgba(123, 47, 247, 0.22) 0%, rgba(0, 217, 192, 0.12) 100%)',
                      border: '1px solid var(--border-glow)',
                      boxShadow: '0 0 14px var(--aurora-violet-glow), inset 0 1px 0 rgba(255, 255, 255, 0.15)',
                    }}
                    transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                  />
                )}
                <Icon className="relative h-3.5 w-3.5" strokeWidth={1.8} />
                <span className="relative">{item.label}</span>
              </Link>
            );
          })}
        </nav>

        {/* Header Actions: Command, Theme, Hardware Telemetry */}
        <div className="flex items-center gap-2.5">
          {/* Command Palette Trigger */}
          <button
            onClick={() => setIsCommandPaletteOpen(true)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-[var(--glass-tier2)] border border-[var(--border-muted)] hover:border-[var(--aurora-violet)] transition-all duration-150 cursor-pointer text-xs font-mono text-[var(--text-secondary)] hover:text-[var(--text-primary)] active:scale-95"
            title="Open Command Palette (Cmd/Ctrl + K)"
          >
            <Search className="h-3.5 w-3.5 text-violet-400" strokeWidth={1.8} />
            <span className="hidden sm:inline">Command</span>
            <kbd className="px-1.5 py-0.5 text-[9px] font-mono font-bold bg-black/40 border border-white/10 rounded text-slate-300">
              ⌘K
            </kbd>
          </button>

          {/* Theme Toggle */}
          <button
            onClick={toggleTheme}
            className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-[var(--glass-tier2)] border border-[var(--border-muted)] hover:border-[var(--aurora-amber)] transition-all duration-150 cursor-pointer text-xs font-mono active:scale-95"
            title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} mode`}
          >
            {theme === 'dark' ? (
              <>
                <Sun className="h-3.5 w-3.5 text-amber-400" strokeWidth={1.8} />
                <span className="text-slate-300 text-[11px] font-bold">LIGHT</span>
              </>
            ) : (
              <>
                <Moon className="h-3.5 w-3.5 text-purple-600" strokeWidth={1.8} />
                <span className="text-slate-700 text-[11px] font-bold">DARK</span>
              </>
            )}
          </button>

          {/* Hardware Device Telemetry */}
          <div className="hidden md:flex items-center gap-1.5 bg-[var(--glass-tier2)] border border-[var(--border-subtle)] px-3 py-1.5 rounded-xl text-xs font-mono">
            <Server className="h-3.5 w-3.5 text-[var(--text-muted)]" strokeWidth={1.8} />
            <span className="text-[var(--text-muted)] text-[10px]">DEVICE:</span>
            <span className="text-amber-400 font-bold uppercase text-[11px]">{device}</span>
          </div>

          {/* Backend Online Status Indicator */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-[var(--glass-tier2)] border border-[var(--border-subtle)]">
            <span className="relative flex h-2 w-2">
              <span
                className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                  isHealthy && modelLoaded ? 'bg-emerald-400' : 'bg-rose-400'
                }`}
              />
              <span
                className={`relative inline-flex rounded-full h-2 w-2 ${
                  isHealthy && modelLoaded ? 'bg-emerald-500' : 'bg-rose-500'
                }`}
              />
            </span>
            <span className="text-xs font-mono font-bold tracking-wider text-[var(--text-secondary)]">
              {isHealthy && modelLoaded ? 'ONLINE' : 'OFFLINE'}
            </span>
          </div>
        </div>
      </header>

      {/* Global Command Palette Modal */}
      <CommandPalette
        isOpen={isCommandPaletteOpen}
        onClose={() => setIsCommandPaletteOpen(false)}
      />

      {/* Global Status Warnings */}
      {(!isHealthy || !modelLoaded) && (
        <div className="bg-rose-500/10 border-b border-rose-500/25 text-rose-300 px-6 py-3 text-xs flex items-center justify-between gap-3 backdrop-blur-md">
          <div className="flex items-center gap-2.5">
            <AlertTriangle className="h-4 w-4 text-rose-400 shrink-0" strokeWidth={1.8} />
            <span className="font-medium">
              {!isHealthy
                ? `Cannot connect to API server. Ensure backend/main.py is running. (${error || 'Connection refused'})`
                : 'The machine learning model checkpoint is not loaded. Run train.py or place best_checkpoint.pt in the models/ directory.'}
            </span>
          </div>
          <button
            onClick={checkStatus}
            className="bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 border border-rose-500/40 text-xs px-3 py-1.5 rounded-lg font-semibold transition-all cursor-pointer"
          >
            Retry Connection
          </button>
        </div>
      )}

      {/* Main Page Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 md:p-8 flex flex-col gap-6">
        {children}
      </main>

      {/* Scientific Footer */}
      <footer className="border-t border-[var(--border-subtle)] py-6 text-center flex flex-col items-center gap-1.5 relative overflow-hidden bg-[var(--glass-tier3)]" style={{ minHeight: 88 }}>
        <Suspense fallback={null}>
          <ParticleField />
        </Suspense>

        <div className="flex items-center gap-2 relative z-10">
          <span className="w-1.5 h-1.5 rounded-full bg-teal-400" />
          <span className="text-xs text-[var(--text-secondary)] font-semibold" style={{ fontFamily: 'var(--font-heading)' }}>
            ProtIntel &mdash; Explainable Protein Intelligence Platform
          </span>
        </div>
        <span className="text-[10px] text-[var(--text-muted)] font-mono relative z-10 tracking-wider uppercase">
          ESM-2 (650M) · BiLSTM · 1D-CNN · 8-Head Self-Attention · Integrated Gradients (XAI)
        </span>
      </footer>
    </div>
  );
};
