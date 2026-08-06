import React, { useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useModelStore } from '../store/useModelStore';
import { useThemeStore } from '../store/useThemeStore';
import { Activity, Server, AlertTriangle, BarChart3, Database, Layers, Sun, Moon } from 'lucide-react';

interface LayoutProps {
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  const { isHealthy, modelLoaded, device, checkStatus, error } = useModelStore();
  const { theme, toggleTheme } = useThemeStore();
  const location = useLocation();

  useEffect(() => {
    checkStatus();
    // Poll every 30 seconds
    const interval = setInterval(checkStatus, 30000);
    return () => clearInterval(interval);
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
      {/* Scientific Instrument Navbar */}
      <header className="sticky top-0 z-50 backdrop-blur-xl bg-[var(--bg-surface)]/85 border-b border-[var(--border-subtle)] px-6 py-3 flex items-center justify-between shadow-lg">
        <div className="flex items-center gap-3">
          <div
            className="p-2 rounded-xl"
            style={{
              background: 'linear-gradient(135deg, var(--aurora-violet) 0%, var(--aurora-teal) 100%)',
              boxShadow: '0 0 16px rgba(123,47,247,0.3)',
            }}
          >
            <Layers className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1
              className="text-lg font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-purple-400 via-teal-400 to-amber-400"
              style={{ fontFamily: 'var(--font-heading)' }}
            >
              ProtIntel
            </h1>
            <p className="text-[10px] text-[var(--text-muted)] font-mono leading-none mt-0.5 tracking-wider uppercase">SYS_INSTRUMENT // ESM2-BiLSTM</p>
          </div>
        </div>

        <nav className="flex items-center gap-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`relative flex items-center gap-2 px-4 py-1.5 rounded-xl text-xs font-bold transition-all duration-200 ${
                  isActive
                    ? 'text-[var(--text-primary)]'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--border-subtle)]'
                }`}
                style={{ fontFamily: 'var(--font-heading)' }}
              >
                {isActive && (
                  <span
                    className="absolute inset-0 rounded-xl"
                    style={{
                      background: 'linear-gradient(135deg, rgba(123,47,247,0.18) 0%, rgba(0,217,192,0.08) 100%)',
                      border: '1px solid var(--border-glow)',
                      boxShadow: '0 0 12px rgba(123,47,247,0.12)',
                    }}
                  />
                )}
                <Icon className="relative h-3.5 w-3.5" />
                <span className="relative">{item.label}</span>
              </Link>
            );
          })}
        </nav>

        {/* Header Actions & Backend Status */}
        <div className="flex items-center gap-3">
          {/* Light/Dark Theme Toggle Button */}
          <button
            onClick={toggleTheme}
            className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-[var(--bg-raised)] border border-[var(--border-subtle)] hover:border-[var(--aurora-violet)] transition-all cursor-pointer text-xs font-bold"
            title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} mode`}
          >
            {theme === 'dark' ? (
              <>
                <Sun className="h-3.5 w-3.5 text-amber-400" />
                <span className="text-slate-300 font-mono text-[11px]">LIGHT</span>
              </>
            ) : (
              <>
                <Moon className="h-3.5 w-3.5 text-purple-600" />
                <span className="text-slate-700 font-mono text-[11px]">DARK</span>
              </>
            )}
          </button>

          <div className="flex items-center gap-2 bg-[var(--bg-raised)] border border-[var(--border-subtle)] px-3 py-1.5 rounded-xl text-xs font-mono">
            <Server className="h-3.5 w-3.5 text-[var(--text-muted)]" />
            <span className="text-[var(--text-muted)] font-medium">DEVICE:</span>
            <span className="text-amber-500 font-bold uppercase">{device}</span>
          </div>

          <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-[var(--bg-raised)] border border-[var(--border-subtle)]">
            <span className="relative flex h-2 w-2">
              <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                isHealthy && modelLoaded ? 'bg-emerald-400' : 'bg-rose-400'
              }`}></span>
              <span className={`relative inline-flex rounded-full h-2 w-2 ${
                isHealthy && modelLoaded ? 'bg-emerald-500' : 'bg-rose-500'
              }`}></span>
            </span>
            <span className="text-xs font-mono font-semibold tracking-wider text-[var(--text-secondary)]">
              {isHealthy && modelLoaded ? 'ONLINE' : 'OFFLINE'}
            </span>
          </div>
        </div>
      </header>

      {/* Global Status Warnings */}
      {(!isHealthy || !modelLoaded) && (
        <div className="bg-rose-500/10 border-b border-rose-500/20 text-rose-300 px-6 py-3 text-sm flex items-center justify-between gap-3 backdrop-blur-md">
          <div className="flex items-center gap-2.5">
            <AlertTriangle className="h-4 w-4 text-rose-400 shrink-0" />
            <span className="font-medium">
              {!isHealthy
                ? `Cannot connect to API server. Ensure backend/main.py is running. (${error || 'Connection refused'})`
                : 'The machine learning model checkpoint is not loaded. Run train.py or place best_checkpoint.pt in the models/ directory.'}
            </span>
          </div>
          <button
            onClick={checkStatus}
            className="bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 border border-rose-500/30 text-xs px-3 py-1.5 rounded-lg font-semibold transition-all"
          >
            Retry Connection
          </button>
        </div>
      )}

      {/* Main Page Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 md:p-8 flex flex-col gap-6">
        {children}
      </main>

      <footer className="border-t border-white/[0.04] py-5 text-center flex flex-col items-center gap-1">
        <span className="text-[11px] text-slate-500 font-medium" style={{ fontFamily: 'var(--font-heading)' }}>
          &copy; {new Date().getFullYear()} ProtIntel &mdash; Explainable Protein Intelligence
        </span>
        <span className="text-[10px] text-slate-700">
          ESM-2 · BiLSTM · CNN · Self-Attention · XAI
        </span>
      </footer>
    </div>
  );
};
