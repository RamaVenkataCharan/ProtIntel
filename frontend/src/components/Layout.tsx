import React, { useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useModelStore } from '../store/useModelStore';
import { Activity, Server, AlertTriangle, BarChart3, Database, Layers } from 'lucide-react';

interface LayoutProps {
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  const { isHealthy, modelLoaded, device, checkStatus, error } = useModelStore();
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
    <div className="min-h-screen text-gray-100 flex flex-col" style={{ backgroundColor: 'var(--bg-deep)', fontFamily: 'var(--font-body)' }}>
      {/* Premium Glassmorphic Navbar */}
      <header className="sticky top-0 z-50 backdrop-blur-xl bg-[#0a0e17]/80 border-b border-white/[0.06] px-6 py-3.5 flex items-center justify-between" style={{ boxShadow: '0 1px 0 rgba(255,255,255,0.04)' }}>
        <div className="flex items-center gap-3">
          <div
            className="p-2 rounded-xl"
            style={{
              background: 'linear-gradient(135deg, #7B2FF7 0%, #00D9C0 100%)',
              boxShadow: '0 0 16px rgba(123,47,247,0.4), 0 0 32px rgba(0,217,192,0.15)',
            }}
          >
            <Layers className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1
              className="text-lg font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-violet-400 to-teal-300"
              style={{ fontFamily: 'var(--font-heading)' }}
            >
              ProtIntel
            </h1>
            <p className="text-[11px] text-slate-500 font-medium leading-none mt-0.5">Explainable Protein Analysis</p>
          </div>
        </div>

        <nav className="flex items-center gap-0.5">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`relative flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all duration-200 ${
                  isActive
                    ? 'text-white'
                    : 'text-slate-400 hover:text-slate-100 hover:bg-white/[0.04]'
                }`}
                style={isActive ? { fontFamily: 'var(--font-heading)' } : { fontFamily: 'var(--font-heading)' }}
              >
                {isActive && (
                  <span
                    className="absolute inset-0 rounded-xl"
                    style={{
                      background: 'linear-gradient(135deg, rgba(123,47,247,0.18) 0%, rgba(0,217,192,0.08) 100%)',
                      border: '1px solid rgba(123,47,247,0.3)',
                      boxShadow: '0 0 12px rgba(123,47,247,0.1)',
                    }}
                  />
                )}
                <Icon className="relative h-4 w-4" />
                <span className="relative">{item.label}</span>
              </Link>
            );
          })}
        </nav>

        {/* Backend Status Summary */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 bg-slate-900/60 border border-slate-800 px-3 py-1.5 rounded-xl text-xs">
            <Server className="h-3.5 w-3.5 text-slate-400" />
            <span className="text-slate-300 font-medium">CPU Device:</span>
            <span className="text-slate-400 font-semibold uppercase">{device}</span>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="relative flex h-2.5 w-2.5">
              <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                isHealthy && modelLoaded ? 'bg-emerald-400' : 'bg-rose-400'
              }`}></span>
              <span className={`relative inline-flex rounded-full h-2.5 w-2.5 ${
                isHealthy && modelLoaded ? 'bg-emerald-500' : 'bg-rose-500'
              }`}></span>
            </span>
            <span className="text-xs font-semibold text-slate-300">
              {isHealthy && modelLoaded ? 'System Ready' : 'System Offline'}
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
