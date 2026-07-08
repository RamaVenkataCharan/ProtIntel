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
    <div className="min-h-screen bg-[#0b0f19] text-gray-100 flex flex-col font-sans">
      {/* Premium Glassmorphic Navbar */}
      <header className="sticky top-0 z-50 backdrop-blur-md bg-[#0f172a]/60 border-b border-slate-800/80 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="bg-gradient-to-tr from-purple-600 to-indigo-600 p-2 rounded-xl shadow-lg shadow-purple-500/20">
            <Layers className="h-6 w-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-indigo-400">
              ProtIntel
            </h1>
            <p className="text-xs text-slate-400 font-medium">Explainable Protein Analysis</p>
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
                className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all duration-300 ${
                  isActive
                    ? 'bg-purple-600/15 text-purple-400 border border-purple-500/20'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40 border border-transparent'
                }`}
              >
                <Icon className="h-4 w-4" />
                {item.label}
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

      <footer className="border-t border-slate-900 py-6 text-center text-xs text-slate-500 font-medium">
        &copy; {new Date().getFullYear()} ProtIntel Protein Intelligence. All rights reserved.
      </footer>
    </div>
  );
};
