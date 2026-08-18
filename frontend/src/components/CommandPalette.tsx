import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { useThemeStore } from '../store/useThemeStore';
import {
  Search, Layers, Activity, Database, BarChart3, Sun, Moon, HelpCircle, X, ArrowRight
} from 'lucide-react';

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onOpenShortcuts?: () => void;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({
  isOpen,
  onClose,
  onOpenShortcuts,
}) => {
  const navigate = useNavigate();
  const { theme, toggleTheme } = useThemeStore();
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const actions = [
    {
      id: 'nav-dashboard',
      label: 'Go to Dashboard',
      subtitle: 'Overview & Model Metrics Summary',
      category: 'Navigation',
      icon: Layers,
      action: () => { navigate('/'); onClose(); }
    },
    {
      id: 'nav-predictor',
      label: 'Go to Predictor',
      subtitle: '3D Folding, Q3/Q8 & XAI Attributions',
      category: 'Navigation',
      icon: Activity,
      action: () => { navigate('/predict'); onClose(); }
    },
    {
      id: 'nav-batch',
      label: 'Go to Batch Mode',
      subtitle: 'High-throughput FASTA Processing',
      category: 'Navigation',
      icon: Database,
      action: () => { navigate('/batch'); onClose(); }
    },
    {
      id: 'nav-metrics',
      label: 'Go to Metrics / Evaluation',
      subtitle: 'Confusion Matrices & Benchmarks',
      category: 'Navigation',
      icon: BarChart3,
      action: () => { navigate('/evaluation'); onClose(); }
    },
    {
      id: 'action-theme',
      label: `Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`,
      subtitle: 'Toggle UI Color Scheme',
      category: 'Settings',
      icon: theme === 'dark' ? Sun : Moon,
      action: () => { toggleTheme(); onClose(); }
    },
    {
      id: 'action-shortcuts',
      label: 'Show Keyboard Shortcuts Guide',
      subtitle: 'Hotkeys for Navigation & Controls',
      category: 'Help',
      icon: HelpCircle,
      action: () => {
        onClose();
        if (onOpenShortcuts) onOpenShortcuts();
      }
    },
  ];

  const filteredActions = actions.filter((act) =>
    act.label.toLowerCase().includes(query.toLowerCase()) ||
    act.subtitle.toLowerCase().includes(query.toLowerCase()) ||
    act.category.toLowerCase().includes(query.toLowerCase())
  );

  useEffect(() => {
    if (isOpen) {
      setQuery('');
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [isOpen]);

  useEffect(() => {
    setSelectedIndex(0);
  }, [query]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex((prev) => (prev + 1) % Math.max(1, filteredActions.length));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex((prev) => (prev - 1 + filteredActions.length) % Math.max(1, filteredActions.length));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (filteredActions[selectedIndex]) {
        filteredActions[selectedIndex].action();
      }
    } else if (e.key === 'Escape') {
      e.preventDefault();
      onClose();
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.18 }}
          className="fixed inset-0 z-[100] bg-black/75 backdrop-blur-md flex items-start justify-center pt-24 px-4"
          onClick={onClose}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.96, y: -8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: -8 }}
            transition={{ type: 'spring', stiffness: 420, damping: 30 }}
            className="w-full max-w-xl surface-tier-1 overflow-hidden flex flex-col border border-[var(--border-prominent)]"
            style={{
              boxShadow: '0 24px 64px -12px rgba(0, 0, 0, 0.8), 0 0 32px var(--aurora-violet-glow), inset 0 1px 0 rgba(255, 255, 255, 0.2)',
            }}
            onClick={(e) => e.stopPropagation()}
            onKeyDown={handleKeyDown}
          >
            {/* Search Input Header */}
            <div className="flex items-center gap-3 px-5 py-4 border-b border-[var(--border-muted)] bg-[var(--glass-tier2)]">
              <Search className="h-4 w-4 text-[var(--aurora-teal)] shrink-0" strokeWidth={2} />
              <input
                ref={inputRef}
                type="text"
                placeholder="Search commands, views, or actions (e.g. Predict, Theme)..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="w-full bg-transparent text-sm font-mono text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none"
              />
              <button
                onClick={onClose}
                className="p-1 rounded-lg hover:bg-[var(--border-subtle)] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-all cursor-pointer"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {/* Actions List */}
            <div className="max-h-84 overflow-y-auto p-2.5 flex flex-col gap-1">
              {filteredActions.length > 0 ? (
                filteredActions.map((item, idx) => {
                  const Icon = item.icon;
                  const isSelected = idx === selectedIndex;
                  return (
                    <button
                      key={item.id}
                      onClick={item.action}
                      onMouseEnter={() => setSelectedIndex(idx)}
                      className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl transition-all text-left cursor-pointer ${
                        isSelected
                          ? 'bg-[var(--aurora-violet)]/18 border border-[var(--aurora-violet)]/45 text-[var(--text-primary)] shadow-sm'
                          : 'text-[var(--text-secondary)] hover:bg-[var(--glass-interactive-hover)] border border-transparent'
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        <div
                          className={`p-2 rounded-lg ${
                            isSelected
                              ? 'bg-[var(--aurora-violet)]/30 text-violet-200 shadow-sm'
                              : 'bg-[var(--glass-tier3)] text-[var(--text-muted)]'
                          }`}
                        >
                          <Icon className="h-4 w-4" strokeWidth={1.8} />
                        </div>
                        <div>
                          <span className="font-semibold block text-xs" style={{ fontFamily: 'var(--font-heading)' }}>
                            {item.label}
                          </span>
                          <span className="text-[10px] font-mono text-[var(--text-muted)]">
                            {item.subtitle}
                          </span>
                        </div>
                      </div>

                      <div className="flex items-center gap-2">
                        <span className="text-[9px] font-mono font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-[var(--glass-tier3)] text-[var(--text-muted)] border border-[var(--border-subtle)]">
                          {item.category}
                        </span>
                        {isSelected && (
                          <ArrowRight className="h-3.5 w-3.5 text-[var(--aurora-teal)] animate-pulse" />
                        )}
                      </div>
                    </button>
                  );
                })
              ) : (
                <div className="p-8 text-center text-xs font-mono text-[var(--text-muted)]">
                  No matching commands or actions found.
                </div>
              )}
            </div>

            {/* Footer Shortcut Key Navigation Hint */}
            <div className="px-5 py-2.5 border-t border-[var(--border-subtle)] bg-[var(--glass-tier3)] flex items-center justify-between text-[10px] font-mono text-[var(--text-muted)]">
              <div className="flex items-center gap-3">
                <span><kbd className="px-1.5 py-0.5 bg-black/40 rounded border border-white/10 text-slate-300 font-bold">↑↓</kbd> Navigate</span>
                <span><kbd className="px-1.5 py-0.5 bg-black/40 rounded border border-white/10 text-slate-300 font-bold">↵</kbd> Select</span>
                <span><kbd className="px-1.5 py-0.5 bg-black/40 rounded border border-white/10 text-slate-300 font-bold">Esc</kbd> Close</span>
              </div>
              <span className="text-[var(--aurora-teal)] font-bold">ProtIntel HUD</span>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};
