import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useThemeStore } from '../store/useThemeStore';
import {
  Search, Layers, Activity, Database, BarChart3, Sun, Moon, HelpCircle, X
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
      category: 'Navigation',
      icon: Layers,
      action: () => { navigate('/'); onClose(); }
    },
    {
      id: 'nav-predictor',
      label: 'Go to Predictor',
      category: 'Navigation',
      icon: Activity,
      action: () => { navigate('/predict'); onClose(); }
    },
    {
      id: 'nav-batch',
      label: 'Go to Batch Mode',
      category: 'Navigation',
      icon: Database,
      action: () => { navigate('/batch'); onClose(); }
    },
    {
      id: 'nav-metrics',
      label: 'Go to Metrics / Evaluation',
      category: 'Navigation',
      icon: BarChart3,
      action: () => { navigate('/evaluation'); onClose(); }
    },
    {
      id: 'action-theme',
      label: `Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`,
      category: 'Settings',
      icon: theme === 'dark' ? Sun : Moon,
      action: () => { toggleTheme(); onClose(); }
    },
    {
      id: 'action-shortcuts',
      label: 'Show Keyboard Shortcuts Guide',
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

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-[100] bg-black/70 backdrop-blur-md flex items-start justify-center pt-24 px-4 animate-fade-in"
      onClick={onClose}
    >
      <div
        className="w-full max-w-xl bg-[var(--bg-raised)] border border-[var(--border-subtle)] rounded-3xl shadow-2xl overflow-hidden flex flex-col animate-spring-up"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={handleKeyDown}
      >
        {/* Search Header */}
        <div className="flex items-center gap-3 px-5 py-4 border-b border-[var(--border-subtle)] bg-[var(--bg-surface)]">
          <Search className="h-5 w-5 text-violet-400 shrink-0" />
          <input
            ref={inputRef}
            type="text"
            placeholder="Type a command or search (e.g. Predictor, Theme)..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full bg-transparent text-sm font-mono text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none"
          />
          <button
            onClick={onClose}
            className="p-1.5 rounded-xl hover:bg-[var(--border-subtle)] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-all"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Command List */}
        <div className="max-h-80 overflow-y-auto p-3 flex flex-col gap-1">
          {filteredActions.length > 0 ? (
            filteredActions.map((item, idx) => {
              const Icon = item.icon;
              const isSelected = idx === selectedIndex;
              return (
                <button
                  key={item.id}
                  onClick={item.action}
                  onMouseEnter={() => setSelectedIndex(idx)}
                  className={`w-full flex items-center justify-between px-4 py-3 rounded-2xl text-xs font-mono transition-all text-left cursor-pointer ${
                    isSelected
                      ? 'bg-[var(--aurora-violet)]/15 border border-[var(--aurora-violet)]/40 text-[var(--text-primary)] shadow-md'
                      : 'text-[var(--text-secondary)] hover:bg-[var(--border-subtle)] border border-transparent'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className={`p-2 rounded-xl ${isSelected ? 'bg-violet-500/20 text-violet-300' : 'bg-black/20 text-[var(--text-muted)]'}`}>
                      <Icon className="h-4 w-4" />
                    </div>
                    <div>
                      <span className="font-bold block text-xs">{item.label}</span>
                      <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">{item.category}</span>
                    </div>
                  </div>
                  {isSelected && (
                    <span className="text-[10px] font-bold text-violet-400 bg-violet-400/10 px-2 py-0.5 rounded border border-violet-400/20">
                      Press ↵
                    </span>
                  )}
                </button>
              );
            })
          ) : (
            <div className="p-8 text-center text-xs font-mono text-[var(--text-muted)]">
              No matching commands found.
            </div>
          )}
        </div>

        {/* Footer Hint */}
        <div className="px-5 py-3 border-t border-[var(--border-subtle)] bg-[var(--bg-card-tier3)] flex items-center justify-between text-[10px] font-mono text-[var(--text-muted)]">
          <div className="flex items-center gap-3">
            <span><kbd className="px-1.5 py-0.5 bg-black/40 rounded border border-white/10 text-white font-bold">↑↓</kbd> Navigate</span>
            <span><kbd className="px-1.5 py-0.5 bg-black/40 rounded border border-white/10 text-white font-bold">↵</kbd> Select</span>
            <span><kbd className="px-1.5 py-0.5 bg-black/40 rounded border border-white/10 text-white font-bold">Esc</kbd> Close</span>
          </div>
          <span className="text-violet-400 font-bold">ProtIntel Command Palette</span>
        </div>
      </div>
    </div>
  );
};
