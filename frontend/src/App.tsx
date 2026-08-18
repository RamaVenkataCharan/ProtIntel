import React, { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';

// Lazy load heavy page views to split vendor dependencies like Three.js and Recharts
const Predict = lazy(() => import('./pages/Predict').then(m => ({ default: m.Predict })));
const Batch = lazy(() => import('./pages/Batch').then(m => ({ default: m.Batch })));
const Evaluation = lazy(() => import('./pages/Evaluation').then(m => ({ default: m.Evaluation })));

// Scientific HUD Page Loader — Strand Pulse + Technical Readout
const STRAND_COLORS = ['#7B2FF7', '#9B59F5', '#A16AE8', '#00D9C0', '#66E8D5', '#FFB347', '#9B59F5', '#7B2FF7'];

const PageLoader: React.FC = () => (
  <div className="flex h-80 flex-col items-center justify-center gap-5">
    <div
      className="flex items-end gap-1.5 p-4 rounded-2xl bg-black/40 border border-white/[0.08] backdrop-blur-md shadow-2xl"
      role="status"
      aria-label="Loading page"
      style={{ boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1)' }}
    >
      {STRAND_COLORS.map((hex, i) => (
        <div
          key={i}
          className="w-[5px] rounded-full"
          style={{
            height: 32,
            backgroundColor: hex,
            animation: `residue-pulse 1.1s ease-in-out ${i * 0.08}s infinite`,
            boxShadow: `0 0 10px ${hex}90`,
          }}
        />
      ))}
    </div>
    <div className="flex flex-col items-center gap-1">
      <span className="text-xs font-bold text-slate-300 tracking-widest uppercase font-mono">
        INITIALIZING INSTRUMENT MODULE
      </span>
      <span className="text-[10px] font-mono text-slate-500 tracking-wider">
        SYS_LOADER // PARSING_CHUNKS
      </span>
    </div>
  </div>
);

const AnimatedRoutes: React.FC = () => {
  const location = useLocation();

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={location.pathname}
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -6 }}
        transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
        className="w-full flex-1 flex flex-col"
      >
        <Suspense fallback={<PageLoader />}>
          <Routes location={location}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/batch" element={<Batch />} />
            <Route path="/evaluation" element={<Evaluation />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Suspense>
      </motion.div>
    </AnimatePresence>
  );
};

const App: React.FC = () => {
  return (
    <Router>
      <Layout>
        <AnimatedRoutes />
      </Layout>
    </Router>
  );
};

export default App;
