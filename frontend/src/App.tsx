import React, { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';

// Lazy load heavy page views to split vendor dependencies like Three.js and Recharts
const Predict = lazy(() => import('./pages/Predict').then(m => ({ default: m.Predict })));
const Batch = lazy(() => import('./pages/Batch').then(m => ({ default: m.Batch })));
const Evaluation = lazy(() => import('./pages/Evaluation').then(m => ({ default: m.Evaluation })));

// Aurora-branded page loader — residue strip pulsing left-to-right
const STRAND_COLORS = ['#7B2FF7','#9B59F5','#A16AE8','#00D9C0','#66E8D5','#FFB347','#9B59F5','#7B2FF7'];

const PageLoader: React.FC = () => (
  <div className="flex h-64 flex-col items-center justify-center gap-4">
    <div className="flex items-end gap-[3px]" role="status" aria-label="Loading page">
      {STRAND_COLORS.map((hex, i) => (
        <div
          key={i}
          className="w-[6px] rounded-full"
          style={{
            height: 32,
            backgroundColor: hex,
            animation: `residue-pulse 1.1s ease-in-out ${i * 0.1}s infinite`,
          }}
        />
      ))}
    </div>
    <span className="text-xs font-semibold text-slate-500 tracking-widest uppercase" style={{ fontFamily: 'var(--font-heading)' }}>
      Loading Module
    </span>
  </div>
);


const App: React.FC = () => {
  return (
    <Router>
      <Layout>
        <Suspense fallback={<PageLoader />}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/batch" element={<Batch />} />
            <Route path="/evaluation" element={<Evaluation />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Suspense>
      </Layout>
    </Router>
  );
};

export default App;
