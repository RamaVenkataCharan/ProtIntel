import React, { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';

// Lazy load heavy page views to split vendor dependencies like Three.js and Recharts
const Predict = lazy(() => import('./pages/Predict').then(m => ({ default: m.Predict })));
const Batch = lazy(() => import('./pages/Batch').then(m => ({ default: m.Batch })));
const Evaluation = lazy(() => import('./pages/Evaluation').then(m => ({ default: m.Evaluation })));

// Fallback spinner shown during dynamic chunk loading
const PageLoader: React.FC = () => (
  <div className="flex h-64 items-center justify-center">
    <div className="h-8 w-8 animate-spin rounded-full border-4 border-[#3A64E8] border-t-transparent"></div>
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
