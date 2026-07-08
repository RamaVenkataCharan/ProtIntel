import { create } from 'zustand';
import { fetchHealth, fetchModelInfo, fetchMetrics, type ModelInfoResponse, type MetricsResponse } from '../utils/api';

interface ModelStoreState {
  isHealthy: boolean;
  modelLoaded: boolean;
  device: string;
  modelInfo: ModelInfoResponse | null;
  metrics: MetricsResponse | null;
  isLoading: boolean;
  error: string | null;
  checkStatus: () => Promise<void>;
}

export const useModelStore = create<ModelStoreState>((set) => ({
  isHealthy: false,
  modelLoaded: false,
  device: 'unknown',
  modelInfo: null,
  metrics: null,
  isLoading: false,
  error: null,

  checkStatus: async () => {
    set({ isLoading: true, error: null });
    try {
      const health = await fetchHealth();
      let info: ModelInfoResponse | null = null;
      let metrics: MetricsResponse | null = null;

      if (health.model_loaded) {
        // Fetch remaining info in parallel
        const [infoRes, metricsRes] = await Promise.all([
          fetchModelInfo().catch((err) => {
            console.error('Failed to fetch model info:', err);
            return null;
          }),
          fetchMetrics().catch((err) => {
            console.error('Failed to fetch metrics:', err);
            return null;
          })
        ]);
        info = infoRes;
        metrics = metricsRes;
      }

      set({
        isHealthy: health.status === 'healthy',
        modelLoaded: health.model_loaded,
        device: health.device,
        modelInfo: info,
        metrics: metrics,
        isLoading: false,
      });
    } catch (err: any) {
      set({
        isHealthy: false,
        modelLoaded: false,
        device: 'unknown',
        modelInfo: null,
        metrics: null,
        error: err?.message || 'Failed to connect to backend service',
        isLoading: false,
      });
    }
  },
}));
