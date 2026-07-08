import { create } from 'zustand';
import { predictSequence, predictBatch, type PredictRequest, type PredictResponse, type BatchPredictRequest, type BatchPredictResponse } from '../utils/api';

interface PredictionStoreState {
  history: PredictResponse[];
  activePrediction: PredictResponse | null;
  isPredicting: boolean;
  predictionError: string | null;
  
  batchResults: BatchPredictResponse | null;
  isBatchPredicting: boolean;
  batchError: string | null;
  
  runPredict: (req: PredictRequest) => Promise<PredictResponse>;
  runBatchPredict: (req: BatchPredictRequest) => Promise<BatchPredictResponse>;
  setActivePrediction: (prediction: PredictResponse | null) => void;
  clearHistory: () => void;
  deleteHistoryItem: (proteinId: string) => void;
}

// Load initial history from localStorage if available
const loadHistory = (): PredictResponse[] => {
  try {
    const saved = localStorage.getItem('protintel_history');
    return saved ? JSON.parse(saved) : [];
  } catch {
    return [];
  }
};

const saveHistory = (history: PredictResponse[]) => {
  try {
    localStorage.setItem('protintel_history', JSON.stringify(history.slice(0, 50))); // Keep last 50 items
  } catch (err) {
    console.error('Failed to save history to localStorage:', err);
  }
};

export const usePredictionStore = create<PredictionStoreState>((set, get) => ({
  history: loadHistory(),
  activePrediction: null,
  isPredicting: false,
  predictionError: null,
  
  batchResults: null,
  isBatchPredicting: false,
  batchError: null,

  runPredict: async (req) => {
    set({ isPredicting: true, predictionError: null });
    try {
      const res = await predictSequence(req);
      const updatedHistory = [res, ...get().history.filter(h => h.protein_id !== res.protein_id)].slice(0, 50);
      
      set({
        activePrediction: res,
        history: updatedHistory,
        isPredicting: false
      });
      saveHistory(updatedHistory);
      return res;
    } catch (err: any) {
      set({ predictionError: err?.message || 'Prediction failed', isPredicting: false });
      throw err;
    }
  },

  runBatchPredict: async (req) => {
    set({ isBatchPredicting: true, batchError: null, batchResults: null });
    try {
      const res = await predictBatch(req);
      set({
        batchResults: res,
        isBatchPredicting: false
      });
      
      // Optionally add batch items to history
      const updatedHistory = [...res.results, ...get().history]
        .filter((v, i, a) => a.findIndex(t => t.protein_id === v.protein_id) === i)
        .slice(0, 50);
      set({ history: updatedHistory });
      saveHistory(updatedHistory);
      
      return res;
    } catch (err: any) {
      set({ batchError: err?.message || 'Batch prediction failed', isBatchPredicting: false });
      throw err;
    }
  },

  setActivePrediction: (prediction) => set({ activePrediction: prediction }),
  
  clearHistory: () => {
    set({ history: [] });
    try {
      localStorage.removeItem('protintel_history');
    } catch {}
  },
  
  deleteHistoryItem: (proteinId) => {
    const updatedHistory = get().history.filter(h => h.protein_id !== proteinId);
    set({ history: updatedHistory });
    saveHistory(updatedHistory);
    if (get().activePrediction?.protein_id === proteinId) {
      set({ activePrediction: null });
    }
  }
}));
