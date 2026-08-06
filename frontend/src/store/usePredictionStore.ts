import { create } from 'zustand';
import { predictSequence, predictBatch, fetchJobStatus, type PredictRequest, type PredictResponse, type BatchPredictRequest, type BatchPredictResponse } from '../utils/api';

interface PredictionStoreState {
  history: PredictResponse[];
  activePrediction: PredictResponse | null;
  isPredicting: boolean;
  predictionError: string | null;
  jobStatus: 'pending' | 'processing' | 'completed' | 'failed' | null;
  
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

let activePollId = 0;

export const usePredictionStore = create<PredictionStoreState>((set, get) => ({
  history: loadHistory(),
  activePrediction: null,
  isPredicting: false,
  predictionError: null,
  jobStatus: null,
  
  batchResults: null,
  isBatchPredicting: false,
  batchError: null,

  runPredict: async (req) => {
    // Increment poll ID to abort any previous polling loop
    const currentPollId = ++activePollId;
    set({ isPredicting: true, predictionError: null, jobStatus: null });

    try {
      const res = await predictSequence(req);
      
      if (currentPollId !== activePollId) return res as any;

      // Check if this is an asynchronous job status response
      if ('job_id' in res) {
        const jobId = res.job_id;
        set({ jobStatus: res.status });
        
        // Start polling for results
        const maxAttempts = 120; // 60 seconds timeout at 500ms intervals
        let attempts = 0;
        
        while (attempts < maxAttempts) {
          await new Promise(resolve => setTimeout(resolve, 500));
          if (currentPollId !== activePollId) {
            return res as any; // Aborted by newer request
          }

          const jobState = await fetchJobStatus(jobId);
          if (currentPollId !== activePollId) return res as any;

          set({ jobStatus: jobState.status });
          
          if (jobState.status === 'completed' && jobState.result) {
            const finalResult = jobState.result;
            const updatedHistory = [finalResult, ...get().history.filter(h => h.protein_id !== finalResult.protein_id)].slice(0, 50);
            
            set({
              activePrediction: finalResult,
              history: updatedHistory,
              isPredicting: false,
              jobStatus: 'completed'
            });
            saveHistory(updatedHistory);
            return finalResult;
          } else if (jobState.status === 'failed') {
            const errMsg = jobState.error || 'Asynchronous prediction job failed';
            set({ predictionError: errMsg, isPredicting: false, jobStatus: 'failed' });
            throw new Error(errMsg);
          }
          
          attempts++;
        }
        
        const timeoutMsg = 'Attribution calculation timed out. Please try again.';
        set({ predictionError: timeoutMsg, isPredicting: false, jobStatus: 'failed' });
        throw new Error(timeoutMsg);
      } else {
        // Sync response
        const updatedHistory = [res, ...get().history.filter(h => h.protein_id !== res.protein_id)].slice(0, 50);
        
        set({
          activePrediction: res,
          history: updatedHistory,
          isPredicting: false,
          jobStatus: 'completed'
        });
        saveHistory(updatedHistory);
        return res;
      }
    } catch (err: any) {
      if (currentPollId === activePollId) {
        set({ predictionError: err?.message || 'Prediction failed', isPredicting: false });
      }
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
