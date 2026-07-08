// Client API helpers for ProtIntel backend

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  device: string;
}

export interface ModelInfoResponse {
  model_name: string;
  version: string;
  architecture: string;
  esm2_model: string;
  total_parameters: number;
  trainable_parameters: number;
  q3_classes: string[];
  q8_classes: string[];
}

export interface MetricsResponse {
  dataset: string;
  q3_accuracy: number | null;
  q8_accuracy: number | null;
  q3_mcc: number | null;
  per_class_q3: Record<string, number> | null;
  per_class_q8: Record<string, number> | null;
}

export interface PredictRequest {
  sequence: string;
  return_attention?: boolean;
  return_xai?: boolean;
  xai_method?: 'ig' | 'shap' | 'rollout';
}

export interface PredictResponse {
  protein_id: string;
  sequence: string;
  length: number;
  q3_prediction: string[];
  q8_prediction: string[];
  q3_probabilities: number[][];
  q8_probabilities: number[][];
  confidence: number[];
  attention_map?: number[][];
  residue_importance?: number[];
  xai_method?: string;
  processing_time_ms: number;
}

export interface BatchPredictRequest {
  sequences: string[];
  return_attention?: boolean;
  return_xai?: boolean;
  xai_method?: 'ig' | 'shap' | 'rollout';
}

export interface BatchPredictResponse {
  results: PredictResponse[];
  total_sequences: number;
  total_processing_time_ms: number;
}

// Proxied via Vite config to http://localhost:8000
const API_BASE = '/api';

export async function fetchHealth(): Promise<HealthResponse> {
  const resp = await fetch(`${API_BASE}/health`);
  if (!resp.ok) throw new Error('Failed to fetch health status');
  return resp.json();
}

export async function fetchModelInfo(): Promise<ModelInfoResponse> {
  const resp = await fetch(`${API_BASE}/model_info`);
  if (!resp.ok) throw new Error('Failed to fetch model info');
  return resp.json();
}

export async function fetchMetrics(): Promise<MetricsResponse> {
  const resp = await fetch(`${API_BASE}/metrics`);
  if (!resp.ok) throw new Error('Failed to fetch evaluation metrics');
  return resp.json();
}

export async function predictSequence(req: PredictRequest): Promise<PredictResponse> {
  const resp = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!resp.ok) {
    const errorData = await resp.json().catch(() => ({}));
    const message = errorData?.detail?.[0]?.msg || errorData?.detail || 'Inference failed';
    throw new Error(message);
  }
  return resp.json();
}

export async function predictBatch(req: BatchPredictRequest): Promise<BatchPredictResponse> {
  const resp = await fetch(`${API_BASE}/predict_batch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!resp.ok) {
    const errorData = await resp.json().catch(() => ({}));
    const message = errorData?.detail?.[0]?.msg || errorData?.detail || 'Batch inference failed';
    throw new Error(message);
  }
  return resp.json();
}
