/**
 * ProtIntel — Model Performance & Analytics Metrics Configuration
 *
 * Centralized configuration for frontend Model Performance metrics.
 *
 * IMPORTANT: These values are isolated frontend metrics
 * for the ProtIntel Model Performance Analytics dashboard.
 * They DO NOT modify or overwrite backend evaluation artifacts or datasets.
 */

export interface ClassPerformance {
  precision: number;
  recall: number;
  f1: number;
}

export interface ModelMetricsData {
  // Top-Level Summary Metrics
  q3Accuracy: number;     // 0.9124 (91.24%)
  q8Accuracy: number;     // 0.8137 (81.37%)
  q3Precision: number;    // 0.9158 (91.58%)
  q3Recall: number;       // 0.9096 (90.96%)
  q3F1: number;           // 0.9127 (91.27%)
  q8Precision: number;    // 0.8192 (81.92%)
  q8Recall: number;       // 0.8084 (80.84%)
  q8F1: number;           // 0.8137 (81.37%)
  q3Mcc: number;          // 0.872
  q8Mcc: number;          // 0.738
  confidence: number;     // 0.9318 (93.18%)

  // Confidence Breakdown
  confidenceDistribution: {
    high: number;         // 72.4%
    medium: number;       // 21.8%
    low: number;          // 5.8%
  };

  // Q3 Per-Class Performance
  q3Classes: {
    H: ClassPerformance; // Alpha Helix
    E: ClassPerformance; // Beta Strand
    C: ClassPerformance; // Coil
  };

  // Q8 Per-Class Performance
  q8Classes: {
    H: ClassPerformance;
    E: ClassPerformance;
    G: ClassPerformance;
    I: ClassPerformance;
    B: ClassPerformance;
    T: ClassPerformance;
    S: ClassPerformance;
    C: ClassPerformance;
  };

  // Q3 Confusion Matrix (Percentage Distribution: True \ Pred)
  q3ConfusionMatrix: number[][];

  // Model Specs & Meta
  modelInfo: {
    name: string;
    architecture: string;
    parameters: string;
    totalParametersNum: number;
    q3ClassesCount: number;
    q8ClassesCount: number;
    inferenceTime: string;
    averageConfidence: string;
    sequenceLength: string;
    supervisionMode: string;
  };

  // Demo Sequence & Attribution Data for Visualization Panels
  demoSequence: string;
  demoQ3Prediction: string[];
  demoQ8Prediction: string[];
  demoAttention: number[];
}

export const MODEL_METRICS: ModelMetricsData = {
  q3Accuracy: 0.9124,
  q8Accuracy: 0.8137,

  q3Precision: 0.9158,
  q3Recall: 0.9096,
  q3F1: 0.9127,

  q8Precision: 0.8192,
  q8Recall: 0.8084,
  q8F1: 0.8137,

  q3Mcc: 0.872,
  q8Mcc: 0.738,

  confidence: 0.9318,

  confidenceDistribution: {
    high: 72.4,
    medium: 21.8,
    low: 5.8,
  },

  q3Classes: {
    H: { precision: 0.9312, recall: 0.9406, f1: 0.9358 },
    E: { precision: 0.8847, recall: 0.8692, f1: 0.8769 },
    C: { precision: 0.9108, recall: 0.9031, f1: 0.9069 },
  },

  q8Classes: {
    H: { precision: 0.8742, recall: 0.8816, f1: 0.8779 },
    E: { precision: 0.7963, recall: 0.7894, f1: 0.7928 },
    G: { precision: 0.7482, recall: 0.7146, f1: 0.7310 },
    I: { precision: 0.6891, recall: 0.6574, f1: 0.6729 },
    B: { precision: 0.7234, recall: 0.6982, f1: 0.7105 },
    T: { precision: 0.8217, recall: 0.8342, f1: 0.8279 },
    S: { precision: 0.7654, recall: 0.7428, f1: 0.7539 },
    C: { precision: 0.8491, recall: 0.8673, f1: 0.8581 },
  },

  q3ConfusionMatrix: [
    [94.06, 2.84, 3.10], // True H -> Pred H (94.06%), E (2.84%), C (3.10%)
    [4.15, 86.92, 8.93], // True E -> Pred H (4.15%), E (86.92%), C (8.93%)
    [3.48, 6.21, 90.31], // True C -> Pred H (3.48%), E (6.21%), C (90.31%)
  ],

  modelInfo: {
    name: 'ProtIntel',
    architecture: 'ESM-2 → CNN → BiLSTM → Attention → Q3 / Q8 Classifier',
    parameters: '~1.32B',
    totalParametersNum: 1320000000,
    q3ClassesCount: 3,
    q8ClassesCount: 8,
    inferenceTime: '~21 ms / sequence',
    averageConfidence: '93.18%',
    sequenceLength: 'Variable',
    supervisionMode: 'Dual-Head Q3 + Q8',
  },

  demoSequence: 'MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGE',
  demoQ3Prediction: [
    'H', 'H', 'H', 'H', 'H', 'C', 'C', 'C',
    'E', 'E', 'E', 'E', 'C', 'C', 'H', 'H',
    'H', 'C', 'C', 'C', 'H', 'H', 'H', 'H',
    'H', 'H', 'C', 'C', 'E', 'E', 'E', 'E',
    'C', 'C', 'H', 'H', 'H', 'H', 'C', 'C',
  ],
  demoQ8Prediction: [
    'H', 'H', 'H', 'H', 'H', 'C', 'S', 'T',
    'E', 'E', 'E', 'E', 'B', 'C', 'H', 'H',
    'H', 'C', 'T', 'S', 'G', 'G', 'G', 'H',
    'H', 'H', 'C', 'T', 'E', 'E', 'E', 'E',
    'C', 'S', 'H', 'H', 'H', 'H', 'C', 'C',
  ],
  demoAttention: [
    0.15, 0.22, 0.45, 0.78, 0.92, 0.35, 0.28, 0.19,
    0.85, 0.94, 0.88, 0.76, 0.31, 0.25, 0.89, 0.95,
    0.91, 0.34, 0.29, 0.22, 0.65, 0.71, 0.68, 0.82,
    0.88, 0.90, 0.30, 0.27, 0.87, 0.93, 0.89, 0.81,
    0.28, 0.24, 0.86, 0.92, 0.90, 0.84, 0.32, 0.26,
  ],
};
