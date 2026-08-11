/**
 * ProtIntel — Model Performance Analytics Support Data
 *
 * All primary model metrics originate from src/config/modelMetrics.ts.
 */

import { MODEL_METRICS } from '../config/modelMetrics';

export const DEMO_METRICS = {
  q3Accuracy: MODEL_METRICS.q3Accuracy,
  q8Accuracy: MODEL_METRICS.q8Accuracy,
  q3Mcc: MODEL_METRICS.q3Mcc,
  q3Precision: MODEL_METRICS.q3Precision,
  q3Recall: MODEL_METRICS.q3Recall,
  q3F1: MODEL_METRICS.q3F1,
  q8Precision: MODEL_METRICS.q8Precision,
  q8Recall: MODEL_METRICS.q8Recall,
  q8F1: MODEL_METRICS.q8F1,
  confidence: MODEL_METRICS.confidence,
  q3Classes: {
    H: MODEL_METRICS.q3Classes.H.f1,
    E: MODEL_METRICS.q3Classes.E.f1,
    C: MODEL_METRICS.q3Classes.C.f1,
  },
  q8Classes: {
    H: MODEL_METRICS.q8Classes.H.f1,
    E: MODEL_METRICS.q8Classes.E.f1,
    G: MODEL_METRICS.q8Classes.G.f1,
    I: MODEL_METRICS.q8Classes.I.f1,
    B: MODEL_METRICS.q8Classes.B.f1,
    T: MODEL_METRICS.q8Classes.T.f1,
    S: MODEL_METRICS.q8Classes.S.f1,
    C: MODEL_METRICS.q8Classes.C.f1,
  },
  q3ConfusionMatrix: MODEL_METRICS.q3ConfusionMatrix,
  demoSequence: MODEL_METRICS.demoSequence,
  demoQ3Prediction: MODEL_METRICS.demoQ3Prediction,
  demoQ8Prediction: MODEL_METRICS.demoQ8Prediction,
  demoAttention: MODEL_METRICS.demoAttention,
  verifiedModel: {
    q3Accuracy: MODEL_METRICS.q3Accuracy,
    q8Accuracy: MODEL_METRICS.q8Accuracy,
    q3Mcc: MODEL_METRICS.q3Mcc,
  },
};
