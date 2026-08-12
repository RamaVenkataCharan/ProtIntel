/**
 * ProtIntel — Verified CB513 Benchmark Data
 *
 * ALL values sourced exclusively from actual evaluation runs:
 *   - logs/evaluation/cb513_results.json
 *   - logs/evaluation/final_cb513_metrics.json
 *
 * DO NOT add estimated, target, or aspirational values here.
 */

export interface BenchmarkConfiguration {
  label: string;
  description: string;
  /** Source file and field for traceability */
  source: string;
  q3Accuracy: number;
  q8Accuracy: number;
  q3Mcc: number;
  q3F1Macro: number;
  q8F1Macro: number;
}

/**
 * Verified benchmark configurations from CB513 evaluation artifacts.
 *
 * "Baseline" = final_cb513_metrics.json baseline_* fields
 * "Current Model" = cb513_results.json top-level fields (the actual run output)
 */
export const BENCHMARK_CONFIGS: BenchmarkConfiguration[] = [
  {
    label: 'CB513 Baseline',
    description: 'Baseline reference configuration',
    source: 'logs/evaluation/final_cb513_metrics.json → baseline_q3, baseline_q8, baseline_mcc',
    q3Accuracy: 0.6994,
    q8Accuracy: 0.4428,
    q3Mcc: 0.527,
    q3F1Macro: 0.6994, // baseline_q3 used as proxy (no separate F1 in baseline)
    q8F1Macro: 0.4428, // baseline_q8 used as proxy
  },
  {
    label: 'Current Model',
    description: 'Latest evaluated model on CB513 test set (514 proteins)',
    source: 'logs/evaluation/cb513_results.json → q3_accuracy, q8_accuracy, q3_mcc',
    q3Accuracy: 0.6943,
    q8Accuracy: 0.3406,
    q3Mcc: 0.5269,
    q3F1Macro: 0.6820, // cb513_results.json → q3_f1_macro
    q8F1Macro: 0.2654, // cb513_results.json → q8_f1_macro
  },
];

/** Metric definitions for the 3D chart axes */
export const BENCHMARK_METRICS = [
  { key: 'q3Accuracy' as const, label: 'Q3 Acc', color: '#00E5CC' },
  { key: 'q8Accuracy' as const, label: 'Q8 Acc', color: '#F59E0B' },
  { key: 'q3Mcc' as const, label: 'Q3 MCC', color: '#8B5CF6' },
] as const;

export type BenchmarkMetricKey = (typeof BENCHMARK_METRICS)[number]['key'];
