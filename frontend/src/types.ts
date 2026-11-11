export interface Profile {
  rows: number
  columns: number
  column_names: string[]
  target_distribution: Record<string, number>
  missing_by_column: Record<string, { count: number; percent: number }>
  environment_distribution: { category: string; count: number }[]
  numeric_columns: string[]
  categorical_columns: string[]
  id_column: string
  label_column: string
  numeric_summary: Record<string, {
    n: number; min: number | null; max: number | null; mean: number | null
    median: number | null; std: number | null; negative_count: number
  }>
  duplicate_ids: number
  sha256: string
  scope: string
}

export interface Preview {
  columns: string[]
  rows: Record<string, string | number | null>[]
  total: number
  offset: number
  limit: number
}

export interface Boxplot {
  min: number; max: number; q1: number; median: number; q3: number
  lower_whisker: number; upper_whisker: number
  lower_fence: number; upper_fence: number
  outlier_count: number; outlier_points: number[]
}

export interface Eda {
  column: string
  kind: 'numeric' | 'categorical'
  total: number
  missing: number
  histogram?: { bin_start: number; bin_end: number; count: number }[]
  boxplot?: Boxplot | null
  categories?: { category: string; count: number }[]
}

export interface Correlations {
  columns: string[]
  matrix: (number | null)[][]
  sample_counts: number[][]
  method: string
}

export interface PreprocessingSettings {
  missing_strategy: 'drop' | 'mean' | 'median' | 'knn'
  scaler: 'standard' | 'minmax' | 'none'
  split_strategy: '70_15_15' | '80_20' | 'custom'
  train_percent?: number
  val_percent?: number
  test_percent?: number
  seed: number
  knn_neighbors: 3 | 5 | 7
  categorical_strategy: 'most_frequent' | 'constant'
}

export interface PreparationStats {
  observed: number
  missing: number
  mean: number | null
  median: number | null
  std: number | null
  min: number | null
  max: number | null
}

export interface PreprocessingPreview {
  config: PreprocessingSettings & { column: string; subset: 'train' | 'val' }
  dataset_rows: number
  rows_after_filter: number
  removed_rows: number
  coverage_percent: number
  cohort_by_class: Record<string, { original: number; retained: number }>
  split_counts: Record<'train' | 'val' | 'test', number>
  split_classes: Record<'train' | 'val' | 'test', Record<string, number>>
  partition_fingerprints: Record<'train' | 'val' | 'test', string>
  dataset_sha256: string
  sklearn_version: string
  fitted_on: string
  preview_subset: 'train' | 'val'
  test_transformed: boolean
  output_features: string[]
  output_feature_count: number
  numeric_before_after: {
    column: string
    samples: { sample_id: string; before: number | null; imputed: number; scaled: number }[]
    before_stats: PreparationStats
    imputed_stats: PreparationStats
    scaled_stats: PreparationStats
    fitted_replacement: number | null
    before_histogram: NonNullable<Eda['histogram']>
    imputed_histogram: NonNullable<Eda['histogram']>
    scaled_histogram: NonNullable<Eda['histogram']>
  }[]
  missing_before_after: Record<string, { before: number; after: number }>
  environment: {
    strategy: string
    categories: string[]
    fill_value: string
    before_missing: number
    after_missing: number
    unknown_handling: string
    encoded_counts: Record<string, number>
  }
  warnings: string[]
  python_code: string
}

export interface LearningDraft {
  name: string
  model: {
    hidden_layers: number[]
    activation: 'relu' | 'tanh'
    optimizer: 'adam' | 'sgd' | 'rmsprop'
    learning_rate: number
    l2: number
  }
  training: { epochs: number; batch_size: number; early_stopping: boolean; patience: number; seed: number }
  threshold: number
}

export interface ModelDescription {
  input_features: number
  layers: { name: string; inputs: number; units: number; activation: string; parameters: number }[]
  total_parameters: number
  output_features?: string[]
  python_code?: string
}

export interface ClassificationMetrics {
  classification_report?: Record<'0' | '1' | 'macro avg' | 'weighted avg', {
    precision: number; recall: number; 'f1-score': number; support: number
  }>
  loss?: number
  predictive_loss: number
  accuracy: number
  precision: number
  recall: number
  specificity: number
  f1: number
  confusion_matrix: number[][]
  tn: number; fp: number; fn: number; tp: number
  samples: number
  threshold: number
  undefined_metrics: string[]
}

export interface Experiment {
  id: string
  config: LearningDraft & { preprocess: PreprocessingSettings }
  status: 'queued' | 'running' | 'completed' | 'failed' | 'interrupted'
  created_at: string
  started_at: string | null
  ended_at: string | null
  completed_epochs: number
  history: { epoch: number; loss: number; val_loss: number; accuracy: number; val_accuracy: number; predictive_loss: number; val_predictive_loss: number }[]
  best_epoch: number | null
  saved_epoch: number | null
  validation_metrics: ClassificationMetrics | null
  train_metrics?: ClassificationMetrics | null
  test_metrics: ClassificationMetrics | null
  dataset_sha256: string
  partition_key: string
  split_counts: Record<string, number>
  versions: Record<string, string>
  model_summary: ModelDescription | null
  output_features: string[]
  error: string | null
  duration_seconds?: number
}

export interface Study {
  status: 'open' | 'evaluating' | 'evaluation_failed' | 'completed'
  partition_key: string | null
  preprocess: PreprocessingSettings | null
  selected_experiment_id: string | null
  result: (ClassificationMetrics & { experiment_id: string; evaluated_at: string }) | null
}
