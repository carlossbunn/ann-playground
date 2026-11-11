import type { PreprocessingSettings } from '../types'

export const defaultSettings: PreprocessingSettings = {
  missing_strategy: 'median',
  scaler: 'standard',
  split_strategy: '70_15_15',
  train_percent: 70,
  val_percent: 15,
  test_percent: 15,
  seed: 42,
  knn_neighbors: 5,
  categorical_strategy: 'most_frequent',
}

export const missingNames = { drop: 'Remover', mean: 'Média', median: 'Mediana', knn: 'KNN' }
export const scalerNames = { standard: 'StandardScaler', minmax: 'MinMaxScaler', none: 'Sem escala' }
export const settingsKey = 'ann-lab-preprocessing-v1'

export function loadSettings(): PreprocessingSettings {
  try {
    const saved = JSON.parse(localStorage.getItem(settingsKey) ?? 'null')
    if (saved && ['drop', 'mean', 'median', 'knn'].includes(saved.missing_strategy)
      && ['standard', 'minmax', 'none'].includes(saved.scaler)
      && ['70_15_15', '80_20', 'custom'].includes(saved.split_strategy)
      && (saved.split_strategy !== 'custom' || validPercentages([saved.train_percent, saved.val_percent, saved.test_percent]))
      && Number.isInteger(saved.seed) && saved.seed >= 0 && saved.seed <= 4294967295
      && [3, 5, 7].includes(saved.knn_neighbors)
      && ['most_frequent', 'constant'].includes(saved.categorical_strategy)) {
      return Object.fromEntries(Object.entries(defaultSettings).map(([key, fallback]) => [key, saved[key] ?? fallback])) as unknown as PreprocessingSettings
    }
  } catch { /* Armazenamento indisponível: usar os padrões em memória. */ }
  return { ...defaultSettings }
}

export function validPercentages(values: number[]) {
  return values.every((value) => Number.isFinite(value) && value > 0 && value < 100)
    && Math.abs(values.reduce((sum, value) => sum + value, 0) - 100) < 1e-8
}

export function splitPercentages(settings: PreprocessingSettings): number[] {
  return settings.split_strategy === 'custom'
    ? [settings.train_percent ?? 70, settings.val_percent ?? 15, settings.test_percent ?? 15]
    : settings.split_strategy === '70_15_15' ? [70, 15, 15] : [64, 16, 20]
}

export function splitLabel(settings: PreprocessingSettings) {
  return splitPercentages(settings).join('/')
}
