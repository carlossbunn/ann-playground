import type { PreprocessingSettings } from '../types'

export const defaultSettings: PreprocessingSettings = {
  missing_strategy: 'median',
  scaler: 'standard',
  split_strategy: '70_15_15',
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
      && ['70_15_15', '80_20'].includes(saved.split_strategy)
      && Number.isInteger(saved.seed) && saved.seed >= 0 && saved.seed <= 4294967295
      && [3, 5, 7].includes(saved.knn_neighbors)
      && ['most_frequent', 'constant'].includes(saved.categorical_strategy)) {
      return Object.fromEntries(Object.keys(defaultSettings).map((key) => [key, saved[key]])) as unknown as PreprocessingSettings
    }
  } catch { /* Armazenamento indisponível: usar os padrões em memória. */ }
  return { ...defaultSettings }
}
