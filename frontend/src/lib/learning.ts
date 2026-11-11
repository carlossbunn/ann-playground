import type { LearningDraft } from '../types'

export const learningKey = 'ann-lab-learning-v1'
export const activeRunKey = 'ann-lab-active-run-v1'
export const defaultLearning: LearningDraft = {
  name: 'Minha MLP',
  model: { hidden_layers: [32, 16], activation: 'relu', optimizer: 'adam', learning_rate: 0.001, l2: 0 },
  training: { epochs: 50, batch_size: 32, early_stopping: true, patience: 8, seed: 42 },
  threshold: 0.5,
}

export const presetLearning: LearningDraft = {
  name: 'Modelo pronto · StandardScaler',
  model: { hidden_layers: [32, 16], activation: 'relu', optimizer: 'adam', learning_rate: 0.001, l2: 0.0001 },
  training: { epochs: 100, batch_size: 32, early_stopping: true, patience: 10, seed: 42 },
  threshold: 0.5,
}

export function validLearning(draft: LearningDraft): boolean {
  try {
    const { model, training } = draft
    return typeof draft.name === 'string' && draft.name.trim().length > 0 && draft.name.length <= 100
      && Array.isArray(model.hidden_layers) && model.hidden_layers.length >= 1 && model.hidden_layers.length <= 4
      && model.hidden_layers.every((n) => Number.isInteger(n) && n >= 4 && n <= 256)
      && ['relu', 'tanh'].includes(model.activation) && ['adam', 'sgd', 'rmsprop'].includes(model.optimizer)
      && model.learning_rate > 0 && model.learning_rate <= 0.1 && model.l2 >= 0 && model.l2 <= 0.1
      && Number.isInteger(training.epochs) && training.epochs >= 1 && training.epochs <= 200
      && Number.isInteger(training.batch_size) && training.batch_size >= 8 && training.batch_size <= 256
      && Number.isInteger(training.seed) && training.seed >= 0 && training.seed <= 2147483647
      && typeof training.early_stopping === 'boolean' && Number.isInteger(training.patience)
      && training.patience >= 1 && training.patience <= 199
      && (!training.early_stopping || training.patience < training.epochs)
      && draft.threshold > 0 && draft.threshold < 1
  } catch { return false }
}

export function loadLearning(): LearningDraft {
  try {
    const saved = JSON.parse(localStorage.getItem(learningKey) ?? 'null')
    if (validLearning(saved)) return saved
  } catch { /* Usar padrões se o armazenamento não estiver disponível. */ }
  return structuredClone(defaultLearning)
}

export function loadActiveRun(): string {
  try { return localStorage.getItem(activeRunKey) ?? '' }
  catch { return '' }
}

export const statusNames = {
  queued: 'Na fila', running: 'Treinando', completed: 'Concluído', failed: 'Falhou', interrupted: 'Interrompido',
}

export function configurationSignature(value: unknown): string {
  return JSON.stringify(value, (_key, entry) => {
    if (entry && typeof entry === 'object' && !Array.isArray(entry)) {
      return Object.fromEntries(Object.keys(entry).sort().map((key) => [key, entry[key]]))
    }
    return entry
  })
}
