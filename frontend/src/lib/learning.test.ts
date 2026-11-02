import { afterEach, describe, expect, it, vi } from 'vitest'
import { configurationSignature, defaultLearning, learningKey, loadLearning, validLearning } from './learning'

afterEach(() => vi.unstubAllGlobals())

describe('integridade do rascunho e resultados salvos', () => {
  it('não sinaliza mudança por diferença de ordem das propriedades recebidas da API', () => {
    const draft = { model: { hidden_layers: [32, 16], activation: 'relu' }, threshold: 0.5 }
    const saved = { threshold: 0.5, model: { activation: 'relu', hidden_layers: [32, 16] } }
    expect(configurationSignature(draft)).toBe(configurationSignature(saved))
    expect(configurationSignature(draft)).not.toBe(configurationSignature({
      ...saved, model: { ...saved.model, hidden_layers: [16, 32] },
    }))
  })

  it('rejeita rascunho persistido incompatível em vez de apresentar parâmetros inválidos', () => {
    vi.stubGlobal('localStorage', { getItem: () => '{"model":null,"training":{}}' })
    expect(loadLearning()).toEqual(defaultLearning)
    const loaded = loadLearning()
    loaded.model.hidden_layers[0] = 256
    expect(defaultLearning.model.hidden_layers[0]).toBe(32)
  })

  it('preserva arquitetura, seed e regularização ao recuperar um rascunho válido', () => {
    const saved = structuredClone(defaultLearning)
    saved.model.hidden_layers = [64, 8]
    saved.model.l2 = 0.01
    saved.training.seed = 123
    vi.stubGlobal('localStorage', { getItem: (key: string) => key === learningKey ? JSON.stringify(saved) : null })
    expect(loadLearning()).toEqual(saved)
  })

  it('impede early stopping inconsistente e valores não finitos antes do envio', () => {
    const draft = structuredClone(defaultLearning)
    draft.training.epochs = 2
    expect(validLearning(draft)).toBe(false)
    draft.training.early_stopping = false
    expect(validLearning(draft)).toBe(true)
    draft.model.learning_rate = Infinity
    expect(validLearning(draft)).toBe(false)
  })

  it('funciona quando o armazenamento está indisponível ou contém JSON quebrado', () => {
    vi.stubGlobal('localStorage', { getItem: () => { throw new Error('indisponível') } })
    expect(loadLearning()).toEqual(defaultLearning)
    vi.stubGlobal('localStorage', { getItem: () => '{quebrado' })
    expect(loadLearning()).toEqual(defaultLearning)
  })
})
