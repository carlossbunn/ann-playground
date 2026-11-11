import { useState, type FormEvent } from 'react'
import { useQuery } from '@tanstack/react-query'
import type { PreprocessingPreview, PreprocessingSettings, Study } from '../types'
import { get, number, post } from '../lib/api'
import { splitLabel, splitPercentages, validPercentages } from '../lib/preprocessing'
import { ErrorState, Loading } from './UI'

function PercentageForm({ settings, onChange, locked }: {
  settings: PreprocessingSettings; onChange: (settings: PreprocessingSettings) => void; locked: boolean
}) {
  const [values, setValues] = useState(() => splitPercentages(settings).map(String))
  const parsed = values.map(Number)
  const valid = values.every((value) => value.trim()) && validPercentages(parsed)
  const changed = parsed.some((value, index) => value !== splitPercentages(settings)[index])
  function apply(event: FormEvent) {
    event.preventDefault()
    if (valid && !locked) onChange({ ...settings, split_strategy: 'custom',
      train_percent: parsed[0], val_percent: parsed[1], test_percent: parsed[2] })
  }
  return <form onSubmit={apply}>
    <fieldset disabled={locked} className="split-fields"><legend className="sr-only">Percentuais dos conjuntos</legend>
      {['Treino (%)', 'Validação (%)', 'Teste (%)'].map((label, index) => <div className="field" key={label}>
        <label htmlFor={`percent-${index}`}>{label}</label>
        <input id={`percent-${index}`} type="number" min="0.01" max="99.99" step="any" value={values[index]} aria-describedby="percent-feedback"
          onChange={(event) => setValues(values.map((value, i) => i === index ? event.target.value : value))} />
      </div>)}
      <button className="button secondary" disabled={!valid || !changed}>Aplicar percentuais</button>
    </fieldset>
    <p id="percent-feedback" role="status">{!valid ? 'Informe três percentuais positivos que somem 100%.' : changed ? 'Clique em Aplicar percentuais para usar esta divisão.' : `Divisão aplicada: ${splitLabel(settings)}%.`}</p>
  </form>
}

export default function SplitControls({ settings, onChange }: {
  settings: PreprocessingSettings; onChange: (settings: PreprocessingSettings) => void
}) {
  const study = useQuery({ queryKey: ['study'], queryFn: ({ signal }) => get<Study>('/study', signal), refetchInterval: 2500 })
  const locked = !study.isSuccess || Boolean(study.data.partition_key)
  const reserved = study.data?.preprocess
  const mismatch = reserved && (splitLabel(reserved) !== splitLabel(settings) || reserved.seed !== settings.seed)
  const preview = useQuery({ queryKey: ['split-preview', settings], queryFn: ({ signal }) => post<PreprocessingPreview>('/preprocessing/preview', settings, signal) })
  return <section className="panel split-settings">
    <div className="panel-heading"><div><h2>Divisão de treino, validação e teste</h2><p>Escolha a porcentagem da coorte para cada conjunto. A soma deve ser 100%.</p></div></div>
    {study.isError && <ErrorState error={study.error} retry={() => void study.refetch()} />}
    <PercentageForm key={`${splitLabel(settings)}-${settings.seed}`} settings={settings} onChange={onChange} locked={locked} />
    {reserved && <p>Divisão reservada no primeiro experimento: <strong>{splitLabel(reserved)}%</strong> · seed {reserved.seed}.</p>}
    {mismatch && <button className="button secondary" onClick={() => onChange({ ...settings, split_strategy: reserved.split_strategy, train_percent: reserved.train_percent, val_percent: reserved.val_percent, test_percent: reserved.test_percent, seed: reserved.seed })}>Restaurar divisão reservada</button>}
    {preview.isPending ? <Loading /> : preview.isError ? <ErrorState error={preview.error} retry={() => void preview.refetch()} /> : <>
      <div className="split-counts" aria-label="Amostras efetivas por conjunto">{(['train', 'val', 'test'] as const).map((key, i) => <div key={key}><span>{['Treino', 'Validação', 'Teste reservado'][i]}</span><strong>{number(preview.data.split_counts[key], 0)}</strong><small>amostras · {number(preview.data.split_counts[key] / preview.data.rows_after_filter * 100)}%</small></div>)}</div>
      <p className="caption">Divisão estratificada · seed {settings.seed} · coorte de {number(preview.data.rows_after_filter, 0)} amostras. Contagens podem sofrer arredondamento. Imputação e escala são ajustadas apenas no treino.</p>
    </>}
  </section>
}
