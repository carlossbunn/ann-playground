import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Activity, ArrowRight, Cpu } from 'lucide-react'
import type { Experiment } from '../types'
import { get, number, post } from '../lib/api'
import { ErrorState, Loading, Notice } from '../components/UI'
import './learning.css'

type Scenario = { numeric: Record<string, number | null>; environment: string | null }
type Sample = Scenario & { sample_id: string; subset: string; offset: number; total: number }
type Prediction = {
  threshold: number; scaler: string; missing_strategy: string
  scenarios: { score: number; predicted_class: number; numeric: {
    name: string; original: number | null; imputed: number; scaled: number; mean?: number; scale?: number
  }[] }[]
}

function ExperimentPanel({ run, navigate }: { run: Experiment; navigate: (step: number) => void }) {
  const [subset, setSubset] = useState<'train' | 'val'>('train')
  const [offset, setOffset] = useState(0)
  const sample = useQuery({
    queryKey: ['use-model-sample', run.id, subset, offset],
    queryFn: ({ signal }) => get<Sample>(`/experiments/${run.id}/sample?subset=${subset}&offset=${offset}`, signal),
  })
  return <>
    <section className="panel"><h2>Do treinamento ao uso</h2><p>Modelo <strong>{run.config.name}</strong>: {run.config.model.hidden_layers.join(' → ')} neurônios ocultos, {run.config.model.activation}, saída sigmoid. Foi treinado com {run.config.model.optimizer}, taxa {run.config.model.learning_rate}, lote {run.config.training.batch_size}, {run.completed_epochs} épocas e L2 {run.config.model.l2}.</p>
      <p>Durante o treinamento, a rede ajustou pesos para reduzir a perda. Agora ela usa esses pesos fixos para calcular uma resposta para cada leitura; modificar as entradas abaixo não muda o modelo.</p>
      <details><summary>Para que servem os parâmetros?</summary><ul>
        <li><strong>Camadas e neurônios:</strong> definem a capacidade de combinar padrões dos atributos.</li>
        <li><strong>Learning rate e otimizador:</strong> controlam os ajustes dos pesos durante o treino; não são velocidade nem ajustes físicos da máquina.</li>
        <li><strong>Épocas e batch:</strong> quantas passagens pelos dados e quantas amostras por atualização de pesos.</li>
        <li><strong>L2 e early stopping:</strong> ajudam a controlar o aprendizado e a avaliar quando parar usando validação.</li>
        <li><strong>Limiar {run.config.threshold}:</strong> transforma o score da classe 1 em decisão 0 ou 1. Isso não altera os pesos.</li>
      </ul></details>
    </section>
    <section className="panel"><div className="panel-heading"><div><h2>Escolha uma leitura de referência</h2><p>Carregue uma amostra real de treino ou validação; depois modifique os campos para comparar.</p></div></div>
      <div className="learning-fields"><div className="field"><label htmlFor="use-subset">Conjunto da amostra</label><select id="use-subset" value={subset} onChange={e => { setSubset(e.target.value as 'train' | 'val'); setOffset(0) }}><option value="train">Treino</option><option value="val">Validação</option></select></div>
        <div className="field"><label htmlFor="use-offset">Posição da amostra</label><input id="use-offset" type="number" min="0" max={sample.data ? sample.data.total - 1 : undefined} value={offset} onChange={e => setOffset(Math.max(0, Number(e.target.value)))} /></div></div>
      {sample.isPending ? <Loading /> : sample.isError ? <ErrorState error={sample.error} retry={() => void sample.refetch()} /> : <p>Amostra {sample.data.sample_id} · posição {sample.data.offset} de {number(sample.data.total, 0)} · sem rótulo de resposta exibido.</p>}
    </section>
    {sample.data && <ScenarioEditor key={`${run.id}-${subset}-${offset}`} run={run} sample={sample.data} />}
    <div className="step-footer"><button className="button secondary" onClick={() => navigate(8)}>Comparar experimentos <ArrowRight size={15} /></button></div>
  </>
}

function ScenarioEditor({ run, sample }: { run: Experiment; sample: Sample }) {
  const [values, setValues] = useState<Record<string, string>>(() => Object.fromEntries(Object.entries(sample.numeric).map(([key, value]) => [key, value === null ? '' : String(value)])))
  const [environment, setEnvironment] = useState(sample.environment ?? '')
  const mutation = useMutation({ mutationFn: (scenarios: Scenario[]) => post<Prediction>(`/experiments/${run.id}/predict`, { scenarios }) })
  const invalid = Object.values(values).some(value => value.trim() !== '' && !Number.isFinite(Number(value)))
  function compare() {
    const edited = Object.fromEntries(Object.entries(values).map(([key, value]) => [key, value.trim() === '' ? null : Number(value)]))
    mutation.mutate([{ numeric: sample.numeric, environment: sample.environment }, { numeric: edited, environment: environment || null }])
  }
  return <>
    <section className="panel"><div className="panel-heading"><div><h2>Compare duas condições</h2><p>O cenário A é a leitura real carregada. Altere o cenário B; campo vazio significa valor ausente.</p></div><Activity size={20} /></div>
      <div className="learning-fields">{Object.entries(values).map(([key, value]) => <div className="field" key={key}><label htmlFor={`sensor-${key}`}>{key}</label><input id={`sensor-${key}`} type="number" step="any" value={value} disabled={mutation.isPending} onChange={e => { mutation.reset(); setValues({ ...values, [key]: e.target.value }) }} /><small>Referência: {sample.numeric[key] === null ? 'ausente' : number(sample.numeric[key], 5)}</small></div>)}
        <div className="field"><label htmlFor="sensor-environment">environment</label><select id="sensor-environment" value={environment} disabled={mutation.isPending} onChange={e => { mutation.reset(); setEnvironment(e.target.value) }}><option value="">Ausente</option><option value="cold">cold</option><option value="mild">mild</option><option value="hot">hot</option></select><small>Referência: {sample.environment ?? 'ausente'}</small></div></div>
      <p className="caption">Na analogia, cada campo seria uma leitura da máquina. Os nomes, unidades e limites físicos reais destes atributos não foram informados.</p>
      <button className="button primary" disabled={invalid || mutation.isPending} onClick={compare}><Cpu size={16} />{mutation.isPending ? 'Calculando…' : 'Comparar com o modelo'}</button>
      {mutation.isError && <ErrorState error={mutation.error} retry={compare} />}
    </section>
    {mutation.data && <section className="panel" aria-live="polite"><h2>O que mudou na resposta?</h2><div className="stats-strip">{mutation.data.scenarios.map((item, index) => <div key={index}><span>Cenário {index === 0 ? 'A · referência' : 'B · alterado'}</span><strong>{number(item.score * 100, 2)}%</strong><span>score da classe 1 · classe prevista {item.predicted_class}</span></div>)}<div><span>Diferença no score</span><strong>{number((mutation.data.scenarios[1].score - mutation.data.scenarios[0].score) * 100, 2)} p.p.</strong></div></div>
      <p>O modelo compara o score ao limiar {mutation.data.threshold}; acima ou igual ao limiar resulta em classe 1. O significado prático das classes não foi informado no CSV. O score não indica, por si só, uma falha, sua causa ou certeza calibrada.</p>
      <h3>Como cada valor chegou à rede</h3><p>O pipeline salvo aplica {mutation.data.missing_strategy} aos faltantes e {mutation.data.scaler} aos atributos numéricos, usando somente parâmetros aprendidos no treino. O cenário B mostra o caminho completo abaixo.</p>
      <div className="table-scroll" tabIndex={0} role="region" aria-label="Transformação das leituras"><table className="data-table"><thead><tr><th>Atributo</th><th>Original</th><th>Após faltantes</th><th>Entrada na rede</th><th>Média / divisor do treino</th></tr></thead><tbody>{mutation.data.scenarios[1].numeric.map(item => <tr key={item.name}><th scope="row">{item.name}</th><td>{item.original === null ? 'Ausente' : number(item.original, 5)}</td><td>{number(item.imputed, 5)}</td><td>{number(item.scaled, 5)}</td><td>{item.mean === undefined ? 'Não se aplica' : `${number(item.mean, 5)} / ${number(item.scale, 5)}`}</td></tr>)}</tbody></table></div>
      <p className="caption">Para StandardScaler: (valor após imputação − média do treino) ÷ divisor do treino. Se o atributo é constante, o divisor usado é 1. environment é preenchido e codificado em colunas one-hot pelo pipeline salvo.</p>
      <Notice>Comparar cenários mostra o comportamento desta rede para essas entradas. A mudança observada não prova que um sensor causou o resultado nem informa a importância global do atributo.</Notice>
    </section>}
  </>
}

export default function UseModel({ navigate }: { navigate: (step: number) => void }) {
  const [selectedId, setSelectedId] = useState('')
  const runs = useQuery({ queryKey: ['experiments'], queryFn: ({ signal }) => get<Experiment[]>('/experiments', signal) })
  const completed = runs.data?.filter(run => run.status === 'completed') ?? []
  const selected = completed.find(run => run.id === selectedId) ?? completed[0]
  return <><div className="step-intro"><span className="eyebrow">APLICAR · MODELO TREINADO</span><h1>Use seu modelo<span className="title-dot">.</span></h1><p>Compare leituras e acompanhe o caminho dos dados até a resposta da rede.</p></div>
    <Notice><strong>Analogia didática: máquina de furação industrial.</strong> Imagine que cada atributo seja uma leitura de um sensor da máquina. O CSV não informa quais sensores, unidades ou significados das classes correspondem a uma máquina real. Os números e previsões abaixo são reais; a relação com a máquina é apenas uma ilustração.</Notice>
    {runs.isPending ? <Loading /> : runs.isError ? <ErrorState error={runs.error} retry={() => void runs.refetch()} /> : !selected ? <section className="panel"><h2>Nenhum modelo concluído</h2><p>Treine um experimento para usar seu pipeline e pesos aqui.</p><button className="button primary" onClick={() => navigate(7)}>Ir para treinamento</button></section> : <><section className="panel"><div className="field"><label htmlFor="use-model">Modelo treinado</label><select id="use-model" value={selected.id} onChange={e => setSelectedId(e.target.value)}>{completed.map(run => <option key={run.id} value={run.id}>{run.config.name} · {run.id.slice(0, 8)}</option>)}</select></div></section><ExperimentPanel key={selected.id} run={selected} navigate={navigate} /></>}
  </>
}
