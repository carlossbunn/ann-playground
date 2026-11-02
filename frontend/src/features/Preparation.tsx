import { useState, type FormEvent } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ArrowLeft, ArrowRight, Check, GitBranch, LockKeyhole, RotateCcw, ShieldCheck } from 'lucide-react'
import type { PreparationStats, PreprocessingPreview, PreprocessingSettings, Profile } from '../types'
import { number, post } from '../lib/api'
import { defaultSettings, missingNames, scalerNames } from '../lib/preprocessing'
import { Distribution } from '../components/Charts'
import { ErrorState, Loading, Notice, PythonCode, Quiz } from '../components/UI'
import './preparation.css'

type Props = {
  step: number
  profile: Profile
  settings: PreprocessingSettings
  onChange: (settings: PreprocessingSettings) => void
  navigate: (step: number) => void
}

const titles = {
  3: ['PREPARAR', 'Dê um destino aos faltantes', 'Compare o que muda ao estimar os valores ou manter somente as amostras completas.'],
  4: ['TRANSFORMAR', 'Coloque as escalas em perspectiva', 'Aproxime as magnitudes dos atributos e veja a diferença nos valores que entrarão na rede.'],
  5: ['RESERVAR', 'Um papel para cada conjunto', 'Treino aprende, validação orienta as escolhas e teste fica reservado para a avaliação final.'],
} as Record<number, string[]>

function SeedControl({ seed, apply }: { seed: number; apply: (seed: number) => void }) {
  const [value, setValue] = useState(String(seed))
  const [error, setError] = useState('')
  function submit(event: FormEvent) {
    event.preventDefault()
    const parsed = Number(value)
    if (!value.trim() || !Number.isInteger(parsed) || parsed < 0 || parsed > 4294967295) {
      setError('Informe um inteiro de 0 a 4294967295.')
      return
    }
    setError('')
    apply(parsed)
  }
  return <form className="seed-form" onSubmit={submit}>
    <div className="field"><label htmlFor="split-seed">Seed reproduzível</label>
      <input id="split-seed" inputMode="numeric" value={value} onChange={(event) => setValue(event.target.value)} aria-describedby="seed-description" />
    </div>
    <button type="submit" className="button secondary">Aplicar seed</button>
    <p id="seed-description">Mesma fonte, estratégia e seed reproduzem a seleção das linhas.</p>
    {error && <p role="alert">{error}</p>}
  </form>
}

function StatisticsTable({ before, after, firstLabel, secondLabel }: {
  before: PreparationStats; after: PreparationStats; firstLabel: string; secondLabel: string
}) {
  const labels: [keyof PreparationStats, string][] = [
    ['observed', 'Valores observados'], ['missing', 'Valores ausentes'], ['mean', 'Média'],
    ['median', 'Mediana'], ['std', 'Desvio padrão (populacional)'], ['min', 'Mínimo'], ['max', 'Máximo'],
  ]
  return <div className="table-scroll" tabIndex={0} role="region" aria-label="Estatísticas antes e depois">
    <table className="data-table preparation-table"><caption className="sr-only">Comparação estatística no subconjunto selecionado</caption>
      <thead><tr><th scope="col">Medida</th><th scope="col">{firstLabel}</th><th scope="col">{secondLabel}</th></tr></thead>
      <tbody>{labels.map(([key, label]) => <tr key={key}><th scope="row">{label}</th><td>{number(before[key], 5)}</td><td>{number(after[key], 5)}</td></tr>)}</tbody>
    </table>
  </div>
}

function Coverage({ data }: { data: PreprocessingPreview }) {
  return <>
    <section className="stats-strip coverage-summary" aria-label="Cobertura da coorte">
      <div><span>Base original</span><strong>{number(data.dataset_rows, 0)}</strong></div>
      <div><span>Amostras mantidas</span><strong>{number(data.rows_after_filter, 0)}</strong></div>
      <div><span>Linhas removidas</span><strong>{number(data.removed_rows, 0)}</strong></div>
      <div><span>Cobertura</span><strong>{number(data.coverage_percent)}%</strong></div>
    </section>
    {data.removed_rows > 0 && <section className="panel">
      <div className="panel-heading"><div><h2>A população do estudo mudou</h2><p>A remoção considera ausências em todos os 11 atributos, antes da divisão.</p></div></div>
      <div className="table-scroll" tabIndex={0} role="region" aria-label="Cobertura por classe"><table className="data-table preparation-table">
        <thead><tr><th scope="col">Classe</th><th scope="col">Original</th><th scope="col">Mantidas</th><th scope="col">Removidas</th><th scope="col">Cobertura</th></tr></thead>
        <tbody>{Object.entries(data.cohort_by_class).map(([label, counts]) => <tr key={label}><th scope="row">Classe {label}</th><td>{number(counts.original, 0)}</td><td>{number(counts.retained, 0)}</td><td>{number(counts.original - counts.retained, 0)}</td><td>{number(counts.retained / counts.original * 100)}%</td></tr>)}</tbody>
      </table></div>
      <p className="caption">O modelo futuro representará apenas a coorte completa. A perda de amostras pode introduzir viés de seleção.</p>
    </section>}
  </>
}

function CategoryPreparation({ data }: { data: PreprocessingPreview }) {
  const environment = data.environment
  return <section className="panel">
    <div className="panel-heading"><div><h2>environment tem um tratamento próprio</h2><p>A categoria não tem ordem numérica. As colunas one-hot indicam presença ou ausência de cada categoria.</p></div><span className="tag green-tag">OneHotEncoder</span></div>
    <div className="category-flow"><span>Ausências: <strong>{number(environment.before_missing, 0)}</strong></span><ArrowRight size={16} /><span>Preenchimento: <strong>{environment.fill_value}</strong></span><ArrowRight size={16} /><span>Ausências depois: <strong>{environment.after_missing}</strong></span></div>
    <div className="encoded-categories">{Object.entries(environment.encoded_counts).map(([category, count]) => <span key={category}><code>environment_{category}</code><strong>{number(count, 0)}</strong></span>)}</div>
    <p className="caption">{data.output_feature_count} atributos após transformação: 10 numéricos + {environment.categories.length} colunas categóricas. Categorias desconhecidas em validação são codificadas como zeros, sem reajustar o encoder. {environment.strategy === 'constant' && '“missing” é um marcador explícito para ausência, não um novo ambiente real.'}</p>
    <details className="feature-list"><summary>Ver nomes de todos os atributos de saída</summary><ul>{data.output_features.map((name) => <li key={name}><code>{name}</code></li>)}</ul></details>
  </section>
}

function BeforeAfter({ data, step }: { data: PreprocessingPreview; step: number }) {
  const comparison = data.numeric_before_after[0]
  const isScaling = step === 4
  const before = isScaling ? comparison.imputed_stats : comparison.before_stats
  const after = isScaling ? comparison.scaled_stats : comparison.imputed_stats
  const beforeHistogram = isScaling ? comparison.imputed_histogram : comparison.before_histogram
  const afterHistogram = isScaling ? comparison.scaled_histogram : comparison.imputed_histogram
  const firstLabel = isScaling ? 'Antes da escala' : 'Antes da imputação'
  const secondLabel = isScaling ? 'Após a escala' : 'Após a imputação'
  return <>
    <section className="panel">
      <div className="panel-heading"><div><h2>{comparison.column}: antes × depois</h2><p>Distribuições de todo o subconjunto {data.preview_subset === 'train' ? 'de treino' : 'de validação'}; parâmetros sempre aprendidos no treino.</p></div></div>
      <div className="histogram-comparison">
        <div><h3>{firstLabel}</h3><Distribution data={{ column: `${comparison.column} · ${firstLabel.toLowerCase()}`, kind: 'numeric', total: before.observed + before.missing, missing: before.missing, histogram: beforeHistogram }} /></div>
        <div><h3>{secondLabel}</h3><Distribution data={{ column: `${comparison.column} · ${secondLabel.toLowerCase()}`, kind: 'numeric', total: after.observed + after.missing, missing: after.missing, histogram: afterHistogram }} /></div>
      </div>
      <StatisticsTable before={before} after={after} firstLabel={firstLabel} secondLabel={secondLabel} />
      <p className="caption">Antes da imputação, estatísticas excluem os ausentes; depois, consideram o conjunto completo. Os gráficos possuem eixos próprios: observe os valores, especialmente após mudar a escala.</p>
      {comparison.fitted_replacement !== null && <p className="fitted-value" aria-live="polite">{missingNames[data.config.missing_strategy]} aprendida no treino para <strong>{comparison.column}</strong>: <strong>{number(comparison.fitted_replacement, 6)}</strong>.</p>}
    </section>
    <section className="panel"><div className="panel-heading"><div><h2>Veja os valores de cada amostra</h2><p>Até 12 exemplos reais de {data.preview_subset === 'train' ? 'treino' : 'validação'}, com ausências primeiro. Esta seleção ilustrativa não é uma amostra aleatória.</p></div></div>
      <div className="table-scroll" tabIndex={0} role="region" aria-label="Amostras antes e depois"><table className="data-table preparation-table">
        <thead><tr><th scope="col">sample_id · apenas identificação</th><th scope="col">Valor original</th><th scope="col">Após imputação</th>{isScaling && <th scope="col">Após escala</th>}</tr></thead>
        <tbody>{comparison.samples.map((sample) => <tr key={sample.sample_id}><th scope="row">{sample.sample_id}</th><td>{sample.before === null ? <span className="missing-value">ausente</span> : number(sample.before, 6)}</td><td>{number(sample.imputed, 6)}</td>{isScaling && <td>{number(sample.scaled, 6)}</td>}</tr>)}</tbody>
      </table></div>
    </section>
  </>
}

function SplitResult({ data }: { data: PreprocessingPreview }) {
  const names = { train: 'Treino', val: 'Validação', test: 'Teste reservado' }
  return <section className="panel">
    <div className="panel-heading"><div><h2>Três conjuntos, funções diferentes</h2><p>Divisão estratificada sobre {number(data.rows_after_filter, 0)} amostras da coorte atual.</p></div><GitBranch size={19} /></div>
    <div className="split-proportion" role="img" aria-label={Object.entries(data.split_counts).map(([name, count]) => `${names[name as keyof typeof names]}: ${number(count, 0)} amostras`).join('; ')}>
      {Object.entries(data.split_counts).map(([name, count]) => <div key={name} className={name} style={{ width: `${count / data.rows_after_filter * 100}%` }}><strong>{number(count / data.rows_after_filter * 100, 1)}%</strong><span>{names[name as keyof typeof names]}</span></div>)}
    </div>
    <div className="table-scroll" tabIndex={0} role="region" aria-label="Contagem e classes por partição"><table className="data-table preparation-table">
      <thead><tr><th scope="col">Partição</th><th scope="col">Registros</th><th scope="col">Classe 0</th><th scope="col">Classe 1</th><th scope="col">Papel</th></tr></thead>
      <tbody>{(['train', 'val', 'test'] as const).map((name) => <tr key={name}><th scope="row">{names[name]}</th><td>{number(data.split_counts[name], 0)}</td><td>{number(data.split_classes[name]['0'], 0)}</td><td>{number(data.split_classes[name]['1'], 0)}</td><td>{name === 'train' ? 'Aprender parâmetros' : name === 'val' ? 'Comparar escolhas' : 'Avaliação final futura'}</td></tr>)}</tbody>
    </table></div>
    <Notice><strong>A divisão já acontece antes das prévias das etapas 3 e 4.</strong> A ordem das telas é didática: imputador, scaler e encoder foram ajustados apenas nas {number(data.split_counts.train, 0)} amostras de treino.</Notice>
    <Notice tone="green"><LockKeyhole size={15} />O teste está reservado. Aqui mostramos somente contagens e identidade da partição; seus atributos não passaram por transformação e não orientam as escolhas.</Notice>
    <details className="partition-manifest"><summary>Rastrear e reproduzir estas partições</summary><p>Seed: {data.config.seed} · scikit-learn: {data.sklearn_version}. Hashes SHA-256 dos IDs na ordem de cada subconjunto.</p>
      <dl>{Object.entries(data.partition_fingerprints).map(([name, hash]) => <div key={name}><dt>{names[name as keyof typeof names]}</dt><dd><code>{hash}</code></dd></div>)}<div><dt>CSV original</dt><dd><code>{data.dataset_sha256}</code></dd></div></dl>
      <p>O bloco “Ver o Python” reproduz a seleção e imprime os IDs de cada partição para registro local.</p>
    </details>
  </section>
}

export default function Preparation({ step, profile, settings, onChange, navigate }: Props) {
  const [column, setColumn] = useState('sensor_b')
  const [subset, setSubset] = useState<'train' | 'val'>('train')
  const [compareStrategy, setCompareStrategy] = useState<PreprocessingSettings['missing_strategy'] | ''>('')
  const request = { ...settings, column, subset, sample_limit: 12, bins: 20 }
  const query = useQuery({
    queryKey: ['preprocessing', request],
    queryFn: ({ signal }) => post<PreprocessingPreview>('/preprocessing/preview', request, signal),
  })
  const comparisonRequest = { ...request, missing_strategy: compareStrategy || settings.missing_strategy }
  const comparisonQuery = useQuery({
    queryKey: ['preprocessing', comparisonRequest],
    queryFn: ({ signal }) => post<PreprocessingPreview>('/preprocessing/preview', comparisonRequest, signal),
    enabled: step === 3 && Boolean(compareStrategy),
  })
  function update(patch: Partial<PreprocessingSettings>) { onChange({ ...settings, ...patch }) }
  const [stage, title, description] = titles[step]
  return <>
    <div className="step-intro"><span className="eyebrow">ETAPA {String(step).padStart(2, '0')} / 09 · {stage}</span><h1>{title}<span className="title-dot">.</span></h1><p>{description}</p></div>
    <Notice><ShieldCheck size={17} /><strong>Fit somente no treino.</strong> As prévias usam uma divisão estratificada já criada. Validação recebe apenas transform; o teste permanece intocado.</Notice>
    <section className="panel preparation-controls">
      <div className="panel-heading"><div><h2>{step === 3 ? 'Como lidar com as ausências?' : step === 4 ? 'Escolha a transformação' : 'Defina a separação e a seed'}</h2><p>As escolhas são compartilhadas entre as etapas 3, 4 e 5.</p></div><button className="icon-button" aria-label="Restaurar padrões de preparação" onClick={() => onChange({ ...defaultSettings })}><RotateCcw size={16} /></button></div>
      {step === 3 && <>
        <div className="choice-grid" role="group" aria-label="Estratégia numérica de faltantes">
          {(['drop', 'mean', 'median', 'knn'] as const).map((strategy) => <button key={strategy} aria-pressed={settings.missing_strategy === strategy} onClick={() => update({ missing_strategy: strategy })}><strong>{missingNames[strategy]}</strong><small>{{ drop: 'Só amostras completas', mean: 'Sensível a extremos', median: 'Robusta a extremos', knn: 'Vizinhos no treino' }[strategy]}</small></button>)}
        </div>
        <div className="preparation-fields">
          {settings.missing_strategy === 'knn' && <div className="field"><label htmlFor="knn-neighbors">Vizinhos do KNN</label><select id="knn-neighbors" value={settings.knn_neighbors} onChange={(event) => update({ knn_neighbors: Number(event.target.value) as 3 | 5 | 7 })}>{[3, 5, 7].map((n) => <option key={n} value={n}>{n} vizinhos</option>)}</select></div>}
          <div className="field"><label htmlFor="category-treatment">Faltantes de environment</label><select id="category-treatment" value={settings.categorical_strategy} onChange={(event) => update({ categorical_strategy: event.target.value as PreprocessingSettings['categorical_strategy'] })}><option value="most_frequent">Moda aprendida no treino</option><option value="constant">Marcador explícito: missing</option></select></div>
        </div>
      </>}
      {step === 4 && <div className="choice-grid scaler-choices" role="group" aria-label="Método de escalonamento">{(['standard', 'minmax', 'none'] as const).map((scaler) => <button key={scaler} aria-pressed={settings.scaler === scaler} onClick={() => update({ scaler })}><strong>{scalerNames[scaler]}</strong><small>{{ standard: 'Média 0 e desvio padrão 1 no treino', minmax: 'Extremos do treino em [0, 1]', none: 'Experimento didático adicional' }[scaler]}</small></button>)}</div>}
      {step === 5 ? <>
        <div className="choice-grid split-choices" role="group" aria-label="Estratégia de divisão"><button aria-pressed={settings.split_strategy === '70_15_15'} onClick={() => update({ split_strategy: '70_15_15' })}><strong>70 / 15 / 15</strong><small>Treino · validação · teste</small></button><button aria-pressed={settings.split_strategy === '80_20'} onClick={() => update({ split_strategy: '80_20' })}><strong>80 / 20 com validação interna</strong><small>Efetivo: 64% treino · 16% validação · 20% teste</small></button></div>
        <SeedControl key={settings.seed} seed={settings.seed} apply={(seed) => update({ seed })} />
      </> : <div className="preparation-fields">
        <div className="field"><label htmlFor="preparation-column">Atributo para comparar</label><select id="preparation-column" value={column} onChange={(event) => setColumn(event.target.value)}>{profile.numeric_columns.map((name) => <option key={name}>{name}</option>)}</select></div>
        <div className="field"><label htmlFor="preview-subset">Visualizar subconjunto</label><select id="preview-subset" value={subset} onChange={(event) => setSubset(event.target.value as 'train' | 'val')}><option value="train">Treino · onde ocorre o fit</option><option value="val">Validação · apenas transform</option></select></div>
      </div>}
      <div className="current-preparation"><Check size={14} /><span>{missingNames[settings.missing_strategy]} · {scalerNames[settings.scaler]} · {settings.split_strategy === '70_15_15' ? '70/15/15' : '64/16/20'} · seed {settings.seed}</span></div>
    </section>
    {query.isPending ? <Loading /> : query.isError ? <ErrorState error={query.error} retry={() => void query.refetch()} /> : <>
      <div className="preparation-status" role="status">Prévia calculada · fit em {number(query.data.split_counts.train, 0)} amostras de treino · seed {query.data.config.seed}</div>
      {query.data.warnings.length > 0 && <Notice tone="amber"><ul className="preparation-warnings">{query.data.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul></Notice>}
      <Coverage data={query.data} />
      {step === 5 ? <SplitResult data={query.data} /> : <BeforeAfter data={query.data} step={step} />}
      {step === 3 && <>
        <CategoryPreparation data={query.data} />
        <section className="panel"><div className="panel-heading"><div><h2>Compare uma segunda estratégia</h2><p>Mesma fonte, seed, coluna e subconjunto. Remover linhas altera a coorte e, por isso, as partições.</p></div></div>
          <div className="field"><label htmlFor="compare-strategy">Estratégia para comparar</label><select id="compare-strategy" value={compareStrategy} onChange={(event) => setCompareStrategy(event.target.value as typeof compareStrategy)}><option value="">Escolha uma estratégia</option>{(['drop', 'mean', 'median', 'knn'] as const).map((strategy) => <option key={strategy} value={strategy}>{missingNames[strategy]}</option>)}</select></div>
          {compareStrategy && (comparisonQuery.isPending ? <Loading /> : comparisonQuery.isError ? <ErrorState error={comparisonQuery.error} retry={() => void comparisonQuery.refetch()} /> : comparisonQuery.data && <>
            <div className="table-scroll comparison-result" tabIndex={0} role="region" aria-label="Comparação entre estratégias"><table className="data-table preparation-table"><thead><tr><th scope="col">Estratégia</th><th scope="col">Coorte mantida</th><th scope="col">Cobertura</th><th scope="col">Média após imputação</th><th scope="col">Mediana após imputação</th></tr></thead><tbody>{[query.data, comparisonQuery.data].map((data, i) => <tr key={i}><th scope="row">{missingNames[data.config.missing_strategy]}</th><td>{number(data.rows_after_filter, 0)}</td><td>{number(data.coverage_percent)}%</td><td>{number(data.numeric_before_after[0].imputed_stats.mean, 6)}</td><td>{number(data.numeric_before_after[0].imputed_stats.median, 6)}</td></tr>)}</tbody></table></div>
            <p className="caption">Estatísticas do atributo {column} no subconjunto {subset === 'train' ? 'de treino' : 'de validação'}. Estas não são métricas de classificação.</p>
          </>)}
        </section>
        <section className="panel"><div className="panel-heading"><div><h2>Ausências neste subconjunto</h2><p>Contagens de {subset === 'train' ? 'treino' : 'validação'}, não da base global da etapa 2.</p></div></div><div className="table-scroll" tabIndex={0} role="region" aria-label="Faltantes antes e depois"><table className="data-table preparation-table"><thead><tr><th scope="col">Atributo</th><th scope="col">Antes</th><th scope="col">Depois</th></tr></thead><tbody>{Object.entries(query.data.missing_before_after).map(([name, counts]) => <tr key={name}><th scope="row">{name}</th><td>{number(counts.before, 0)}</td><td>{counts.after}</td></tr>)}</tbody></table></div></section>
      </>}
      <PythonCode code={query.data.python_code} />
    </>}
    <Quiz key={step} question={step === 3 ? 'Posso calcular a mediana juntando treino e teste?' : step === 4 ? 'StandardScaler precisa ser ajustado em qual conjunto?' : 'Qual conjunto orienta a escolha dos hiperparâmetros?'}
      options={step === 3 ? ['Sim, assim uso mais informações', 'Não, somente o treino'] : step === 4 ? ['Somente no treino', 'Na base inteira antes do split'] : ['O conjunto de teste', 'O conjunto de validação']}
      correct={step === 4 ? 0 : 1} explanation={step === 3 ? 'O teste não deve influenciar os valores usados para preparar a entrada. A mediana é aprendida só no treino.' : step === 4 ? 'O scaler aprende a média e o desvio padrão do treino. Validação e teste só recebem transform.' : 'A validação orienta as escolhas. O teste só é aberto após selecionar o experimento final.'} />
    <div className="step-footer"><button className="button secondary" onClick={() => navigate(step - 1)}><ArrowLeft size={16} />{step === 3 ? 'Explorar os dados' : step === 4 ? 'Tratar os faltantes' : 'Ajustar as escalas'}</button><button className="button primary" onClick={() => navigate(step + 1)}>{step === 3 ? 'Ajustar as escalas' : step === 4 ? 'Separar os dados' : 'Conhecer a MLP'}<ArrowRight size={16} /></button></div>
  </>
}
