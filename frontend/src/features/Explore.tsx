import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ArrowDownWideNarrow, ArrowLeft, ArrowRight, ChartNoAxesCombined, Grid2X2, ScanLine } from 'lucide-react'
import type { Correlations, Eda, Profile } from '../types'
import { get, number } from '../lib/api'
import { BoxPlot, CorrelationHeatmap, Distribution } from '../components/Charts'
import { ErrorState, Loading, Notice, PythonCode, Quiz } from '../components/UI'

type Tab = 'distribution' | 'missing' | 'correlations'

function Missing({ profile }: { profile: Profile }) {
  const [sort, setSort] = useState(true)
  const entries = Object.entries(profile.missing_by_column).filter(([name]) => profile.numeric_columns.includes(name) || profile.categorical_columns.includes(name))
  if (sort) entries.sort((a, b) => b[1].count - a[1].count)
  return <section className="panel">
    <div className="panel-heading"><div><h2>Onde faltam informações?</h2><p>Percentual de ausências em cada atributo, calculado sobre {number(profile.rows, 0)} registros.</p></div></div>
    <button className="sort-button" aria-pressed={sort} onClick={() => setSort(!sort)}><ArrowDownWideNarrow size={16} />{sort ? 'Maior percentual primeiro' : 'Ordem original das colunas'}</button>
    <div className="missing-chart" role="list" aria-label="Ausências por atributo">{entries.map(([name, missing]) => <div className="missing-row" role="listitem" key={name}>
      <span>{name}</span><div className="missing-track"><div style={{ width: `${missing.percent}%` }} /></div><strong>{number(missing.percent)}%</strong><small>{number(missing.count, 0)} ausentes</small>
    </div>)}</div>
    <div className="axis-caption"><span>0%</span><span>Percentual sobre todos os registros</span><span>100%</span></div>
    <Notice tone="amber">Cada barra conta ausências de uma coluna. <strong>Não some esses números para contar linhas incompletas</strong>: uma mesma amostra pode ter mais de um campo ausente.</Notice>
    <p className="caption">sample_id e target: {number(profile.missing_by_column.sample_id.count, 0)} e {number(profile.missing_by_column.target.count, 0)} ausências, respectivamente. Nenhuma linha foi removida.</p>
  </section>
}

function CorrelationsPanel() {
  const query = useQuery({ queryKey: ['correlations'], queryFn: ({ signal }) => get<Correlations>('/dataset/correlations', signal) })
  return <section className="panel"><div className="panel-heading"><div><h2>Como os atributos se relacionam?</h2><p>Correlação de Pearson: associação linear de −1 a +1. Clique em uma célula.</p></div></div>
    {query.isPending ? <Loading /> : query.isError ? <ErrorState error={query.error} retry={() => void query.refetch()} /> : <CorrelationHeatmap data={query.data} />}
  </section>
}

export default function Explore({ profile, back, next }: { profile: Profile; back: () => void; next: () => void }) {
  const [tab, setTab] = useState<Tab>('distribution')
  const [column, setColumn] = useState('sensor_a')
  const [bins, setBins] = useState(20)
  const query = useQuery({
    queryKey: ['eda', column, bins],
    queryFn: ({ signal }) => get<Eda>(`/dataset/eda?column=${encodeURIComponent(column)}&bins=${bins}`, signal),
    enabled: tab === 'distribution',
  })
  const isNumeric = profile.numeric_columns.includes(column)
  const stats = profile.numeric_summary[column]
  const baseCode = 'from pathlib import Path\nimport pandas as pd\n\ndf = pd.read_csv(Path("data/base_classificacao_binaria.csv"))\n'
  const code = tab === 'missing' ? `${baseCode}\nmissing = pd.DataFrame({\n    "count": df.isna().sum(),\n    "percent": df.isna().mean() * 100,\n})\nprint(missing.sort_values("percent", ascending=False))`
    : tab === 'correlations' ? `${baseCode}\nnumeric = df.drop(columns=["sample_id", "target"]).select_dtypes("number")\nprint(numeric.corr(method="pearson"))\npresent = numeric.notna().astype(int)\nprint(present.T @ present)  # número de pares válidos`
    : isNumeric ? `${baseCode}\nimport numpy as np\n\nvalues = df["${column}"].dropna()\ncounts, edges = np.histogram(values, bins=${bins})\nq1, median, q3 = values.quantile([0.25, 0.5, 0.75])\niqr = q3 - q1\ninliers = values[values.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]\nprint(counts, edges)\nprint(q1, median, q3)\nprint(inliers.min(), inliers.max())\nprint("Outliers:", len(values) - len(inliers))`
    : `${baseCode}\nprint(df["${column}"].fillna("(ausente)").astype(str).value_counts().sort_index())`
  return <>
    <div className="step-intro"><span className="eyebrow">ETAPA 02 / 09 · INVESTIGAR</span><h1>Conheça os seus dados<span className="title-dot">.</span></h1><p>Antes de treinar, procure distribuições, faltantes e relações. Observe os dados antes de decidir como tratá-los.</p></div>
    <Notice><strong>Exploração descritiva global.</strong> Estas estatísticas descrevem o CSV inteiro. Não serão reutilizadas para ajustar imputação, escala ou escolher o modelo; esses ajustes usarão somente o treino.</Notice>
    <div className="tabs" role="group" aria-label="Visualização da análise exploratória">
      <button aria-pressed={tab === 'distribution'} onClick={() => setTab('distribution')}><ChartNoAxesCombined size={17} />Distribuições</button>
      <button aria-pressed={tab === 'missing'} onClick={() => setTab('missing')}><ScanLine size={17} />Dados faltantes</button>
      <button aria-pressed={tab === 'correlations'} onClick={() => setTab('correlations')}><Grid2X2 size={17} />Correlações</button>
    </div>
    {tab === 'distribution' && <>
      <section className="panel">
        <div className="panel-heading"><div><h2>{isNumeric ? 'O formato dos valores' : 'Contagem por categoria'}</h2><p>{isNumeric ? 'Cada barra agrupa os registros em um intervalo de valores.' : 'Categorias são grupos, não posições em uma escala numérica.'}</p></div><span className="tag blue-tag">Dados reais</span></div>
        <div className="chart-controls"><div className="field"><label htmlFor="eda-column">Atributo</label><select id="eda-column" value={column} onChange={(event) => setColumn(event.target.value)}>{[...profile.numeric_columns, ...profile.categorical_columns, profile.label_column].map((name) => <option key={name}>{name}</option>)}</select></div>
          {isNumeric && <div className="field compact"><label htmlFor="eda-bins">Intervalos (bins)</label><select id="eda-bins" value={bins} onChange={(event) => setBins(Number(event.target.value))}>{[10, 20, 30].map((value) => <option key={value} value={value}>{value} intervalos</option>)}</select></div>}</div>
        {query.isPending ? <Loading /> : query.isError ? <ErrorState error={query.error} retry={() => void query.refetch()} /> : <>
          <Distribution data={query.data} />
          <div className="chart-summary" aria-live="polite"><span><strong>{number(query.data.total - query.data.missing, 0)}</strong> valores observados</span><span><strong>{number(query.data.missing, 0)}</strong> ausentes ({number(query.data.missing / query.data.total * 100)}%)</span></div>
          {isNumeric ? <p className="caption">Ausências não entram nos intervalos. {bins} barras calculadas para {column}. Unidades físicas não informadas.</p>
            : <p className="caption">A distribuição inclui ausências, quando existentes. “(ausente)” é um diagnóstico de falta de informação, não uma categoria real.</p>}
        </>}
      </section>
      {isNumeric && stats && <section className="stats-strip" aria-label={`Estatísticas de ${column}`}>
        {([['Média', stats.mean], ['Mediana', stats.median], ['Mínimo', stats.min], ['Máximo', stats.max]] as const).map(([name, value]) => <div key={name}><span>{name}</span><strong>{number(value, 4)}</strong></div>)}
      </section>}
      {isNumeric && query.data?.boxplot && !query.isFetching && <section className="panel">
        <div className="panel-heading"><div><h2>Valores típicos e pontos extremos</h2><p>O boxplot destaca a mediana, os 50% centrais e valores fora dos limites de 1,5 × IQR.</p></div></div>
        <BoxPlot box={query.data.boxplot} column={column} />
        <Notice tone="amber"><strong>Outlier não significa erro.</strong> {number(stats.negative_count, 0)} valores negativos observados nesta coluna. Sem documentação de domínio, não há justificativa para excluí-los automaticamente.</Notice>
      </section>}
    </>}
    {tab === 'missing' && <Missing profile={profile} />}
    {tab === 'correlations' && <CorrelationsPanel />}
    <PythonCode code={code} />
    <Quiz question="Uma correlação alta prova que um atributo causa a classe?" options={['Sim, correlação é causa', 'Não, mostra apenas associação']} correct={1} explanation="Correlação mede associação linear. Ela não estabelece causa, e uma correlação baixa também não descarta relações não lineares." />
    <div className="step-footer"><button className="button secondary" onClick={back}><ArrowLeft size={16} />Entendimento do problema</button><button className="button primary" onClick={next}>Tratar os faltantes<ArrowRight size={16} /></button></div>
  </>
}
