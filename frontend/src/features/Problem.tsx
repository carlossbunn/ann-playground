import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ArrowDown, ArrowRight, ChevronLeft, ChevronRight, Database, Hash, LockKeyhole, Target } from 'lucide-react'
import type { Preview, Profile } from '../types'
import { get, number } from '../lib/api'
import { ErrorState, Loading, Notice, PythonCode, Quiz } from '../components/UI'

function DataPreview() {
  const [offset, setOffset] = useState(0)
  const pageSize = 10
  const query = useQuery({
    queryKey: ['preview', offset],
    queryFn: ({ signal }) => get<Preview>(`/dataset/preview?offset=${offset}&limit=${pageSize}`, signal),
  })
  return <section className="panel">
    <div className="panel-heading"><div><h2>Uma olhada nos dados</h2><p>Dados originais, sem tratamento. Células vazias são valores ausentes.</p></div><span className="tag">CSV original</span></div>
    {query.isPending ? <Loading /> : query.isError ? <ErrorState error={query.error} retry={() => void query.refetch()} /> : <>
      <div className="table-scroll" tabIndex={0} role="region" aria-label="Prévia do CSV, rolável horizontalmente">
        <table className="data-table"><caption className="sr-only">Registros {offset + 1} a {offset + query.data.rows.length} da base original</caption>
          <thead><tr>{query.data.columns.map((col) => <th scope="col" key={col}>{col}</th>)}</tr></thead>
          <tbody>{query.data.rows.map((row) => <tr key={String(row.sample_id)}>{query.data.columns.map((col) =>
            <td key={col} className={col === 'target' ? 'target-cell' : ''}>{row[col] == null ? <span className="missing-value" title="Valor ausente">ausente</span>
              : typeof row[col] === 'number' ? number(row[col] as number, 4) : row[col]}</td>)}</tr>)}</tbody>
        </table>
      </div>
      <div className="pagination"><span aria-live="polite">{number(offset + 1, 0)}–{number(offset + query.data.rows.length, 0)} de {number(query.data.total, 0)} registros</span>
        <div><button className="icon-button" aria-label="Página anterior" disabled={offset === 0} onClick={() => setOffset(offset - pageSize)}><ChevronLeft size={18} /></button>
          <button className="icon-button" aria-label="Próxima página" disabled={offset + pageSize >= query.data.total} onClick={() => setOffset(offset + pageSize)}><ChevronRight size={18} /></button></div>
      </div>
    </>}
  </section>
}

export default function Problem({ profile, next }: { profile: Profile; next: () => void }) {
  const [column, setColumn] = useState('sensor_a')
  const role = column === profile.id_column ? 'id' : column === profile.label_column ? 'label' : 'feature'
  const explanations = {
    id: 'sample_id identifica a amostra. Ele não descreve o fenômeno e será sempre excluído dos atributos preditores.',
    label: 'target é a resposta que queremos prever. Colocá-la na entrada revelaria a resposta à rede: isso é vazamento de dados (data leakage).',
    feature: column === 'environment'
      ? 'environment é um atributo categórico, sem ordem numérica. Na preparação, ele será imputado e codificado com OneHotEncoder.'
      : `${column} é um atributo numérico. Pode ajudar a rede a reconhecer padrões, mas sua unidade e seu significado físico não foram informados.`,
  }
  const code = `from pathlib import Path\nimport pandas as pd\n\ndf = pd.read_csv(Path("data/base_classificacao_binaria.csv"))\nX = df.drop(columns=["sample_id", "target"])\ny = df["target"]\n\nprint(df.shape)\nprint(y.value_counts().sort_index())\nprint(X.dtypes)\n# Apenas inspeção: nenhum pré-processador foi ajustado.\n# O fit de imputação/escala deve acontecer só depois do split.`
  return <>
    <div className="step-intro"><span className="eyebrow">ETAPA 01 / 09 · CONHECER</span><h1>Entenda o problema<span className="title-dot">.</span></h1>
      <p>A rede aprende com os atributos de cada amostra para prever uma resposta: <code>target</code>, que pode ser 0 ou 1.</p></div>
    <section className="role-grid" aria-label="Papéis das colunas">
      <div className="role-card"><span className="role-icon"><Database size={21} /></span><span className="small-label">ENTRADA · X</span><h2>{profile.numeric_columns.length + profile.categorical_columns.length} atributos</h2><p>{profile.numeric_columns.length} numéricos + {profile.categorical_columns.length} categórico</p><span className="tag blue-tag">Features</span></div>
      <div className="role-card"><span className="role-icon teal"><Target size={21} /></span><span className="small-label">RESPOSTA · y</span><h2>target</h2><p>Classe 0 ou classe 1</p><span className="tag green-tag">Label</span></div>
      <div className="role-card"><span className="role-icon gray"><Hash size={21} /></span><span className="small-label">IDENTIFICAÇÃO</span><h2>sample_id</h2><p>Rastreia cada registro</p><span className="tag"><LockKeyhole size={12} />Fora da entrada</span></div>
    </section>
    <section className="panel">
      <div className="panel-heading"><div><h2>Qual é o papel desta coluna?</h2><p>Explore os campos para separar o que entra na rede do que ela deve prever.</p></div></div>
      <div className="field-explorer"><div className="field"><label htmlFor="role-column">Coluna do dataset</label><select id="role-column" value={column} onChange={(event) => setColumn(event.target.value)}>{profile.column_names.map((name) => <option key={name}>{name}</option>)}</select></div>
        <div className={`role-result ${role}`} aria-live="polite"><span className="tag">{role === 'feature' ? 'Atributo preditor' : role === 'id' ? 'Identificador · bloqueado' : 'Alvo · bloqueado como entrada'}</span><p>{explanations[role]}</p></div>
      </div>
      <div className="pipeline-diagram"><span>11 atributos <small>padrões observados</small></span><ArrowRight size={21} /><span className="network-node">Rede neural <small>aprende uma relação</small></span><ArrowRight size={21} /><span>0 ou 1 <small>classe prevista</small></span></div>
      <Notice>O significado de <strong>0 e 1</strong>, as unidades e o contexto dos sensores não foram informados. Não podemos interpretar as classes como “falha” ou “sucesso”.</Notice>
    </section>
    <section className="panel class-balance"><div className="panel-heading"><div><h2>Duas classes, a mesma quantidade</h2><p>Aprendizado supervisionado: cada registro já possui uma resposta conhecida.</p></div></div>
      <div className="balance-bar" role="img" aria-label={`Classe 0: ${profile.target_distribution['0']}; classe 1: ${profile.target_distribution['1']}`}>
        {Object.entries(profile.target_distribution).map(([label, count]) => <div key={label} style={{ width: `${count / profile.rows * 100}%` }}>Classe {label} · {number(count, 0)}</div>)}
      </div><p className="caption">O balanceamento não garante que o problema seja fácil nem que a rede terá alta acurácia.</p>
    </section>
    <div className="section-marker"><ArrowDown size={16} />Agora, veja como isso aparece no arquivo</div>
    <DataPreview />
    <PythonCode code={code} />
    <Quiz question="Se eu colocar target entre os atributos de entrada, o que acontece?" options={['A rede ganha uma informação útil', 'A resposta vaza para o modelo']} correct={1} explanation="target é a resposta. Usá-la como entrada produz uma avaliação enganosa, em vez de medir aprendizado real." />
    <div className="step-footer"><span>Próximo: conhecer as distribuições e ausências.</span><button className="button primary" onClick={next}>Explorar os dados<ArrowRight size={17} /></button></div>
  </>
}
