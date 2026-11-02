import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useState } from 'react'
import type { Boxplot, Correlations, Eda } from '../types'
import { number } from '../lib/api'

export function Distribution({ data }: { data: Eda }) {
  const numeric = data.kind === 'numeric'
  const chart = numeric ? data.histogram?.map((bin) => ({
    name: number(bin.bin_start, 3), count: bin.count,
    interval: `${number(bin.bin_start, 4)} a ${number(bin.bin_end, 4)}`,
  })) : data.categories?.map((item) => ({ name: item.category, count: item.count, interval: item.category }))
  if (!chart?.length) return <p className="empty-state">Não há valores observados para construir esta distribuição.</p>
  return <div className="distribution" role="img" aria-label={`Distribuição de ${data.column}. ${number(data.total - data.missing, 0)} valores observados; ${number(data.missing, 0)} ausentes.`}>
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={chart} margin={{ top: 16, right: 12, bottom: 30, left: 4 }} barCategoryGap={numeric ? '8%' : '34%'}>
        <CartesianGrid strokeDasharray="3 4" vertical={false} stroke="#e6ebf3" />
        <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#66758b' }} axisLine={false} tickLine={false}
          interval="preserveStartEnd" label={{ value: numeric ? `${data.column} · início de cada intervalo` : data.column, position: 'bottom', fontSize: 11, fill: '#66758b' }} />
        <YAxis tick={{ fontSize: 11, fill: '#66758b' }} axisLine={false} tickLine={false} width={48}
          tickFormatter={(value: number) => number(value, 0)} label={{ value: 'Frequência', angle: -90, position: 'insideLeft', fontSize: 11, fill: '#66758b' }} />
        <Tooltip cursor={{ fill: '#eef2fc' }} content={({ active, payload }) => {
          const item = payload?.[0]?.payload
          return active && item ? <div className="chart-tooltip"><strong>{item.interval}</strong><br />{number(item.count, 0)} registros</div> : null
        }} />
        <Bar dataKey="count" fill="#5679df" radius={[4, 4, 0, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  </div>
}

export function BoxPlot({ box, column }: { box: Boxplot; column: string }) {
  const range = box.max - box.min || 1
  const x = (value: number) => 36 + (value - box.min) / range * 568
  const ticks = [box.min, box.min + range / 2, box.max]
  return <div className="boxplot">
    <svg viewBox="0 0 640 134" role="img" aria-label={`Boxplot de ${column}: mediana ${number(box.median, 4)}, ${number(box.outlier_count, 0)} outliers pelo critério 1,5 IQR.`}>
      <line x1="36" x2="604" y1="96" y2="96" stroke="#e5eaf2" />
      {ticks.map((value, i) => <g key={i}><line x1={x(value)} x2={x(value)} y1="93" y2="99" stroke="#a7b2c4" />
        <text x={x(value)} y="117" textAnchor="middle" fontSize="11" fill="#66758b">{number(value, 3)}</text></g>)}
      <line x1={x(box.lower_whisker)} x2={x(box.upper_whisker)} y1="52" y2="52" stroke="#315ddb" strokeWidth="2" />
      {[box.lower_whisker, box.upper_whisker].map((value, i) => <line key={i} x1={x(value)} x2={x(value)} y1="36" y2="68" stroke="#315ddb" strokeWidth="2" />)}
      <rect x={x(box.q1)} y="29" width={Math.max(1, x(box.q3) - x(box.q1))} height="46" rx="4" fill="#dce5ff" stroke="#315ddb">
        <title>50% centrais: {number(box.q1, 4)} a {number(box.q3, 4)}</title>
      </rect>
      <line x1={x(box.median)} x2={x(box.median)} y1="29" y2="75" stroke="#315ddb" strokeWidth="3"><title>Mediana: {number(box.median, 4)}</title></line>
      {box.outlier_points.map((value, i) => <circle key={i} cx={x(value)} cy={48 + (i % 3 - 1) * 6} r="2.5" fill="#b96d21" fillOpacity="0.65"><title>Valor observado: {number(value, 5)}</title></circle>)}
    </svg>
    <div className="box-legend"><span><i className="legend-dot blue-dot" />Intervalo interquartil (Q1–Q3)</span><span><i className="legend-dot amber-dot" />Outliers (1,5 × IQR)</span></div>
    <p className="caption">Q1: {number(box.q1, 4)} · Mediana: {number(box.median, 4)} · Q3: {number(box.q3, 4)}. {number(box.outlier_count, 0)} outliers; até 80 pontos reais exibidos. Unidades não informadas.</p>
  </div>
}

export function CorrelationHeatmap({ data }: { data: Correlations }) {
  const [selected, setSelected] = useState<[number, number]>([0, 1])
  const [row, col] = selected
  return <>
    <div className="heatmap-scroll" tabIndex={0} role="region" aria-label="Matriz de correlação, rolável horizontalmente">
      <table className="heatmap"><caption className="sr-only">Correlação de Pearson entre os dez atributos numéricos. Selecione uma célula para detalhes.</caption>
        <thead><tr><th scope="col">Atributo</th>{data.columns.map((name, i) => <th key={name} scope="col" title={name}>{i + 1}</th>)}</tr></thead>
        <tbody>{data.columns.map((name, i) => <tr key={name}><th scope="row"><span>{i + 1}</span>{name}</th>
          {data.matrix[i].map((value, j) => {
            const magnitude = Math.abs(value ?? 0)
            const background = value == null ? '#f0f2f5' : value >= 0
              ? `rgba(49, 93, 219, ${0.06 + magnitude * 0.94})` : `rgba(179, 74, 51, ${0.06 + magnitude * 0.94})`
            return <td key={j}><button style={{ background, color: magnitude > 0.6 ? '#fff' : '#34435c' }}
              aria-label={`${name} e ${data.columns[j]}: ${number(value, 2)}`}
              aria-pressed={row === i && col === j} onClick={() => setSelected([i, j])}>{value == null ? '—' : number(value, 2)}</button></td>
          })}</tr>)}</tbody>
      </table>
    </div>
    <div className="heatmap-legend"><span>−1 · negativa</span><i /><span>+1 · positiva</span></div>
    <p className="correlation-detail" aria-live="polite"><strong>{data.columns[row]} × {data.columns[col]}</strong><br />
      r = {number(data.matrix[row][col], 4)} · {number(data.sample_counts[row][col], 0)} pares válidos. Ausências excluídas par a par.</p>
    <p className="caption">Os números do cabeçalho correspondem aos atributos das linhas. ID, target e environment excluídos. Correlação não demonstra causalidade.</p>
  </>
}
