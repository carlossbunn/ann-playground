import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowLeft, ArrowRight, Check, Download, FlaskConical, LockKeyhole, Network, Play, Plus, RefreshCw, Trash2 } from 'lucide-react'
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { ClassificationMetrics, Experiment, LearningDraft, ModelDescription, PreprocessingSettings, Study } from '../types'
import { get, number, post } from '../lib/api'
import { configurationSignature, statusNames, validLearning } from '../lib/learning'
import { missingNames, scalerNames, splitLabel } from '../lib/preprocessing'
import { ErrorState, Loading, Notice, PythonCode, Quiz } from '../components/UI'
import './learning.css'

type Props = {
  step: number
  settings: PreprocessingSettings
  onSettings: (settings: PreprocessingSettings) => void
  draft: LearningDraft
  onDraft: (draft: LearningDraft) => void
  activeId: string
  onActive: (id: string) => void
  navigate: (step: number) => void
}

function NumericField({ id, label, value, min, max, step = 1, change }: {
  id: string; label: string; value: number; min: number; max: number; step?: number
  change: (value: number) => void
}) {
  return <div className="field"><label htmlFor={id}>{label}</label><input id={id} type="number" min={min} max={max} step={step} value={value} onChange={(event) => change(Number(event.target.value))} /></div>
}

function ExportLinks({ run }: { run: Experiment }) {
  const reportName = run.test_metrics ? 'Relatório final' : 'Relatório parcial'
  return <>
    {!run.test_metrics && <p className="caption">Relatório parcial: a tabela de Precision, Recall, F1-score, Support, Acurácia, Média macro, Média ponderada e a matriz de confusão usam os resultados de validação. Para obter os resultados do teste, selecione o modelo pela validação na etapa 8, conclua a avaliação na etapa 9 e exporte novamente.</p>}
    <div className="export-links" aria-label="Exportar experimento">
      <Download size={16} /><strong>Exportar</strong>{(['html', 'markdown', 'json', 'csv', 'python'] as const).map((format) => <a key={format} href={`/api/experiments/${run.id}/export?format=${format}`} download>{({ html: `${reportName} HTML`, markdown: `${reportName} Markdown`, json: 'JSON', csv: 'Histórico CSV', python: 'Python' })[format]}</a>)}
    </div>
  </>
}

function LearningCurves({ runs }: { runs: Experiment[] }) {
  return <div className="learning-curves"><LearningCurve runs={runs} metric="accuracy" /><LearningCurve runs={runs} metric="loss" /></div>
}

function LearningCurve({ runs, metric }: { runs: Experiment[]; metric: 'accuracy' | 'loss' }) {
  const colors = ['#315ddb', '#107b73', '#a34a30', '#7a48ac']
  const length = Math.max(0, ...runs.map((run) => run.history.length))
  if (!length) return <div className="empty-learning"><Network size={25} /><h3>Aguardando a primeira época</h3><p>As curvas aparecem somente quando o backend registrar resultados reais.</p></div>
  const points = Array.from({ length }, (_, index) => {
    const point: Record<string, number | null> = { epoch: index + 1 }
    runs.forEach((run) => {
      point[`${run.id}_train`] = run.history[index]?.[metric] ?? null
      point[`${run.id}_val`] = run.history[index]?.[metric === 'loss' ? 'val_loss' : 'val_accuracy'] ?? null
    })
    return point
  })
  return <>
    <h3>Curva de aprendizado — {metric === 'accuracy' ? 'Acurácia × Épocas' : 'Loss × Épocas'}</h3>
    <div className="learning-chart" role="img" aria-label={`Curvas reais de ${metric === 'loss' ? 'perda' : 'acurácia'} de ${runs.length} experimento(s), até ${length} épocas. Linha contínua: treino; tracejada: validação.`}>
      <ResponsiveContainer width="100%" height={320}><LineChart data={points} margin={{ top: 14, right: 20, left: 0, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 4" stroke="#e5eaf2" /><XAxis dataKey="epoch" allowDecimals={false} label={{ value: 'Época', position: 'bottom', fontSize: 11 }} tick={{ fontSize: 10 }} />
        <YAxis domain={metric === 'accuracy' ? [0, 1] : ['auto', 'auto']} label={{ value: metric === 'accuracy' ? 'Acurácia (%)' : 'Loss', angle: -90, position: 'insideLeft', fontSize: 10 }} tick={{ fontSize: 10 }} tickFormatter={(value: number) => metric === 'accuracy' ? `${number(value * 100, 0)}%` : number(value, 3)} width={70} />
        <Tooltip formatter={(value) => metric === 'accuracy' ? `${number(Number(value) * 100)}%` : number(Number(value), 6)} labelFormatter={(label) => `Época ${label}`} />
        <Legend wrapperStyle={{ fontSize: 10 }} verticalAlign="top" />
        {runs.map((run, index) => run.saved_epoch && <ReferenceLine key={run.id} x={run.saved_epoch} stroke={colors[index % colors.length]} strokeDasharray="3 4" label={{ value: `Pesos salvos: ${run.saved_epoch}`, fontSize: 10, position: 'insideTopRight' }} />)}
        {runs.flatMap((run, index) => [<Line key={`${run.id}_train`} dataKey={`${run.id}_train`} name={`${run.config.name} · treino`} stroke={colors[index % colors.length]} strokeWidth={2} dot={length <= 3} isAnimationActive={false} />, <Line key={`${run.id}_val`} dataKey={`${run.id}_val`} name={`${run.config.name} · validação`} stroke={colors[index % colors.length]} strokeWidth={2} strokeDasharray="6 4" dot={length <= 3} isAnimationActive={false} />])}
      </LineChart></ResponsiveContainer>
    </div>
    <p className="caption">Cada ponto é uma época daquele experimento. Loss inclui a penalização L2 quando habilitada. A tabela de validação também mostra a perda preditiva sem penalização.</p>
    {metric === 'loss' && runs.map((run) => {
      const last = run.history.at(-1)
      if (!last || run.history.length < 5) return null
      const previousBest = Math.min(...run.history.slice(0, -2).map((epoch) => epoch.val_loss))
      const overfitting = last.loss < run.history[0].loss && last.val_loss > previousBest * 1.08
      const lowBoth = last.accuracy < 0.6 && last.val_accuracy < 0.6
      const recent = run.history.slice(-3).map((epoch) => epoch.val_predictive_loss)
      const stable = Math.max(...recent) - Math.min(...recent) < 0.005
      if (overfitting) return <Notice key={run.id} tone="amber"><strong>{run.config.name}: possível overfitting.</strong> O treino melhorou, mas a validação se afastou do melhor valor. Investigue a curva; isso não é um diagnóstico definitivo.</Notice>
      if (lowBoth) return <Notice key={run.id} tone="amber"><strong>{run.config.name}: vale investigar underfitting.</strong> Acurácias de treino e validação abaixo de 60% são um sinal heurístico nesta base balanceada; escala, limiar e tempo de treino também podem explicar o resultado.</Notice>
      if (stable) return <Notice key={run.id}><strong>{run.config.name}: possível estabilização.</strong> As três últimas perdas preditivas de validação variaram menos de 0,005. Isso não prova convergência nem garante bom desempenho.</Notice>
      return null
    })}
  </>
}

function MetricCards({ metrics, title }: { metrics: ClassificationMetrics; title: string }) {
  return <section className="metric-cards" aria-label={title}>{(['accuracy', 'precision', 'recall', 'f1', 'specificity'] as const).map((key) => <div key={key}><span>{{ accuracy: 'Acurácia', precision: 'Precisão', recall: 'Recall', f1: 'F1', specificity: 'Especificidade' }[key]}</span><strong>{number(metrics[key] * 100)}%</strong></div>)}</section>
}

function RunResults({ run }: { run: Experiment }) {
  return <section className="panel run-results">
    <div className="panel-heading"><div><h2>{run.config.name}</h2><p>Experimento {run.id.slice(0, 8)} · configuração original preservada.</p></div><span className={`tag status-${run.status}`}>{statusNames[run.status]}</span></div>
    <div className="run-progress" role="status">{run.completed_epochs} de {run.config.training.epochs} épocas registradas{run.status === 'completed' && run.completed_epochs < run.config.training.epochs ? ' · early stopping' : ''}</div>
    <progress aria-label="Épocas concluídas" value={run.completed_epochs} max={run.config.training.epochs} />
    {run.error && <Notice tone="amber">{run.error}</Notice>}
    <LearningCurves runs={[run]} />
    {run.validation_metrics && <>
      <MetricCards metrics={run.validation_metrics} title="Métricas de validação do modelo salvo" />
      <p className="caption">Validação do modelo salvo · loss {number(run.validation_metrics.loss, 6)} · perda preditiva {number(run.validation_metrics.predictive_loss, 6)} · melhor época observada {run.best_epoch} · pesos salvos da época {run.saved_epoch}.</p>
      <p className="caption">Seed do split: {run.config.preprocess.seed} · seed do treino: {run.config.training.seed} · {number(run.model_summary?.total_parameters, 0)} parâmetros · duração medida: {number(run.duration_seconds)} s.</p>
      <ExportLinks run={run} />
    </>}
  </section>
}

function FinalResult({ result, run }: { result: ClassificationMetrics; run?: Experiment }) {
  const formulas = [
    ['Acurácia', '(TP + TN) / total', `${result.tp} + ${result.tn}`, result.samples, result.accuracy],
    ['Precisão', 'TP / (TP + FP)', result.tp, result.tp + result.fp, result.precision],
    ['Recall', 'TP / (TP + FN)', result.tp, result.tp + result.fn, result.recall],
    ['Especificidade', 'TN / (TN + FP)', result.tn, result.tn + result.fp, result.specificity],
    ['F1', '2TP / (2TP + FP + FN)', 2 * result.tp, 2 * result.tp + result.fp + result.fn, result.f1],
  ] as const
  return <>
    {run && <section className="panel"><div className="panel-heading"><div><h2>Resumo do desempenho do modelo salvo</h2><p>Divisão {splitLabel(run.config.preprocess)}% · seed {run.config.preprocess.seed} · pesos da época {run.saved_epoch}.</p></div></div>
      <div className="table-scroll"><table className="data-table"><thead><tr><th scope="col">Conjunto</th><th scope="col">Accuracy</th><th scope="col">Loss</th><th scope="col">Perda preditiva</th><th scope="col">Amostras</th></tr></thead><tbody>
        {([['Treino', run.train_metrics, run.split_counts.train], ['Validação', run.validation_metrics, run.split_counts.val], ['Teste', result, run.split_counts.test]] as const).map(([label, metrics, count]) => <tr key={label}><th scope="row">{label}</th><td>{metrics ? `${number(metrics.accuracy * 100)}%` : 'Não disponível'}</td><td>{metrics?.loss == null ? 'Não disponível' : number(metrics.loss, 6)}</td><td>{metrics ? number(metrics.predictive_loss, 6) : 'Não disponível'}</td><td>{number(count, 0)}</td></tr>)}
      </tbody></table></div><p className="caption">Loss inclui L2 quando habilitada; perda preditiva não inclui penalização. O resumo usa os pesos salvos; as curvas mostram as medições durante cada época.</p>
      <LearningCurves runs={[run]} />
    </section>}
    <section className="panel"><div className="panel-heading"><div><h2>Relatório de classificação — teste final</h2><p>Métricas por classe e médias, calculadas no teste reservado.</p></div></div>
      {result.classification_report ? <div className="table-scroll"><table className="data-table"><thead><tr><th scope="col">Categoria / Métrica</th><th scope="col">Precision</th><th scope="col">Recall</th><th scope="col">F1-score</th><th scope="col">Support</th></tr></thead><tbody>
        {(['0', '1', 'macro avg', 'weighted avg'] as const).map((key) => { const row = result.classification_report![key]; return <tr key={key}><th scope="row">{key === '0' || key === '1' ? `Classe ${key}` : key === 'macro avg' ? 'Média macro' : 'Média ponderada'}</th><td>{number(row.precision * 100)}%</td><td>{number(row.recall * 100)}%</td><td>{number(row['f1-score'] * 100)}%</td><td>{number(row.support, 0)}</td></tr> })}
      </tbody></table></div> : <p>Relatório por classe não disponível neste experimento antigo.</p>}
      <p className="caption">Support: quantidade real de amostras. Média macro: média simples das classes. Média ponderada: média ponderada pelo support. Divisões por zero são apresentadas como 0.</p>
      <p>Loss no teste: {result.loss == null ? 'Não disponível' : number(result.loss, 6)} · Perda preditiva: {number(result.predictive_loss, 6)}.</p>
    </section>
    <section className="panel"><div className="panel-heading"><div><h2>Matriz de confusão do teste final</h2><p>{number(result.samples, 0)} amostras · positivo = classe 1 · limiar {result.threshold}.</p></div><span className="tag green-tag">Resultado real</span></div>
      <div className="confusion-scroll"><table className="confusion-matrix"><caption>Linhas: classe real · colunas: classe prevista</caption><thead><tr><th scope="col">Real ↓ / Previsto →</th><th scope="col">Classe 0</th><th scope="col">Classe 1</th></tr></thead><tbody><tr><th scope="row">Classe 0</th><td className="correct" title="Negativos classificados corretamente"><strong>{number(result.tn, 0)}</strong><span>TN · verdadeiro negativo</span></td><td className="incorrect" title="Classe 0 prevista como 1"><strong>{number(result.fp, 0)}</strong><span>FP · falso positivo</span></td></tr><tr><th scope="row">Classe 1</th><td className="incorrect" title="Classe 1 prevista como 0"><strong>{number(result.fn, 0)}</strong><span>FN · falso negativo</span></td><td className="correct" title="Positivos classificados corretamente"><strong>{number(result.tp, 0)}</strong><span>TP · verdadeiro positivo</span></td></tr></tbody></table></div>
      <MetricCards metrics={result} title="Métricas finais do teste" />
      <div className="table-scroll" tabIndex={0} role="region" aria-label="Fórmulas das métricas finais"><table className="data-table"><thead><tr><th scope="col">Métrica</th><th scope="col">Fórmula</th><th scope="col">Cálculo</th><th scope="col">Resultado</th></tr></thead><tbody>{formulas.map(([name, formula, numerator, denominator, value]) => <tr key={name}><th scope="row">{name}</th><td>{formula}</td><td>({numerator}) / {denominator}</td><td>{number(value * 100)}%</td></tr>)}</tbody></table></div>
      <p className="caption">Quando o denominador é zero, a métrica é apresentada como 0 e identificada como indefinida. {result.undefined_metrics.length > 0 ? `Métricas indefinidas: ${result.undefined_metrics.join(', ')}.` : 'Nenhuma métrica ficou indefinida neste resultado.'} Classes e unidades físicas não têm significado de domínio informado.</p>
    </section>
  </>
}

export default function Learning({ step, settings, onSettings, draft, onDraft, activeId, onActive, navigate }: Props) {
  const client = useQueryClient()
  const [compared, setCompared] = useState<string[]>([])
  const [criterion, setCriterion] = useState<'loss' | 'predictive_loss'>('loss')
  const [confirmed, setConfirmed] = useState(false)
  const valid = validLearning(draft)
  const config = { ...draft, preprocess: settings }
  const runsQuery = useQuery({ queryKey: ['experiments'], queryFn: ({ signal }) => get<Experiment[]>('/experiments', signal), refetchInterval: (query) => query.state.data?.some((run) => ['queued', 'running'].includes(run.status)) ? 1000 : false })
  const studyQuery = useQuery({ queryKey: ['study'], queryFn: ({ signal }) => get<Study>('/study', signal), refetchInterval: 2500 })
  const modelQuery = useQuery({ queryKey: ['model-preview', config], queryFn: ({ signal }) => post<ModelDescription>('/model/preview', config, signal), enabled: valid && [6, 7, 8].includes(step) })
  const runs = runsQuery.data ?? []
  const study = studyQuery.data
  const active = runs.find((run) => run.id === activeId) ?? runs[0]
  const selected = runs.find((run) => run.id === study?.selected_experiment_id)
  const comparisonRuns = runs.filter((run) => compared.includes(run.id)).slice(0, 4)
  const dirty = active && configurationSignature({ ...active.config, name: '' }) !== configurationSignature({ ...config, name: '' })

  function refresh() {
    void client.invalidateQueries({ queryKey: ['experiments'] })
    void client.invalidateQueries({ queryKey: ['study'] })
  }
  const trainMutation = useMutation({ mutationFn: () => post<Experiment>('/experiments', config), onSuccess: (run) => { onActive(run.id); refresh(); navigate(7) } })
  const selectMutation = useMutation({ mutationFn: (id: string) => post<Study>('/study/select', { experiment_id: id }), onSuccess: () => { setConfirmed(false); refresh(); navigate(9) } })
  const evaluateMutation = useMutation({ mutationFn: () => post<Study>('/evaluation/final', { experiment_id: study?.selected_experiment_id, confirm: true }), onSuccess: () => { setConfirmed(false); refresh() } })

  function modelChange(patch: Partial<LearningDraft['model']>) { onDraft({ ...draft, model: { ...draft.model, ...patch } }) }
  function trainingChange(patch: Partial<LearningDraft['training']>) { onDraft({ ...draft, training: { ...draft.training, ...patch } }) }
  function copyRun(run: Experiment) {
    onSettings({ ...run.config.preprocess })
    onDraft({ name: `${run.config.name.slice(0, 85)} · cópia`, model: { ...run.config.model, hidden_layers: [...run.config.model.hidden_layers] }, training: { ...run.config.training }, threshold: run.config.threshold })
    onActive(run.id)
    navigate(6)
  }
  const titles = { 6: ['CONSTRUIR', 'Desenhe a sua rede neural'], 7: ['APRENDER', 'Acompanhe o aprendizado real'], 8: ['COMPARAR', 'Escolha pelas evidências'], 9: ['AVALIAR', 'Uma prova final, uma escolha'], 10: ['REGISTRAR', 'Seu laboratório de experimentos'] } as Record<number, string[]>
  const trainButton = <><button className="button primary train-button" disabled={!valid || trainMutation.isPending || study?.status !== 'open'} onClick={() => trainMutation.mutate()}><Play size={16} />{trainMutation.isPending ? 'Enviando…' : 'Treinar novo experimento'}</button>{trainMutation.isError && <ErrorState error={trainMutation.error} retry={() => trainMutation.mutate()} />}</>
  return <>
    <div className="step-intro"><span className="eyebrow">{step === 10 ? 'EXPERIMENTOS' : `ETAPA ${String(step).padStart(2, '0')} / 09`} · {titles[step][0]}</span><h1>{titles[step][1]}<span className="title-dot">.</span></h1><p>{step === 6 ? 'Camadas e neurônios mudam a capacidade da MLP. A saída sigmoid estima um score da classe 1.' : step === 9 ? 'Selecione pela validação antes de abrir o teste reservado. O resultado final encerra este ciclo de escolha.' : 'Configurações e resultados pertencem a experimentos imutáveis, salvos localmente.'}</p></div>
    {studyQuery.isError && <ErrorState error={studyQuery.error} retry={() => void studyQuery.refetch()} />}
    {study?.preprocess && <Notice><LockKeyhole size={15} /><strong>Divisão reservada neste ciclo:</strong> seed {study.preprocess.seed}, {splitLabel(study.preprocess)}, coorte {study.preprocess.missing_strategy === 'drop' ? 'completa (remoção)' : 'sem remoção'}. Novos treinos devem manter essas partições. {study.status !== 'open' && 'O teste já foi aberto; novos treinos e trocas de seleção estão encerrados.'}</Notice>}
    {[6, 7, 8].includes(step) && <>
      <section className="panel learning-controls"><div className="panel-heading"><div><h2>Rascunho do próximo experimento</h2><p>{missingNames[settings.missing_strategy]} · {scalerNames[settings.scaler]} · seed do split {settings.seed}. Altere a preparação nas etapas 3–5.</p></div></div>
        <div className="field"><label htmlFor="experiment-name">Nome do experimento</label><input id="experiment-name" maxLength={100} value={draft.name} onChange={(event) => onDraft({ ...draft, name: event.target.value })} /></div>
        {step === 6 && <>
          <div className="layer-controls">{draft.model.hidden_layers.map((units, index) => <div className="layer-row" key={index}><NumericField id={`layer-${index}`} label={`Neurônios da camada ${index + 1}`} value={units} min={4} max={256} change={(value) => modelChange({ hidden_layers: draft.model.hidden_layers.map((n, i) => i === index ? value : n) })} /><button className="icon-button" aria-label={`Remover camada ${index + 1}`} disabled={draft.model.hidden_layers.length === 1} onClick={() => modelChange({ hidden_layers: draft.model.hidden_layers.filter((_, i) => index !== i) })}><Trash2 size={15} /></button></div>)}</div>
          <button className="button secondary" disabled={draft.model.hidden_layers.length >= 4} onClick={() => modelChange({ hidden_layers: [...draft.model.hidden_layers, 16] })}><Plus size={15} />Adicionar camada</button>
          <div className="field activation-field"><label htmlFor="activation">Ativação das camadas ocultas</label><select id="activation" value={draft.model.activation} onChange={(event) => modelChange({ activation: event.target.value as 'relu' | 'tanh' })}><option value="relu">ReLU</option><option value="tanh">tanh</option></select></div>
        </>}
        {step === 7 && <div className="learning-fields">
          <div className="field"><label htmlFor="optimizer">Otimizador</label><select id="optimizer" value={draft.model.optimizer} onChange={(event) => modelChange({ optimizer: event.target.value as LearningDraft['model']['optimizer'] })}><option value="adam">Adam</option><option value="sgd">SGD</option><option value="rmsprop">RMSprop</option></select></div>
          <NumericField id="learning-rate" label="Learning rate" value={draft.model.learning_rate} min={0.000001} max={0.1} step={0.0001} change={(learning_rate) => modelChange({ learning_rate })} />
          <NumericField id="epochs" label="Épocas solicitadas" value={draft.training.epochs} min={1} max={200} change={(epochs) => trainingChange({ epochs, patience: Math.max(1, Math.min(draft.training.patience, epochs - 1)), early_stopping: epochs > 1 && draft.training.early_stopping })} />
          <NumericField id="batch-size" label="Batch size" value={draft.training.batch_size} min={8} max={256} change={(batch_size) => trainingChange({ batch_size })} />
          <NumericField id="training-seed" label="Seed do treinamento" value={draft.training.seed} min={0} max={2147483647} change={(seed) => trainingChange({ seed })} />
          <NumericField id="threshold" label="Limiar de classificação" value={draft.threshold} min={0.01} max={0.99} step={0.01} change={(threshold) => onDraft({ ...draft, threshold })} />
        </div>}
        {step === 8 && <div className="learning-fields"><label className="check-field"><input type="checkbox" checked={draft.training.early_stopping} disabled={draft.training.epochs <= 1} onChange={(event) => trainingChange({ early_stopping: event.target.checked })} />Early stopping · restaurar melhores pesos</label><NumericField id="patience" label="Patience (épocas sem melhoria)" value={draft.training.patience} min={1} max={Math.max(1, draft.training.epochs - 1)} change={(patience) => trainingChange({ patience })} /><NumericField id="l2" label="Regularização L2" value={draft.model.l2} min={0} max={0.1} step={0.0001} change={(l2) => modelChange({ l2 })} /><p>Monitor: val_loss. {draft.training.epochs} épocas solicitadas. Volte à etapa 7 para mudar épocas, batch ou learning rate.</p></div>}
        <p className="caption">Arquitetura [{draft.model.hidden_layers.join(', ')}] · {draft.model.activation} · {draft.model.optimizer} · LR {draft.model.learning_rate} · L2 {draft.model.l2} · early stopping {draft.training.early_stopping ? `ativo, patience ${draft.training.patience}` : 'desligado'}. Execução local em CPU; seeds não garantem determinismo perfeito entre equipamentos.</p>
        {!valid && <Notice tone="amber">Confira os limites dos campos. Com early stopping, patience precisa ser menor que o número de épocas.</Notice>}
        {step !== 6 && trainButton}
      </section>
      {dirty && <Notice tone="amber">O rascunho mudou. Os resultados exibidos abaixo pertencem a <strong>{active.config.name}</strong> e à configuração salva desse treino. Inicie um novo experimento para medir as alterações.</Notice>}
    </>}
    {step === 6 && <>
      {modelQuery.isPending && valid ? <Loading /> : modelQuery.isError ? <ErrorState error={modelQuery.error} retry={() => void modelQuery.refetch()} /> : modelQuery.data && valid && <section className="panel"><div className="panel-heading"><div><h2>Da entrada à probabilidade</h2><p>Dimensão calculada pelo pipeline real: {modelQuery.data.input_features} atributos de entrada.</p></div><Network size={20} /></div><div className="network-diagram"><div><strong>{modelQuery.data.input_features}</strong><span>Entrada</span></div>{modelQuery.data.layers.map((layer, index) => <div key={layer.name}><strong>{layer.units}</strong><span>{index === modelQuery.data.layers.length - 1 ? 'Saída sigmoid' : `${layer.activation} · oculta ${index + 1}`}</span></div>)}</div><p className="caption">Cada bloco representa uma camada; o número indica seus neurônios. Uma única saída sigmoid é suficiente para prever a classe 1 versus a classe 0.</p><div className="parameter-total"><strong>{number(modelQuery.data.total_parameters, 0)}</strong> parâmetros treináveis</div><div className="table-scroll" tabIndex={0} role="region" aria-label="Resumo da arquitetura"><table className="data-table"><thead><tr><th scope="col">Camada</th><th scope="col">Entrada</th><th scope="col">Neurônios</th><th scope="col">Ativação</th><th scope="col">Parâmetros</th></tr></thead><tbody>{modelQuery.data.layers.map((layer) => <tr key={layer.name}><th scope="row">{layer.name}</th><td>{layer.inputs}</td><td>{layer.units}</td><td>{layer.activation}</td><td>{number(layer.parameters, 0)}</td></tr>)}</tbody></table></div><p className="caption">Por camada densa: (entradas + 1 bias) × neurônios. O treinador verifica essa contagem com model.count_params() do Keras.</p></section>}
      <Quiz question="Por que uma saída sigmoid basta para duas classes?" options={['Ela produz um score para a classe 1', 'Cada classe exige uma camada oculta']} correct={0} explanation="O limiar converte o score em classe 0 ou 1. O score não é uma certeza sobre o resultado." />
    </>}
    {step === 7 && <>
      {runs.length > 0 && <div className="field active-run-selector"><label htmlFor="active-run">Experimento exibido</label><select id="active-run" value={active?.id ?? ''} onChange={(event) => onActive(event.target.value)}>{runs.map((run) => <option key={run.id} value={run.id}>{run.config.name} · {statusNames[run.status]} · {run.id.slice(0, 8)}</option>)}</select></div>}
      {runsQuery.isPending ? <Loading /> : runsQuery.isError ? <ErrorState error={runsQuery.error} retry={() => void runsQuery.refetch()} /> : active ? <RunResults run={active} /> : <section className="panel empty-learning"><Play size={28} /><h2>Seu primeiro treino começa aqui</h2><p>Escolha os parâmetros e clique em “Treinar novo experimento”. As curvas serão produzidas pelo Keras.</p></section>}
      <Quiz question="Acurácia alta no treino garante bom desempenho em novos dados?" options={['Sim, basta medir o treino', 'Não, preciso observar a validação']} correct={1} explanation="O modelo pode memorizar o treino. A validação ajuda a observar a capacidade de generalizar sem abrir o teste." />
    </>}
    {[8, 10].includes(step) && <>
      <section className="panel experiments-panel"><div className="panel-heading"><div><h2>Experimentos salvos</h2><p>Selecione até quatro curvas para comparar. Cada linha mantém sua configuração original.</p></div><button className="icon-button" aria-label="Atualizar experimentos" onClick={refresh}><RefreshCw size={16} /></button></div>
        <div className="field ranking-field"><label htmlFor="ranking">Ordenar pela validação</label><select id="ranking" value={criterion} onChange={(event) => setCriterion(event.target.value as typeof criterion)}><option value="loss">val_loss do modelo salvo (inclui L2)</option><option value="predictive_loss">Perda preditiva (sem L2)</option></select></div>
        {runsQuery.isPending ? <Loading /> : runsQuery.isError ? <ErrorState error={runsQuery.error} retry={() => void runsQuery.refetch()} /> : !runs.length ? <div className="empty-learning"><FlaskConical size={28} /><h3>Nenhum experimento salvo ainda</h3><p>Treine na etapa 7 para comparar evidências reais.</p><button className="button primary" onClick={() => navigate(7)}>Ir ao treinamento<ArrowRight size={15} /></button></div> : <div className="table-scroll" tabIndex={0} role="region" aria-label="Tabela de experimentos"><table className="data-table experiment-table"><thead><tr><th scope="col">Comparar</th><th scope="col">Experimento / configuração</th><th scope="col">Estado</th><th scope="col">val_loss</th><th scope="col">Perda preditiva</th><th scope="col">Acurácia val.</th><th scope="col">Ações</th></tr></thead><tbody>{[...runs].sort((a, b) => (a.validation_metrics?.[criterion] ?? Infinity) - (b.validation_metrics?.[criterion] ?? Infinity)).map((run) => <tr key={run.id}><td><input type="checkbox" aria-label={`Comparar ${run.config.name}`} checked={compared.includes(run.id)} disabled={!run.history.length || (!compared.includes(run.id) && compared.length >= 4)} onChange={(event) => setCompared(event.target.checked ? [...compared, run.id] : compared.filter((id) => id !== run.id))} /></td><th scope="row"><strong>{run.config.name}</strong><small>[{run.config.model.hidden_layers.join(', ')}] · {run.config.model.optimizer} · L2 {run.config.model.l2}</small><small>{missingNames[run.config.preprocess.missing_strategy]} · {scalerNames[run.config.preprocess.scaler]} · seed {run.config.preprocess.seed} · {run.completed_epochs} épocas</small></th><td>{statusNames[run.status]}</td><td>{run.validation_metrics ? number(run.validation_metrics.loss, 6) : 'Pendente'}</td><td>{run.validation_metrics ? number(run.validation_metrics.predictive_loss, 6) : 'Pendente'}</td><td>{run.validation_metrics ? `${number(run.validation_metrics.accuracy * 100)}%` : 'Pendente'}</td><td><div className="run-actions"><button onClick={() => { onActive(run.id); navigate(7) }}>Ver treino</button><button onClick={() => copyRun(run)}>Usar como base</button><button disabled={run.status !== 'completed' || study?.status !== 'open' || selectMutation.isPending} onClick={() => selectMutation.mutate(run.id)}>{study?.selected_experiment_id === run.id ? 'Selecionado para teste' : 'Selecionar para teste'}</button></div></td></tr>)}</tbody></table></div>}
        <p className="caption">Menor perda é melhor. val_loss inclui L2; ao comparar regularizações diferentes, consulte também a perda preditiva sem penalização. Acurácia é do modelo efetivamente salvo, não de uma época escolhida posteriormente. A seleção final é explícita, sempre pela validação.</p>
        {selectMutation.isError && <ErrorState error={selectMutation.error} retry={() => { if (active) selectMutation.mutate(active.id) }} />}
      </section>
      {comparisonRuns.length > 0 && <section className="panel"><div className="panel-heading"><div><h2>Curvas lado a lado</h2><p>{comparisonRuns.length} experimento(s) selecionado(s).</p></div></div>{new Set(comparisonRuns.map((run) => run.partition_key)).size > 1 && <Notice tone="amber">Estas partições diferem; os resultados não formam uma comparação controlada.</Notice>}<LearningCurves runs={comparisonRuns} /></section>}
      {step === 8 && <Quiz question="Reduzir a loss do treino garante melhorar a validação?" options={['Sim, basta treinar por mais épocas', 'Não, pode haver overfitting']} correct={1} explanation="O modelo pode continuar melhorando no treino enquanto perde capacidade de generalizar. Compare as curvas de validação e os pesos efetivamente salvos, mantendo o teste reservado." />}
    </>}
    {step === 9 && <>
      {studyQuery.isPending ? <Loading /> : study?.result ? <><Notice tone="green"><Check size={17} />Teste final concluído e salvo. Consultar ou exportar não reexecuta a avaliação.</Notice><FinalResult result={study.result} run={selected} />{selected && <ExportLinks run={selected} />}</> : <section className="panel final-gate"><LockKeyhole size={34} /><h2>Teste final reservado</h2>{selected ? <><p>Selecionado pela validação: <strong>{selected.config.name}</strong>.</p><MetricCards metrics={selected.validation_metrics!} title="Validação do experimento selecionado" /><p>Serão usadas {number(selected.split_counts.test, 0)} amostras de teste, o pipeline e os pesos salvos deste experimento. A abertura encerra os treinos e a seleção deste ciclo.</p><label className="check-field final-confirmation"><input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} />Confirmo que escolhi este experimento somente pela validação e quero abrir o teste final.</label><button className="button primary" disabled={!confirmed || evaluateMutation.isPending || study?.status === 'evaluating'} onClick={() => evaluateMutation.mutate()}>{evaluateMutation.isPending || study?.status === 'evaluating' ? 'Avaliando no teste…' : 'Avaliar no teste final'}</button>{study?.status === 'evaluation_failed' && <Notice tone="amber">A tentativa anterior não foi concluída. A seleção continua congelada; é possível tentar novamente somente este experimento.</Notice>}{evaluateMutation.isError && <ErrorState error={evaluateMutation.error} retry={() => evaluateMutation.mutate()} />}</> : <><p>Compare os experimentos na etapa 8 e selecione um modelo concluído pela validação.</p><button className="button primary" onClick={() => navigate(8)}>Comparar experimentos<ArrowRight size={16} /></button></>}</section>}
      <Quiz question="Se aumentam os falsos negativos e TP fica constante, o que acontece com recall?" options={['O recall diminui', 'O recall sempre aumenta']} correct={0} explanation="Recall = TP / (TP + FN). Aumentar FN aumenta o denominador e reduz a proporção de positivos encontrados." />
      <PythonCode code={'import json\nfrom urllib.request import urlopen\n\n# Consulta o resultado já salvo. GET não abre nem reexecuta o teste.\nwith urlopen("http://127.0.0.1:8000/api/evaluation/final") as response:\n    study = json.load(response)\n\nif study["status"] != "completed":\n    print("Teste reservado ou avaliação ainda não concluída.")\nelse:\n    result = study["result"]\n    tn, fp = result["confusion_matrix"][0]\n    fn, tp = result["confusion_matrix"][1]\n    def ratio(a, b):\n        return a / b if b else 0.0\n    print("accuracy:", ratio(tp + tn, tp + tn + fp + fn))\n    print("precision:", ratio(tp, tp + fp))\n    print("recall:", ratio(tp, tp + fn))\n    print("specificity:", ratio(tn, tn + fp))\n    print("F1:", ratio(2 * tp, 2 * tp + fp + fn))'} />
    </>}
    {[6, 7, 8].includes(step) && modelQuery.data?.python_code && <PythonCode code={modelQuery.data.python_code} />}
    {step === 10 && active?.validation_metrics && <ExportLinks run={active} />}
    <div className="step-footer"><button className="button secondary" onClick={() => navigate(step === 10 ? 8 : step - 1)}><ArrowLeft size={16} />{step === 6 ? 'Preparação dos dados' : step === 7 ? 'Construção da MLP' : step === 8 || step === 10 ? 'Treinos e comparação' : 'Voltar à validação'}</button>{step < 9 && <button className="button primary" onClick={() => navigate(step + 1)}>{step === 6 ? 'Configurar treinamento' : step === 7 ? 'Validar e comparar' : 'Avaliação final'}<ArrowRight size={16} /></button>}</div>
  </>
}
