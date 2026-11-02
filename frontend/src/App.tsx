import { lazy, Suspense, useEffect, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ArrowRight, BarChart3, BookOpen, BrainCircuit, Check, ChevronRight, CircleHelp, Database, FlaskConical, GitBranch, Home, Layers3, Lightbulb, LockKeyhole, Network, Play, ScanLine, ShieldCheck, SlidersHorizontal, Sparkles, Target, X } from 'lucide-react'
import type { Profile } from './types'
import { get, number } from './lib/api'
import { loadSettings, missingNames, scalerNames, settingsKey } from './lib/preprocessing'
import { activeRunKey, learningKey, loadActiveRun, loadLearning } from './lib/learning'
import { ErrorState, Loading, Notice } from './components/UI'
import Problem from './features/Problem'

const Explore = lazy(() => import('./features/Explore'))
const Preparation = lazy(() => import('./features/Preparation'))
const Learning = lazy(() => import('./features/Learning'))

const steps = [
  { name: 'Entenda o problema', icon: Target, concept: 'Uma MLP aprende uma relação entre atributos (X) e respostas conhecidas (y). A tarefa é prever target: 0 ou 1.', why: 'Separar atributos, resposta e identificador evita ensinar a rede com informações que não deveria conhecer.', mistake: 'Usar target ou sample_id como entrada da rede.', question: 'O que entra na rede?', answer: 'Dez atributos numéricos e environment. target é a resposta; sample_id identifica a amostra.' },
  { name: 'Explore os dados', icon: BarChart3, concept: 'A análise exploratória (EDA) descreve o que existe na base: distribuição, ausências, extremos e associações.', why: 'Observar escalas e dados faltantes ajuda a formular hipóteses de preparação, sem ajustar nada no teste.', mistake: 'Excluir todo valor negativo ou confundir correlação com causalidade.', question: 'Por que olhar antes de tratar?', answer: 'Porque média, mediana e dispersão contam histórias diferentes. Os gráficos ajudam a reconhecer assimetrias e pontos extremos.' },
  { name: 'Trate os faltantes', icon: ScanLine, concept: 'Valores ausentes podem ser removidos ou estimados por média, mediana ou vizinhos próximos (KNN).', why: 'Cada estratégia altera a informação disponível e deve ser comparada sem vazamento de dados.', mistake: 'Calcular uma mediana com treino e teste juntos.' },
  { name: 'Ajuste as escalas', icon: SlidersHorizontal, concept: 'Escalonamento aproxima as magnitudes dos atributos com StandardScaler ou MinMaxScaler.', why: 'Redes neurais são sensíveis à escala dos valores de entrada.', mistake: 'Ajustar o scaler antes de separar os dados.' },
  { name: 'Separe os dados', icon: GitBranch, concept: 'Treino aprende pesos; validação orienta escolhas; teste estima a generalização final.', why: 'Uma avaliação confiável precisa de dados que não participaram da seleção do modelo.', mistake: 'Escolher hiperparâmetros olhando a acurácia do teste.' },
  { name: 'Construa a MLP', icon: Network, concept: 'Uma MLP combina camadas densas, ativações não lineares e uma saída sigmoid para a classe 1.', why: 'Camadas e neurônios determinam a capacidade e o número de parâmetros da rede.', mistake: 'Achar que mais camadas sempre produzem um modelo melhor.' },
  { name: 'Treine a rede', icon: Play, concept: 'A rede calcula predições e atualiza seus pesos por backpropagation para reduzir a perda.', why: 'Learning rate, batch size, épocas e otimizador mudam a dinâmica do aprendizado.', mistake: 'Confundir acurácia de treino com desempenho em novos dados.' },
  { name: 'Valide e compare', icon: Layers3, concept: 'Curvas de treino e validação ajudam a investigar convergência, underfitting e overfitting.', why: 'Comparar experimentos pela validação permite escolher sem contaminar o teste.', mistake: 'Tratar uma curva isolada como diagnóstico infalível.' },
  { name: 'Avalie o resultado', icon: ShieldCheck, concept: 'A avaliação final usa o teste reservado e uma matriz 2×2 com TN, FP, FN e TP.', why: 'Acurácia, precisão, recall, F1 e especificidade mostram diferentes aspectos dos erros.', mistake: 'Abrir o teste repetidamente para escolher o melhor modelo.' },
]

function HelpPanel({ step, onClose }: { step: number; onClose: () => void }) {
  const content = steps[Math.max(0, Math.min(8, step === 10 ? 7 : step - 1))]
  return <aside className="help-panel" aria-label="Entenda o porquê">
    <div className="help-heading"><span><Lightbulb size={18} />Entenda o porquê</span><button className="icon-button close-help" aria-label="Fechar ajuda" onClick={onClose}><X size={18} /></button></div>
    <div className="help-section"><span className="eyebrow">O CONCEITO</span><h3>{step === 0 ? 'Aprender fazendo' : content.name}</h3><p>{step === 0 ? 'Um experimento começa com uma pergunta, passa por escolhas e termina com evidências. Aqui, você poderá acompanhar cada decisão.' : content.concept}</p></div>
    <div className="help-section"><span className="eyebrow">POR QUE IMPORTA?</span><p>{content.why}</p></div>
    <div className="common-mistake"><span><ShieldCheck size={16} />Erro comum</span><p>{content.mistake}</p></div>
    {content.question && <details className="help-details"><summary>{content.question}<ChevronRight size={16} /></summary><p>{content.answer}</p></details>}
    <div className="help-bottom"><BookOpen size={20} /><p><strong>Do conceito ao código</strong><br />Nas etapas disponíveis, abra “Ver o Python” para reproduzir a análise.</p></div>
  </aside>
}

function Welcome({ profile, start }: { profile: Profile; start: () => void }) {
  return <>
    <section className="welcome-hero"><span className="hero-badge"><Sparkles size={14} />SEU LABORATÓRIO DE APRENDIZAGEM</span>
      <h1>Dos dados à sua<br />primeira <span>rede neural.</span></h1>
      <p>Entenda cada decisão de uma classificação binária. Explore dados reais, descubra padrões e construa conhecimento, uma etapa por vez.</p>
      <button className="button primary" onClick={start}>Começar a explorar<ArrowRight size={18} /></button>
      <div className="hero-note"><ShieldCheck size={15} />Local, em português e com dados reais.</div>
      <div className="hero-network" aria-hidden="true"><svg viewBox="0 0 240 240"><g stroke="currentColor" strokeWidth="1.3">{[55, 120, 185].flatMap((y, i) => [35, 90, 150, 205].map((z, j) => <line key={`${i}-${j}`} x1="35" y1={y} x2="120" y2={z} />))}{[35, 90, 150, 205].map((y, i) => <line key={i} x1="120" y1={y} x2="205" y2="120" />)}</g>{[55, 120, 185].map((y) => <circle key={y} cx="35" cy={y} r="9" />)}{[35, 90, 150, 205].map((y) => <circle key={y} cx="120" cy={y} r="9" />)}<circle cx="205" cy="120" r="13" /></svg></div>
    </section>
    <section className="welcome-stats" aria-label="Resumo da base real"><div><Database size={19} /><strong>{number(profile.rows, 0)}</strong><span>amostras reais</span></div><div><Layers3 size={19} /><strong>{profile.numeric_columns.length + profile.categorical_columns.length}</strong><span>atributos de entrada</span></div><div><Target size={19} /><strong>{Object.keys(profile.target_distribution).length}</strong><span>classes: 0 e 1</span></div></section>
    <section className="journey"><div className="section-heading"><span className="eyebrow">UM CAMINHO, NOVE DESCOBERTAS</span><h2>Aprenda o processo inteiro.</h2><p>Mais do que chegar a um número: entender como você chegou até ele.</p></div>
      <div className="journey-grid"><div><span className="journey-icon"><Database size={22} /></span><small>01 — 05</small><h3>Conheça e prepare</h3><p>Investigue os atributos, encontre ausências e prepare uma base confiável com fit somente no treino.</p><span className="tag green-tag">Dados e preparação reais</span></div><div><span className="journey-icon"><BrainCircuit size={22} /></span><small>06 — 08</small><h3>Construa e experimente</h3><p>Configure sua rede, acompanhe o aprendizado Keras e compare configurações salvas.</p><span className="tag green-tag">Treinamento disponível</span></div><div><span className="journey-icon"><ShieldCheck size={22} /></span><small>09</small><h3>Avalie com evidências</h3><p>Use o teste reservado para entender os acertos, os erros e exportar seu relatório.</p><span className="tag green-tag">Avaliação e exportação</span></div></div>
    </section>
    <Notice><strong>O percurso completo está disponível.</strong> Explore os dados, prepare a entrada, treine uma MLP real e compare pela validação. A avaliação do teste acontece somente depois da seleção explícita de um experimento.</Notice>
    <div className="source-note"><Database size={16} /><span>Fonte: <code>base_classificacao_binaria.csv</code> · {profile.columns} colunas · CSV original preservado</span></div>
  </>
}

function readStep() {
  if (window.location.hash === '#experimentos') return 10
  const match = window.location.hash.match(/^#etapa-([1-9])$/)
  return match ? Number(match[1]) : 0
}

export default function App() {
  const [step, setStep] = useState(readStep)
  const [helpOpen, setHelpOpen] = useState(false)
  const [settings, setSettings] = useState(loadSettings)
  const [draft, setDraft] = useState(loadLearning)
  const [activeId, setActiveId] = useState(loadActiveRun)
  const contentRef = useRef<HTMLElement>(null)
  const query = useQuery({ queryKey: ['profile'], queryFn: ({ signal }) => get<Profile>('/dataset/profile', signal) })
  useEffect(() => {
    try { localStorage.setItem(settingsKey, JSON.stringify(settings)) }
    catch { /* O rascunho continua em memória se o armazenamento estiver indisponível. */ }
  }, [settings])
  useEffect(() => {
    try {
      localStorage.setItem(learningKey, JSON.stringify(draft))
      localStorage.setItem(activeRunKey, activeId)
    } catch { /* Rascunho mantido em memória. Experimentos continuam salvos no backend. */ }
  }, [draft, activeId])
  useEffect(() => {
    const onHash = () => setStep(readStep())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  function navigate(value: number) {
    setStep(value)
    window.location.hash = value === 10 ? 'experimentos' : value ? `etapa-${value}` : 'inicio'
    setHelpOpen(false)
    window.scrollTo({ top: 0, behavior: 'instant' })
    contentRef.current?.focus({ preventScroll: true })
  }
  return <div className="app-shell">
    <a className="skip-link" href="#conteudo" onClick={(event) => { event.preventDefault(); contentRef.current?.focus() }}>Pular para o conteúdo</a>
    <aside className="sidebar"><button className="brand" onClick={() => navigate(0)} aria-label="ANN Lab, início"><span className="brand-mark"><Network size={24} /></span><span>ANN<span className="brand-light">LAB</span><small>APRENDA EXPERIMENTANDO</small></span></button>
      <button className={`home-link ${step === 0 ? 'active' : ''}`} aria-current={step === 0 ? 'page' : undefined} onClick={() => navigate(0)}><Home size={18} />Visão geral</button>
      <div className="nav-label">SEU PERCURSO<span>9 etapas</span></div>
      <nav aria-label="Etapas do laboratório">{steps.map((item, index) => {
        const Icon = item.icon
        return <button key={item.name} className={`step-link ${step === index + 1 ? 'active' : ''}`} aria-current={step === index + 1 ? 'step' : undefined} onClick={() => navigate(index + 1)}>
          <span className="step-number">{String(index + 1).padStart(2, '0')}</span><Icon size={16} /><span>{item.name}</span>
        </button>
      })}</nav>
      <button className={`home-link experiments-link ${step === 10 ? 'active' : ''}`} onClick={() => navigate(10)}><FlaskConical size={17} />Experimentos</button>
      <div className="sidebar-phase"><span className="phase-dot" /><div><strong>ANN Lab · Percurso completo</strong><small>9 etapas disponíveis</small></div></div>
      <div className="sidebar-footer"><FlaskConical size={16} /><span>Um laboratório, não uma caixa-preta.</span></div>
    </aside>
    <div className="app-body"><header className="topbar"><div><span className="breadcrumb">Laboratório<ChevronRight size={14} /></span><strong>Classificação binária</strong></div><div className="topbar-actions"><span className={`connection ${query.isSuccess ? 'online' : ''}`}><i />{query.isSuccess ? 'CSV conectado' : query.isError ? 'API indisponível' : 'Conectando…'}</span><button className="help-toggle" aria-label="Entenda o porquê" aria-expanded={helpOpen} onClick={() => setHelpOpen(!helpOpen)}><CircleHelp size={19} /><span>Entenda o porquê</span></button></div></header>
      <div className="workspace"><main id="conteudo" ref={contentRef} tabIndex={-1} className="main-content" aria-label="Conteúdo da etapa">
        <Suspense fallback={<Loading />}>{query.isPending ? <Loading /> : query.isError ? <ErrorState error={query.error} retry={() => void query.refetch()} /> : step === 0 ? <Welcome profile={query.data} start={() => navigate(1)} /> : step === 1 ? <Problem profile={query.data} next={() => navigate(2)} /> : step === 2 ? <Explore profile={query.data} back={() => navigate(1)} next={() => navigate(3)} /> : step >= 3 && step <= 5 ? <Preparation step={step} profile={query.data} settings={settings} onChange={setSettings} navigate={navigate} /> : <Learning step={step} settings={settings} onSettings={setSettings} draft={draft} onDraft={setDraft} activeId={activeId} onActive={setActiveId} navigate={navigate} />}</Suspense>
      </main><div className={`help-container ${helpOpen ? 'is-open' : ''}`}><HelpPanel step={step} onClose={() => setHelpOpen(false)} /></div></div>
      <footer className="reasoning-bar"><span><GitBranch size={15} /><strong>Sua linha de raciocínio</strong></span><span>{missingNames[settings.missing_strategy]} · {scalerNames[settings.scaler]}</span><span>{settings.split_strategy === '70_15_15' ? '70/15/15' : '64/16/20'} · seed {settings.seed}</span><span><Check size={13} />ID e target fora de X</span><span><LockKeyhole size={13} />Teste final na etapa 9</span></footer>
    </div>
  </div>
}
