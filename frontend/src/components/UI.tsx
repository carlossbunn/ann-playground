import { useId, useState, type ReactNode } from 'react'
import { AlertCircle, Check, ChevronRight, Code2, Lightbulb, RefreshCw } from 'lucide-react'

export function ErrorState({ error, retry }: { error: Error; retry: () => void }) {
  return <div className="error-state" role="alert">
    <AlertCircle size={24} /><h2>Não foi possível carregar</h2><p>{error.message}</p>
    <button className="button secondary" onClick={retry}><RefreshCw size={16} />Tentar novamente</button>
  </div>
}

export function Loading() {
  return <div className="loading" role="status"><span className="spinner" />Calculando a partir do CSV real…</div>
}

export function Notice({ children, tone = 'blue' }: { children: ReactNode; tone?: 'blue' | 'amber' | 'green' }) {
  return <div className={`notice ${tone}`}><Lightbulb size={18} /><div>{children}</div></div>
}

export function PythonCode({ code }: { code: string }) {
  return <details className="code-block">
    <summary><Code2 size={17} /><strong>Ver o Python</strong><span>Reproduza esta análise</span><ChevronRight size={16} /></summary>
    <div className="code-inner"><p>Execute na raiz do projeto, com o ambiente Python do backend ativado.</p>
      <pre tabIndex={0}><code>{code}</code></pre>
    </div>
  </details>
}

export function Quiz({ question, options, correct, explanation }: {
  question: string; options: string[]; correct: number; explanation: string
}) {
  const [answer, setAnswer] = useState<number | null>(null)
  const id = useId()
  return <section className="quiz" aria-labelledby={id}>
    <div className="eyebrow"><Lightbulb size={15} />PAUSA PARA PENSAR</div>
    <h3 id={id}>{question}</h3>
    <div className="quiz-options">{options.map((option, index) =>
      <button key={option} aria-pressed={answer === index}
        className={answer === index ? 'chosen' : ''} onClick={() => setAnswer(index)}>{option}</button>)}</div>
    {answer !== null && <p className="quiz-answer" role="status">
      {answer === correct ? <><Check size={16} />Isso mesmo! </> : 'Vamos revisar: '}{explanation}
    </p>}
  </section>
}
