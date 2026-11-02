# 02 — Arquitetura de referência

## Stack recomendada

- **UI:** React + TypeScript + Vite, React Router (opcional), TanStack Query para estado de requisições, Tailwind (ou CSS Modules), Recharts, Lucide icons.
- **API e ML:** Python 3.11/3.12 conforme versões suportadas pela instalação, FastAPI, Pydantic, Uvicorn, pandas, NumPy, scikit-learn, TensorFlow/Keras.
- **Persistência:** SQLite local para metadados/experimentos e JSON para histórico por época. Armazenar artefatos de modelo somente se necessário, com limite de tamanho.
- **Qualidade:** pytest + httpx/TestClient, ruff, frontend Vitest + Testing Library e testes E2E Playwright. Nada de SaaS obrigatório.
- **CORS:** liberar somente endereços de desenvolvimento (`localhost`), configurável por `.env`.

## Organização final sugerida (A SER GERADA PELO OPENCODE)

```text
ann-playground-opencode/
├── data/                       # Arquivos originais, somente leitura
├── frontend/
│   ├── package.json
│   ├── src/
│   │   ├── app/
│   │   ├── components/          # StepShell, ExplainPanel, StatCard, Chart...
│   │   ├── features/            # onboarding, eda, missing, scaling, split, mlp...
│   │   ├── lib/api.ts
│   │   └── types/api.ts
│   └── ...
├── backend/
│   ├── pyproject.toml ou requirements.txt
│   ├── app/
│   │   ├── main.py
│   │   ├── api/                 # rotas FastAPI
│   │   ├── schemas/             # validação Pydantic
│   │   ├── services/dataset.py
│   │   ├── services/preprocessing.py
│   │   ├── services/training.py
│   │   ├── services/evaluation.py
│   │   └── db/                  # repositório SQLite
│   └── tests/
├── docs/                        # este kit + decisões
├── scripts/
├── AGENTS.md
└── README.md
```

## Regras de arquitetura

- Carregar o dataset a partir da pasta `data/` do repositório, independente do working directory de execução. Nunca confiar em `../../` hardcoded no navegador.
- `DatasetService` produz estatísticas e gráficos agregados; não precisa enviar todos os 10k registros ao browser. Endpoint de preview paginado, payload compactado.
- `ExperimentService` orquestra pipeline e valida hiperparâmetros; configuração imutável após iniciar um run.
- `Trainer` faz treinamento **fora da thread principal de request**, atualiza progresso a cada época e marca estados (`queued`, `running`, `completed`, `failed`). O GET de status é leve. No MVP, thread worker com fila de 1 trabalho e limites de execução; não iniciar vários treinos TensorFlow simultâneos.
- Persistir runs em SQLite; salvar por época de maneira incremental; ao reiniciar, runs `running` sem processo associado vão para `interrupted`, não para `completed`.
- Quando o usuário muda configuração, a tela mostra **rascunho** com status `dirty`. Um run já salvo continua referindo-se à configuração com que foi executado.
- API com respostas explícitas para erro de input (422), conflito em teste final (409), run inexistente (404) e falha operacional (500 com erro seguro).
- Tempo e recursos: permitir no máximo 200 épocas, 4 camadas ocultas, 256 neurônios/camada e no máximo 1 run concorrente por padrão; aceitar stop/cancel como opcional depois do MVP.
- Experimento local: sem upload arbitrário de código, sem execução de scripts de terceiros e sem compartilhamento público da API. A entrada da API é apenas configuração validada.

## Como executar (objetivo para o OpenCode criar)

```bash
# Terminal 1: backend
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Terminal 2: frontend
cd frontend
npm install
npm run dev
```

Preferir um `make dev` ou scripts de inicialização que imprimam URL do frontend e da API. Os comandos acima são **alvo esperado**, só funcionarão depois que o OpenCode gerar a aplicação.

## O que não fazer

Não usar chamadas do browser ao próprio notebook, não colocar Keras no frontend, não persistir 10.000 linhas duplicadas em SQLite, não gerar estatísticas via Math.random, não instalar banco em nuvem, não solicitar tokens de API.
