# Prompt para o OpenCode — Fase 1: MVP de dados e EDA

Leia `AGENTS.md`, `docs/01-produto-e-ux.md`, `docs/02-arquitetura.md`, `docs/03-contrato-api.md`, `docs/05-nove-etapas.md` e `docs/06-criterios-de-aceite.md`.

Implemente e execute agora a Fase 1 descrita em `docs/07-roadmap.md`, em código real. Quero `frontend/` React+TypeScript+Vite e `backend/` FastAPI; não reutilize prévias fictícias. O backend lê `data/base_classificacao_binaria.csv`. Crie rotas `/api/health`, `/api/dataset/profile`, `/api/dataset/preview`, `/api/dataset/eda`, `/api/dataset/correlations`. Garanta que o perfil calculado corresponde a `data/perfil_base.json`.

Crie UI de boas-vindas, sidebar com 9 etapas (as futuras claramente indisponíveis), etapas 1 e 2 completas, histogramas, boxplot, correlação, distribuição categórica, faltantes, explicações contextuais, “Ver o Python”, quizzes curtos. Estilize de forma profissional e responsiva. Escreva testes de backend e frontend, instale dependências compatíveis, execute verificações e forneça os comandos para iniciar backend e frontend. Não entregue apenas mocks ou componentes soltos. Não implemente training ainda se isso comprometer completar a Fase 1.
