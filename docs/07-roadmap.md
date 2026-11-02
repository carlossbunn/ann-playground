# 07 — Sequência incremental para o OpenCode

**Importante:** uma fase deve terminar com funcionalidade testável e demonstrável. É preferível completar um caminho vertical real antes de adicionar acabamento visual elaborado.

## Fase 0 — Preparação (insumos)

1. Conferir arquivo CSV de 10k linhas e PDF de 5 páginas.
2. Criar monorepo `frontend/`, `backend/` com scripts.
3. Documentar versões e comandos de setup.
4. Criar `GET /api/health` testado e pagina principal com layout.

## Fase 1 — Dados e EDA (MVP)

1. Backend lê CSV e retorna dataset profile/preview/EDA/correlação.
2. Frontend tem 9 itens de navegação, com passos futuros exibidos como “a implementar” em vez de telas aparentemente prontas.
3. Etapas 1 e 2 **reais** e funcionais; histogramas, boxplot, correlação e ausentes.
4. Layout com sidebar, painel didático, tooltips, estado de loading/erro.
5. API tests e smoke test; registrar captura do funcionamento.

**Definição de pronto:** EDA muda quando seleciono `sensor_b` ou `measurement_i`; percentuais conferem com CSV.

## Fase 2 — Pré-processamento e split

1. `POST /api/preprocessing/preview` com split e pipelines corretos.
2. Etapa 3 comparação entre 4 estratégias; etapa 4 scaler; etapa 5 visualização da estratificação.
3. UI guarda um rascunho de configuração (não um modelo treinado).
4. Testes leakage, 70/15/15 vs 64/16/20 e coorte reduzida para remoção.

**Definição de pronto:** alterar mediana para média muda dados imputados e nunca produz `fit` usando teste.

## Fase 3 — MLP e treino real

1. Construtor visual da arquitetura (1–4 layers), contador de parâmetros.
2. Job trainer assíncrono simples (1 worker), Keras real, histórico por época e SQLite.
3. Etapas 6 e 7 completas; progress bar baseada nas épocas concluídas; UI consome polling.
4. Teste de treino rápido (2 epochs), erro recuperável e persistência de histórico.

**Definição de pronto:** mudar 32→64 neurônios altera contagem de parâmetros; run Keras grava histórico e métricas verdadeiras.

## Fase 4 — Comparação, validação e avaliação final

1. Etapa 8 com curvas e opções de EarlyStopping/L2 reais.
2. Painel de comparações com 2+ experimentos.
3. Seleção pela validação; teste fechado até confirmar; API avalia uma vez.
4. Etapa 9 matriz 2x2, métricas verificadas, relatório local.

**Definição de pronto:** métricas calculadas de TN/FP/FN/TP conferem; teste não é usado antes da escolha.

## Fase 5 — Polimento

1. Revisar textos pedagógicos e tooltips em todas as telas.
2. Acessibilidade, mobile e vazio/erro; responsividade.
3. E2E com Playwright; reforçar acessibilidade.
4. README de uso, `make dev`, `make test` ou equivalentes.
5. Exibir resumo do que foi aprendido e decisões do experimento.

## Modo de operação do estudante

Após cada fase: abrir a aplicação, explorar a etapa, anotar o que entendeu, pedir ajuste ao OpenCode e testar novamente. Fazer versões pequenas e comparáveis em vez de pedir 20 features de uma vez.
