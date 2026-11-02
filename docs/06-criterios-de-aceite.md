# 06 — Critérios de aceite e plano de testes

## A. Integridade dos dados

- [ ] Abre `data/base_classificacao_binaria.csv` sem copiar dados para outra fonte; `rows=10000`, `columns=13`.
- [ ] `target`: 5000 registros 0, 5000 registros 1; `environment`: cold/mild/hot e ausentes.
- [ ] `sample_id` nunca entra no tensor de entrada da MLP.
- [ ] `target` nunca entra nas features.
- [ ] Percentuais de faltantes calculados conferem com `data/perfil_base.json` dentro de tolerância 0.01 p.p.
- [ ] Estatísticas EDA exibem `null`/avisos para valores indefinidos em vez de `NaN` em JSON.
- [ ] Exibir 10 atributos numéricos + 1 categórico original (11 features).

## B. Pré-processamento

- [ ] Remover/mean/median/KNN funcionam, com diferenças reais.
- [ ] Remoção de linhas informa total excluído e mudança de cobertura; divisão feita na coorte reduzida.
- [ ] StandardScaler e MinMaxScaler produzem resultados distintos observáveis.
- [ ] Teste automatizado garante **fit dos transformadores apenas no treino**, nunca teste/val.
- [ ] OneHotEncoder categórico trata desconhecidos e ausências de forma previsível.
- [ ] Proporção 70/15/15 com split estratificado e contagens que somam total.
- [ ] Proporção 80/20 com validação interna efetiva 64/16/20, explicitada na UI.
- [ ] Repetir config + seed mantém seleção de linhas estável.

## C. Treinamento

- [ ] Keras executa treino de verdade e reporta, por época, loss/val_loss e accuracy/val_accuracy.
- [ ] API não bloqueia durante treino: retorna run ID; acompanhamento por polling funciona.
- [ ] Alterar layer/neurônios/LR/optimizer/epoch/batch cria novo experimento e influencia execução real.
- [ ] Validação de ranges impede config inválida, erro legível.
- [ ] Teste que compara shape de entrada com encoding categórico.
- [ ] EarlyStopping e L2 são aplicados realmente, não apenas UI.
- [ ] Salvamento e recuperação de experimentos no SQLite; corrida interrompida não fica “completed” no restart.
- [ ] Um erro durante training retorna `failed` com mensagem útil.

## D. Validação e avaliação final

- [ ] Histórico das curvas deriva do mesmo treino/mesmo ID do experimento.
- [ ] Comparar ao menos 2 runs por `val_loss` e `val_accuracy`; explicar se splits diferentes.
- [ ] Teste permanece bloqueado antes da escolha explícita do experimento.
- [ ] Matriz é **2x2** e apresenta `TN FP / FN TP` corretamente.
- [ ] Accuracy, Precision, Recall, Specificity e F1 conferem com valores recalculados da matriz.
- [ ] Teste não é usado no ajuste de hiperparâmetros, pré-processamento ou seleção do run.
- [ ] Tentativas de nova avaliação final indevida retornam resposta de conflito, sem resultados de treino mascarados.
- [ ] Se for possível exportar relatório, documentar configurações, seed, conjunto, métricas e limites sem alegar significado físico das classes.

## E. UX e acessibilidade

- [ ] Todas as 9 etapas navegáveis e com conteúdo real específico (sem texto genérico repetido).
- [ ] Interface em pt-BR; exibir `0/1` e termos técnicos claros.
- [ ] Estados loading/empty/error reais e elegíveis para retry.
- [ ] Layout desktop e mobile sem rolagem horizontal da página inteira.
- [ ] Teclado e foco visível; labels; gráficos com resumo textual.
- [ ] Bloco “Ver o Python” usa parâmetros da seleção atual e corresponde ao backend.
- [ ] Mudanças pós-treino invalidam resultados exibidos **na configuração editada**, sem apagar resultados imutáveis do run original.
- [ ] Nenhuma métrica ou curva inventada em carga vazia ou indisponibilidade do backend.

## F. Testes essenciais automatizados (código que o OpenCode deve escrever)

1. `test_dataset_profile_matches_csv` — shape, classes, faltantes.
2. `test_pipeline_excludes_id_and_target` — proíbe leakage direto.
3. `test_split_is_stratified_reproducible_and_disjoint` — mesma seed, sem interseção de IDs.
4. `test_preprocessors_fit_train_only` — spies ou abordagem controlada com valores-sentinela no teste.
5. `test_environment_one_hot` — n categorias e faltantes.
6. `test_train_real_mlp_smoke` — 1–2 épocas em subconjunto de treino, curvas numéricas e dimensão correta.
7. `test_training_job_state_transitions` — queued→running→completed/failed.
8. `test_validation_metrics_and_holdout_lock` — avaliação apenas quando autorizada.
9. `test_confusion_matrix_metrics` — casos sintéticos isolados com TP/FP/TN/FN conhecidos.
10. `test_api_bad_config_422` — parâmetros inválidos não iniciam job.
11. `e2e_navigate_all_stages` — abre telas, mostra contexto e dados corretos.
12. `e2e_run_experiment_and_compare` — treino rápido de 2 runs e comparação de resultados reais; marcar como lento.

## Entregas por fase

- **Fase 1:** A + etapas 1–2 + layout + APIs de leitura + testes básicos.
- **Fase 2:** B + etapas 3–5 + testes de leakage.
- **Fase 3:** C + etapas 6–7 + treino real mínimo e persistência.
- **Fase 4:** D + etapas 8–9 + final lock e comparação.
- **Fase 5:** E + testes E2E, polimento e README de execução.
