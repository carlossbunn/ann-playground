# Prompt para o OpenCode — Fase 4: validação, comparação e teste final

Leia `AGENTS.md`, `docs/04-metodologia-ml.md`, `docs/05-nove-etapas.md` e `docs/06-criterios-de-aceite.md`.

Implemente etapa 8 com gráficos reais train/validation loss e accuracy, early stopping com patience, kernel_regularizer L2, avisos de possíveis overfitting e underfitting explicitamente qualificados como hipóteses, e painel para comparar dois ou mais runs por métricas de validação (alertar para seeds/splits distintos).

Implemente etapa 9 bloqueada até selecionar explicitamente o experimento pela VALIDAÇÃO. API `POST /api/evaluation/final` com confirmação, avaliação holdout usando transformadores treinados apenas no treino, matriz 2x2 corretamente rotulada TN/FP/FN/TP, Accuracy/Precision/Recall/F1/Specificity calculadas de forma consistente. Não permitir mais de uma seleção do holdout no mesmo ciclo de estudo. Calcular métricas reais, executar testes e validar manualmente a matriz. Sem estatísticas inventadas.
