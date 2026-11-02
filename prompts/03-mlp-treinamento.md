# Prompt para o OpenCode — Fase 3: arquitetura MLP e treinamento real

Inspecione implementação existente; leia `AGENTS.md`, `docs/02-arquitetura.md`, `docs/03-contrato-api.md`, `docs/04-metodologia-ml.md` e critérios de aceite.

Construa etapas 6 e 7. UI controla número de camadas e neurônios, ativação, learning rate, épocas, batch size e otimizador. Diagramas e contagem de parâmetros devem refletir a configuração e a dimensão após OneHotEncoder. Backend cria e treina MLP `keras.Sequential` real com `Dense(1,sigmoid)`, `binary_crossentropy`, validando ranges. Ajustadores exclusivamente no treino.

Crie API de experimentos com POST que devolve job id, fila de execução com no máximo 1 treino concorrente, GET de status/histórico real por época. Persistir configurações e histórico em SQLite. Frontend acompanha epochs e mostra curvas reais com loading/error; não mostrar métricas que ainda não existem. Ao mudar knobs, exigir novo run. Adicione testes rápidos 1–2 epochs, estados e recuperação de erros; execute e mostre resultados.
