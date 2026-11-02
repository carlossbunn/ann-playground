# 03 — API proposta (FastAPI)

> É um contrato de implementação sugerido; endpoints podem evoluir mantendo os critérios de aceite. Todos os números em JSON abaixo marcados como `exemplo` são **illustrativos**, não resultados medidos no treino.

## Convenções

- Base URL: `http://localhost:8000/api`.
- JSON com tipos explícitos, mensagens de erro em português, percentuais como 0-100 (documentar), classes de rótulo como 0/1.
- A aplicação precisa distinguir estatísticas **globais (EDA descritiva)** e transformações **fitted somente no treino**.

### `GET /api/health`

```json
{"status":"ok","dataset_loaded":true,"ml_engine":"tensorflow"}
```

### `GET /api/dataset/profile`

Retorna: `{rows, columns, target_distribution, missing_by_column, environment_distribution, numeric_columns, categorical_columns, id_column, label_column}` e `numeric_summary`. Validar contra `data/perfil_base.json`.

### `GET /api/dataset/preview?offset=0&limit=30`

Devolve `columns`, `rows`, `total`, `offset`, `limit`; converter CSV vazio em `null`; limite máximo 100; sem todo dataset no mesmo response.

### `GET /api/dataset/eda?column=sensor_a&bins=20`

Para coluna numérica: histogramas `{bin_start,bin_end,count}`, quartis/boxplot com `q1,median,q3,lower_whisker,upper_whisker,outlier_count` (critério 1.5 IQR); matriz de correlação por endpoint separado. Para coluna categórica: contagens por categoria. `target` pode aparecer em gráficos descritivos agregados, nunca como feature de treinamento.

### `GET /api/dataset/correlations`

Correlações numéricas (Pearson por pares válidos) com `columns`, `matrix` (`number|null`), `method` e `sample_counts` se possível. Excluir `sample_id` e, por padrão, `target` da matriz de features; correlação com `target` pode ser painel separado claramente identificado.

### `POST /api/preprocessing/preview`

Entrada (configuração, nunca pipeline já fitted):

```json
{
  "missing_strategy":"median",
  "scaler":"standard",
  "split_strategy":"70_15_15",
  "seed":42,
  "sample_limit":40
}
```

Resposta esperada:

```json
{
  "dataset_rows":10000,
  "rows_after_filter":10000,
  "removed_rows":0,
  "split_counts":{"train":7000,"val":1500,"test":1500},
  "numeric_before_after":[{"column":"sensor_a","before":[0.0],"after":[0.1]}],
  "missing_before_after":{"sensor_a":{"before":414,"after":0}},
  "fitted_on":"train_only",
  "warnings":[]
}
```

Valores `before/after` no JSON acima são exemplos de forma: **não** usá-los como dados reais. Para `remove`, separar primeiro os registros completos para formar a coorte reduzida, informar cobertura e depois estratificar; a prévia deve refletir esse processo.

### `POST /api/experiments`

Cria run e inicia treino com no máximo um worker em paralelo.

```json
{
  "name":"MLP 32-16, mediana + standard",
  "preprocess":{"missing_strategy":"median","scaler":"standard","split_strategy":"70_15_15","seed":42},
  "model":{"hidden_layers":[32,16],"activation":"relu","optimizer":"adam","learning_rate":0.001,"l2":0.0},
  "training":{"epochs":50,"batch_size":32,"early_stopping":true,"patience":8,"monitor":"val_loss"},
  "threshold":0.5
}
```

Validação Pydantic: `missing_strategy ∈ {drop,mean,median,knn}`, `scaler ∈ {standard,minmax,none}` (`none` é apenas opção didática), `split_strategy ∈ {70_15_15,80_20}`, `hidden_layers` 1–4, neurônios por camada 4–256, epochs 1–200, batch 8–256, learning rate >0 e <=0.1, L2 entre 0 e 0.1, patience < epochs ou tratar explicitamente, seed inteiro limitado. Otimizador `adam|sgd|rmsprop`.

Retorna `202 Accepted` e `{experiment_id,status:"queued"}`; se fila indisponível retornar erro explícito, não sucesso falso.

### `GET /api/experiments` / `GET /api/experiments/{id}`

Inclui configuração original, status, tempos, seed, versões, conjunto de validação e histórico por época (`epoch`, `loss`, `val_loss`, `accuracy`, `val_accuracy`). Status possíveis: `queued`, `running`, `completed`, `failed`, `interrupted`.

Exemplo de um run ainda sem treino concluído:

```json
{"id":"run-id","status":"running","completed_epochs":7,"requested_epochs":50,"history":[],"validation_metrics":null,"test_metrics":null,"error":null}
```

`history` real deve ser preenchido em cada epoch quando possível. **Não incluir métricas de teste** até avaliação final.

### `POST /api/evaluation/final`

Entrada: `{ "experiment_id":"run-id", "confirm":true }`.

Regras:
- run deve estar `completed`.
- avaliar usando conjunto **teste intocado** com imputador/scaler treinados no treino correspondente; threshold do experimento.
- persistir as contagens `tn,fp,fn,tp` e métricas accuracy, precision, recall, f1, specificity (com `zero_division=0` quando necessário e explicação de impacto).
- guardar que teste foi aberto e evitar um segundo experimento escolhido com base nesse resultado no mesmo projeto de estudo; retornar `409` em tentativa indevida.
- mostrar positivo = 1 e eixos `classe real` versus `classe prevista`.
- permitir consultar o resultado já persistido, não reexecutar silenciosamente.

### `GET /api/evaluation/final`

Retorna `{status:"locked"}` ou `{status:"completed",experiment_id,confusion_matrix,metrics,...}`.

## Segurança, integração e performance

- Job runner no backend, não bloquear browser durante múltiplas épocas.
- Usar respostas pequenas para curvas: no máximo 200 pontos por série.
- Não retornar traceback bruto da API ao usuário; log interno com ID da falha.
- CORS restrito a origem local.
- `GET /api/health` mostra falha real se TensorFlow não inicializou: não marcar sucesso fictício.

## Critérios contra resultado artificial

Se endpoint de treino retornar métricas sem iniciar Keras ou usar números fixos, o projeto não atende. Se API sempre retornar mesma curva para configurações diferentes, investigação obrigatória. Execuções com mesma configuração/seed são **aproximadamente** reproduzíveis, não necessariamente bitwise iguais entre hardware/plataforma.
