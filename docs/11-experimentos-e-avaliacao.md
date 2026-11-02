# API implementada de experimentos e avaliação — versão 1.0

Este documento descreve o contrato atual. `docs/03-contrato-api.md` preserva a proposta original; os schemas OpenAPI em `/docs` são a referência executável.

## Criar ou pré-visualizar

`POST /api/model/preview` e `POST /api/experiments` aceitam a configuração abaixo. No primeiro caso, só calculam a arquitetura e o código; no segundo, registram o run e devolvem **202**, com `id`, `status: queued` e configuração original.

```json
{
  "name": "Minha MLP",
  "preprocess": {
    "missing_strategy": "median",
    "scaler": "standard",
    "split_strategy": "70_15_15",
    "seed": 42,
    "knn_neighbors": 5,
    "categorical_strategy": "most_frequent"
  },
  "model": {
    "hidden_layers": [32, 16],
    "activation": "relu",
    "optimizer": "adam",
    "learning_rate": 0.001,
    "l2": 0.0
  },
  "training": {
    "epochs": 50,
    "batch_size": 32,
    "early_stopping": true,
    "patience": 8,
    "seed": 42
  },
  "threshold": 0.5
}
```

Configurações inválidas retornam 422, com mensagem em português e campos rejeitados. Fila cheia, partições incompatíveis ou ciclo já avaliado retornam 409. O ID de criação é `id`, não `experiment_id`.

## Consultar

- `GET /api/experiments`: lista runs, mais recentes primeiro.
- `GET /api/experiments/{id}`: consulta um run; ID inexistente retorna 404.
- Estados: `queued`, `running`, `completed`, `failed`, `interrupted`.
- `history`: épocas reais, com `epoch`, `loss`, `val_loss`, `accuracy`, `val_accuracy`, `predictive_loss`, `val_predictive_loss`.
- `validation_metrics`: só após completar treino e salvar modelo/pipeline. Refere-se aos pesos efetivamente salvos.
- `test_metrics`: `null` até abertura autorizada do teste desse run.
- Metadados: versões, seeds, SHA-256 da fonte, fingerprint das partições, dimensões, nomes de features, contagem de parâmetros, épocas melhor/salva e horários.

Consultas comuns omitem a lista inteira de IDs para reduzir payload. O JSON exportado inclui os IDs ordenados das três partições.

## Selecionar e avaliar

`GET /api/study` e `GET /api/evaluation/final` retornam o ciclo de estudo:

- `status: open`: holdout ainda fechado, podendo haver `selected_experiment_id` pela validação.
- `status: evaluating`: avaliação em andamento e seleção congelada.
- `status: evaluation_failed`: seleção congelada; retry somente do mesmo modelo.
- `status: completed`: resultado persistido em `result`; novos treinos/seleções encerrados.

Seleção explícita:

```http
POST /api/study/select
Content-Type: application/json

{"experiment_id":"ID_DO_RUN_CONCLUIDO"}
```

Abertura final:

```http
POST /api/evaluation/final
Content-Type: application/json

{"experiment_id":"ID_PREVIAMENTE_SELECIONADO","confirm":true}
```

Sem seleção correspondente, enquanto houver trabalhos na fila ou em uma avaliação concorrente: 409. `confirm` ausente/falso: 422. Pedir novamente o mesmo resultado concluído devolve os dados persistidos, sem reexecutar inferência.

Resultado: matriz `[[TN, FP], [FN, TP]]`, acurácia, precisão, recall, F1, especificidade, perda preditiva, limiar, amostras, data e identificação da fonte/run. Métricas com denominador zero retornam 0 e constam em `undefined_metrics`.

## Exportar

`GET /api/experiments/{id}/export?format=...`:

| Formato | Conteúdo |
|---|---|
| `json` | Documento completo do run, configurações, IDs, histórico, métricas e hashes |
| `markdown` | Relatório com configuração, arquitetura, partições, versões, validação, teste e limites |
| `csv` | Histórico real por época |
| `python` | Código da preparação e do treino, com verificação SHA-256 da fonte |

Todos os arquivos usam `Content-Disposition: attachment`. A exportação não dispara treino nem avaliação. O código Python deixa o teste reservado; o teste só é aberto pela ação explícita de avaliação.
