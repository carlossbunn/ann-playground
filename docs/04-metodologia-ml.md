# 04 — Metodologia correta de ML

## Definição do problema

- `X`: 10 numéricas (`sensor_a`, `sensor_b`, `sensor_c`, `measurement_d` a `measurement_i`, `ratio_j`) + categórica `environment`.
- `y`: `target` binária (0/1).
- `sample_id`: identificador, **nunca predictor**.
- Significado de `0`, `1`, sensores e unidades físicas: **não informado**. Evitar narrativa de “falha”, “doença”, “fraude” sem fonte.

## Ordem de execução sem leakage

1. Inspecionar CSV bruto (EDA descritiva global, sem fit de modelos nem decisões guiadas pelo teste).
2. Escolher estratégia de faltantes e parâmetros; se `drop`, filtrar registros incompletos pelo padrão de ausência das features **antes** da divisão, avisando sobre população alterada e cobertura. Caso contrário manter linhas.
3. Fazer split **estratificado** e reproduzível: 70/15/15 (padrão) ou 80/20. Para 80/20, reservar validação interna estratificada em 20% dos 80% de treino, resultando em **64% treino efetivo, 16% validação, 20% teste**; explicar esse detalhe no UI.
4. Preparar `ColumnTransformer`: numéricas → `SimpleImputer(strategy=mean|median)` ou `KNNImputer` → scaler `StandardScaler|MinMaxScaler` (ou comparação didática de nenhum scaler); categórica → `SimpleImputer(strategy=most_frequent)` ou constante explícita `"missing"` → `OneHotEncoder(handle_unknown="ignore")`.
5. `fit_transform(X_train)`; apenas `transform(X_val)` e, **somente no momento de avaliação final**, `transform(X_test)`. Não usar arrays transformados do teste para tunar nada.
6. MLP real com Keras: entrada = número de features após transformação; `keras.Sequential([...Dense...Dense(1, activation='sigmoid')])`; camadas ocultas `relu` (padrão) ou `tanh`; `kernel_regularizer=l2(...)` se habilitado; optimizer selecionável; `binary_crossentropy`; métricas `BinaryAccuracy`, opcional `Precision/Recall`.
7. `model.fit(..., validation_data=(X_val,y_val))` com epochs, batch size, callbacks (per-epoch history, `EarlyStopping(monitor='val_loss', patience=..., restore_best_weights=True)`). Não escolher melhor modelo pelo teste.
8. Extrair histórico **verdadeiro** de cada época, treino vs validação; medir validação e selecionar o experimento vencedor por `val_loss` por padrão.
9. Escolher 1 experimento pela validação, destravar teste explicitamente; `y_prob = predict(X_test)`; `y_pred = y_prob >= threshold`; calcular matriz de confusão `[[TN,FP],[FN,TP]]` e métricas finais. Teste não influencia treinos seguintes.

## Hiperparâmetros iniciais sugeridos (não são exigência do enunciado)

- 2 camadas ocultas: `[32, 16]`; ativação `relu`.
- saída 1 neurônio `sigmoid`; otimizador `adam`; LR `0.001`.
- epochs `50`; batch size `32`; seed `42`.
- regularização L2 `0.0` inicial; early stopping habilitado com patience 8.
- seleção por `val_loss`; threshold `0.5`.

## Tratamentos de faltantes — ensinar diferenças

- **Remover:** menos linhas, potencial viés; treina/testa somente coorte completa; informar quantas linhas removidas e percentuais por classe.
- **Média:** sensível a outliers; somente numéricas.
- **Mediana:** robusta a outliers; somente numéricas.
- **KNN Imputer:** baseada em vizinhos e distâncias; cuidado com escalas diferentes e custo computacional. Para KNNImputer, uma implementação sem leakage pode inserir um escalonamento prévio de numéricas ajustado somente no treino antes da imputação por vizinhos; se optar por `KNNImputer` antes do scaler, documentar que as distâncias ficam influenciadas pela escala. Não ocultar este efeito.
- **Categorical:** usar método apropriado separado (moda ou marcador). Não aplicar mediana/média em `environment`.

## EDA versus transformação para treinamento

EDA global permite **descrever** distribuições e faltantes; estatísticas da EDA são dados do conjunto completo, inclusive linhas futuras de teste, e não devem ser reutilizadas como parâmetros de transformação ou seleção do modelo. Se houver dúvida, mostrar por padrão EDA no conjunto treino após split para decisões de modelagem. É aceitável visualizar uma EDA descritiva global inicial desde que a aplicação identifique sua finalidade e não guie tuning pelo conjunto de teste.

## Métricas

```text
                  Previsto 0   Previsto 1
Real 0                TN           FP
Real 1                FN           TP

accuracy     = (TP+TN) / (TP+TN+FP+FN)
precision    = TP / (TP+FP)
recall       = TP / (TP+FN)
specificity  = TN / (TN+FP)
F1          = 2*(precision*recall)/(precision+recall)
```

Forneça exemplos numéricos independentes do dataset para ensinar cada fórmula, marcados claramente como “exemplo hipotético”. Não rotule a matriz de confusão 10x10 do PDF como resultado desta tarefa.

## Erros recorrentes que a UI deve antecipar

- Escalonar antes da divisão (vazamento estatístico).
- One-hot não aplicado a `environment`, ou tratá-lo como 0/1/2 ordinal.
- Usar `sample_id` como feature, levando a sinal artificial.
- Medir accuracy apenas no treino e concluir que generaliza.
- Comparar configurações ajustando epochs com dados de teste.
- Confundir *probabilidade* da sigmoid com classe prevista.
- Interpretar underfitting/overfitting como diagnóstico definitivo de apenas uma curva.
- Assumir que balanceamento 50/50 garante 100% de acerto.

## Reprodutibilidade

Guardar: hash SHA-256 do CSV, versão do TensorFlow e sklearn, seed, proporções, strategy de faltantes, scaler, shape real após encoding, arquitetura, otimizador, LR, epochs solicitadas e realizadas, métricas por época, horário de início/fim, métricas de validação e estado da avaliação final. Sem esses metadados a comparação não é científica.
