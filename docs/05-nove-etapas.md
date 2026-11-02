# 05 — Roteiro pedagógico: as 9 etapas

Para cada etapa, a tela deve oferecer: **[Conceito] → [Experimente] → [Observe] → [Explique] → [Código Python] → [Verificação de aprendizado]**.

## Etapa 1 — Entendimento do problema (PDF p.1)

- **Ensinar:** aprendizado supervisionado, classificação binária, features vs label, o que é uma predição.
- **Exibir:** dataset com 10 mil linhas, 13 colunas, 0 e 1 balanceados; tabela com campos e funções (`sample_id`, atributos, `target`).
- **Interação:** escolher uma variável e receber classificação “feature/label/ID”; marcar como preditora e imediatamente explicar por que `target` e `sample_id` são bloqueados.
- **Observe:** X com 11 atributos originais (10 numéricos + 1 categórico); y binário.
- **Quiz:** “Se eu colocar `target` em X, o que acontece?”

## Etapa 2 — EDA (PDF p.1)

- **Ensinar:** histograma (frequências), boxplot (mediana, quartis, outliers), correlação (associação, não causalidade), inconsistências.
- **Exibir:** seleção de coluna, histograma configurável em bins 10/20/30; boxplot com outliers IQR; matriz heatmap de correlação entre numéricas; barras da variável `environment`; mapa de percentuais de faltantes.
- **Interação:** alternar a coluna, passar o mouse para ler valores reais, escolher ordenar por % faltante.
- **Observe:** valores negativos podem existir e não significam erro sem dicionário de domínio. Outliers não são automaticamente inválidos.
- **Quiz:** “Correlação alta prova que uma feature causa a classe?”

## Etapa 3 — Tratamento de faltantes (PDF pp.1–2)

- **Ensinar:** ausência de dados; remoção vs imputação; média, mediana, KNN; tradeoff informação perdida x distorção.
- **Exibir:** contagens faltantes por campo; comparação antes-depois para coluna selecionada; tamanho da coorte se remover.
- **Interação:** 4 estratégias (remover/média/mediana/KNN), KNN vizinhos 3/5/7 opcional, `environment` categórica tratada separadamente.
- **Observe:** mudanças em amostra/distribuição, sem alterar CSV original. Calcular prévias com ajuste somente no conjunto de treino.
- **Quiz:** “Por que não calcular a mediana usando treino + teste?”

## Etapa 4 — Escalonamento (PDF p.2)

- **Ensinar:** diferenças de unidades e magnitudes, por que MLP é sensível à escala; StandardScaler e MinMaxScaler.
- **Exibir:** para variável numérica, histograma antes/depois (ajustado no treino), estatísticas comparativas e faixa transformada.
- **Interação:** `StandardScaler` x `MinMaxScaler` e opção `sem escalonamento` **identificada como experimento didático extra**.
- **Observe:** StandardScaler centraliza e padroniza em torno das estatísticas do treino; MinMaxScaler mapeia mínimos/máximos do treino para o intervalo padrão — valores do teste podem cair fora de [0,1].
- **Quiz:** “Por que variáveis de magnitudes diferentes podem prejudicar uma MLP?”

## Etapa 5 — Separação dos dados (PDF p.2)

- **Ensinar:** treino aprende pesos, validação guia ajustes, teste estima generalização imparcial.
- **Exibir:** barras de proporção, contagem de classes por partição, seed.
- **Interação:** selecionar 70/15/15 ou 80/20. Na segunda opção, deixar explícita a validação interna de 20% da parte de treino (resultado 64/16/20 efetivo).
- **Observe:** classes estratificadas; dados de teste não revelados nos passos seguintes além da descrição global inicial; seed altera quais linhas entram em cada grupo.
- **Quiz:** “Posso escolher hiperparâmetros usando a acurácia do teste?”

## Etapa 6 — Construção da MLP (PDF p.3)

- **Ensinar:** neurônio, camada de entrada, camadas ocultas, saída sigmoid, parâmetros treináveis.
- **Exibir:** diagrama de fluxo X → camadas ocultas → 1 saída, número real de features após one-hot, resumo `model.summary()` ou equivalente legível.
- **Interação:** 1–4 camadas ocultas, 4–256 neurônios por camada, ativação relu/tanh; adição/remoção de camada, contagem de parâmetros.
- **Observe:** diferenciar entrada (número de features pós-encoding) e saída binária (1 neurônio). Evitar diagrama cuja contagem de neurônios não corresponde à configuração atual.
- **Quiz:** “Por que 1 neurônio sigmoid é suficiente para 2 classes?”

## Etapa 7 — Treinamento (PDF p.3)

- **Ensinar:** forward propagation, loss, backpropagation e otimizadores, epochs, batches e learning rate.
- **Exibir:** formulário compacto de hiperparâmetros, status, epochs 1…N, curvas reais de loss e accuracy treino/validação.
- **Interação:** escolher Adam/SGD/RMSprop, LR, epochs, batch size; “Treinar” cria novo run rastreável; botão para ver últimas métricas.
- **Observe:** mudanças de parâmetro não alteram treino anterior; explicação do que foi modificado para o próximo run.
- **Quiz:** “O que muda se aumentar learning rate demais?”

## Etapa 8 — Validação (PDF pp.3–5)

- **Ensinar:** underfitting, overfitting, convergência, early stopping, regularização L2; diferença entre curva de treino e validação.
- **Exibir:** train loss vs validation loss; epoch da melhor validação; badges textuais com hipóteses “Possível overfitting” quando sustentado por sinais, sem diagnóstico infalível.
- **Interação:** ligar/desligar early stopping, patience, L2, iniciar novo run, comparar experimentos em `val_loss` e `val_accuracy`.
- **Observe:** não editar run já concluído; ao comparar, mesma divisão/seed para comparação justa ou exibir alerta.
- **Quiz:** “Por que reduzir a loss do treino não garante melhoria da validação?”

## Etapa 9 — Avaliação final (PDF p.5)

- **Ensinar:** matriz confusão 2x2; TN, FP, FN, TP; accuracy, precision, recall, specificity e F1.
- **Exibir:** seleção do experimento melhor na validação, botão “Avaliar no teste final” com confirmação; depois matriz 2x2 colorida, rótulos de eixo e interpretação de cada célula, valores da métrica com fórmula e cálculo a partir da matriz.
- **Interação:** passar o mouse em TP/FP/TN/FN para significado e explorar exemplo fictício **em aba separada/claramente marcada** sem alterar métricas reais; exportar relatório de experimento.
- **Observe:** teste desbloqueado uma única vez para seleção final; aviso se experimentos diferem no split/seed.
- **Quiz:** “Se FN aumenta, o que acontece com recall, mantendo TP constante?”

## Conjunto de exercícios curtos e respostas

1. `sample_id` é feature? **Não**, é identificador.
2. O professor determina que 0 significa defeito? **Não**, significado das classes não foi dado.
3. É adequado usar a média de todo o CSV para imputar testes? **Não**, gera leakage.
4. Dado ausente em `environment` pode receber média aritmética? **Não**, variável categórica.
5. `accuracy=0.90` no treino significa 90% em novos dados? **Não**, pode haver overfitting.
6. Por que sigmoid na saída binária? **Produz score interpretado como probabilidade da classe 1 sob treinamento adequado.**
7. A matriz do nosso teste tem quantas linhas/colunas? **2x2**.
8. Qual conjunto orienta hiperparâmetros? **Validação**, não teste.

## Critério pedagógico mínimo

Uma etapa **não pode ser considerada pronta** se faltar qualquer um: conceito, ação que produz resultado real, visualização do efeito, interpretação textual e conexão com código reprodutível. Nos passos 6 e 7, o código deve realmente usar Keras no backend.
