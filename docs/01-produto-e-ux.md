# 01 — Produto, fluxo de usuário e UX

## Norte de produto

> “Um laboratório de classificação binária onde o usuário aprende fazendo, sem se perder em código antes de entender o conceito.”

Público principal: estudante universitário de IA iniciando redes neurais. O sistema deve facilitar o *raciocínio científico*, não mascará-lo. Permitir experimentar repetidamente configurações e comparar evidência quantitativa real.

## Layout desktop

```text
┌────────────────────────────────────────────────────────────────────────────┐
│ ANN LAB  / Classificação binária                     Experimentos [3] ?    │
├──────────────────┬──────────────────────────────────────┬─────────────────┤
│  ETAPAS          │  3 / Tratamento de faltantes         │ ENTENDA O PORQUÊ│
│  ✓ 1 Problema    │  [Média] [Mediana] [KNN] [Remover]   │ Conceito em 50  │
│  ✓ 2 Explorar    │                                      │ palavras        │
│  ● 3 Faltantes   │  tabela/heatmap/antes-depois         │                 │
│  ○ 4 Escala      │                                      │ O que muda?     │
│  ○ 5 Divisão     │  [Experimente agora]                  │ Dica e aviso    │
│  ○ 6 MLP         │                                      │                 │
│  ○ 7 Treino      │  Resultado com números reais         │ [Ver Python]    │
│  ○ 8 Validação   │                                      │                 │
│  ○ 9 Avaliação   │  [← Voltar]          [Continuar →]   │                 │
├──────────────────┴──────────────────────────────────────┴─────────────────┤
│ Trilha atual: imputação=mediana · escala=standard · split=70/15/15         │
└────────────────────────────────────────────────────────────────────────────┘
```

**Proporções sugeridas:** sidebar 224 px, workspace elástico (mínimo 560 px) e painel lateral 288 px; abaixo de 1100 px recolher painel de ajuda em drawer; mobile sidebar vira seletor horizontal rolável. Não transbordar tabelas na tela inteira.

## Navegação e conteúdo

1. **Boas-vindas:** problema, total de linhas, colunas, rótulos balanceados, “começar experimento”, tempo *estimado?* não exibir sem medição.
2. **Nove etapas:** cada uma tem o mesmo modelo visual: título, microtexto, controle, visualização, interpretação, código e botão continuar.
3. **Experimentos:** painel tabular com id, nome, data, preprocessing, arquitetura, treino, `val_loss`, `val_accuracy`, status; comparação lado a lado de pelo menos dois runs.
4. **Revisão de escolhas:** resumo consolidado do pipeline e alerta de vazamentos evitados. Mostrar decisões bloqueadas por dependência e por quê.
5. **Glossário contextual:** tooltip “feature”, “label”, “imputação”, “desvio padrão”, “época”, “backpropagation”, “overfitting”, “F1”, “falso positivo”.

## Componentes de aprendizagem

- **Mudar e observar:** controles conectados a consultas reais ao backend, não alterações cosméticas.
- **O que mudou:** resumos gerados do resultado real, por exemplo “A estratégia de remover linhas reduziu a amostra para N registros” (N somente após cálculo).
- **Ver o Python:** código legível, coerente com o backend e com os parâmetros. Nunca simular que a expressão reproduz código que não existe.
- **Antes × Depois:** tabelas e gráficos comparativos para imputação e escala. Visualização de treino/val/teste separadas.
- **Eu entendi?:** pergunta curta, resposta explicada, não impede avançar. Exemplos: “Posso ajustar o StandardScaler no teste?” (não); “Por que `sample_id` não é feature?” (identificador não sinal de domínio).
- **Status:** carregando, pronto, erro recuperável, em treinamento (progresso por época), sem experimento, teste reservado, teste avaliado.

## Controles e microinterações

- Os sliders mostram valor numérico, faixa e efeito provável.
- Ao alterar pré-processamento/arquitetura após treino, exibir “Esta configuração mudou. Execute novo treinamento para atualizar resultados.”
- Um “Resetar escolhas” restaura parâmetros padrão, sem apagar experimentos salvos sem confirmação.
- Cada gráfico tem legenda, títulos de eixo, unidades quando conhecidas e botão para explicação de interpretação.
- Nunca usar *“Excelente acurácia!”* sem resultado explícito; evitar afirmações causais.
- A execução final do teste requer confirmação de que o experimento foi selecionado somente pela validação.

## Acessibilidade

Contraste WCAG AA, foco visível, navegação por teclado, labels claros, resultados não dependerem só de cor, descrições textuais de gráficos, mensagens `aria-live` para status e suporte a redução de movimento.

## Recurso extra, pós-MVP

Sandbox de um único exemplo (amostra de teste sem rótulo durante configuração), permitindo ver features → preprocessamento → probabilidades → previsão, apenas depois do teste final autorizado; o rótulo real deve ser mostrado separadamente após a predição. Não transformar isso em ferramenta preditiva de domínio: classes 0 e 1 não possuem significado físico informado.
