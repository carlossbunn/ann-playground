# Prompt mestre — COPIE O TEXTO ABAIXO PARA O OPENCODE

Você está no repositório **ANN Lab**. Quero que você **construa de verdade** um playground web para aprender passo a passo a desenvolver uma rede neural MLP capaz de classificar a coluna `target` (0 ou 1) do dataset `data/base_classificacao_binaria.csv`. O objetivo não é apenas entregar uma classificação: quero compreender cada decisão enquanto a altero na interface.

Comece lendo `AGENTS.md` e os documentos em `docs/` antes de implementar. O PDF `data/ML-ANN-Classification.pdf` é o enunciado acadêmico. O CSV contém os dados reais. Use o perfil em `data/perfil_base.json` para validar sua leitura, mas **não use valores pré-calculados como substituto para análise executada pelo backend**.

## Experiência web esperada

- Aplicação local com layout profissional, clara e responsiva: sidebar com 9 etapas, área de experimentação e painel “Entenda o porquê”. Sem interface genérica de formulário.
- Cada etapa oferece explicação curta, controles reais, visualização, resumo do efeito e bloco “Ver o Python” correspondente.
- Percurso: (1) problema/target/features; (2) EDA com histogramas, boxplots, correlação e outliers; (3) comparação de remoção e imputações média, mediana e KNN; (4) StandardScaler ou MinMaxScaler; (5) partições treino/validação/teste com proporções e seed; (6) construtor visual MLP; (7) treinamento real com épocas, batch, learning rate, Adam/SGD/RMSprop; (8) curvas treino/validação, identificação de overfitting, early stopping e L2; (9) avaliação 2x2 com acurácia, precisão, recall, F1, especificidade e explicações.
- Laboratório de experimentos: salvar configuração, reproduzir run, comparar duas ou mais curvas/indicadores de validação e escolher experimento para avaliação FINAL no teste.
- Tudo em português; mensagens de erro e estados vazios claros; dados e gráficos nunca fictícios. Se um resultado ainda não existe, mostrar estado pendente, nunca um placeholder que se passe por experimento real.
- Uma visão de “linha de raciocínio” mostra as escolhas acumuladas e avisa sobre leakage, exclusão de sample_id, variável categórica e dados faltantes.

## Stack e estrutura

Frontend React + TypeScript + Vite + Tailwind (ou CSS estruturado) + Recharts; backend Python + FastAPI + Pandas + scikit-learn + TensorFlow/Keras; persistência local SQLite. Acesso local sem autenticação ou serviços externos. Mantenha frontend e backend separados, com documentação de como executar.

## Correção experimental — inegociável

- `target` apenas como y; `sample_id` apenas como identificador; `environment` categórica com OneHotEncoder.
- Dividir dados de forma estratificada, preservando seed; ajustadores (`imputer`, `scaler`, `encoder`) **fit apenas no treino**. Validação e teste só passam por transform.
- Se a escolha for “remover linhas com ausências”, documentar a cobertura/perda de amostras e os efeitos de seleção antes da divisão.
- Dataset inicial com 10.000 linhas, 13 colunas, 5.000 rótulos 0 e 5.000 rótulos 1; isso é conferência, não meta de resultado.
- O teste final fica reservado até selecionar um experimento usando validação.
- Em classificação binária, saída Keras `Dense(1, activation="sigmoid")`, loss `binary_crossentropy`; defina com o usuário a métrica de seleção da validação (padrão `val_loss`).
- Treino real, retorno de progresso por época, curva loss/accuracy reais e inferência de rótulo com threshold padrão 0.5.
- Reprodutibilidade sem prometer determinismo perfeito em todos os hardwares; informe seed e versões.

## Ordem de execução desejada

**Fase 1 — MVP demonstrável:** estrutura do app, leitura real do CSV, tela de boas-vindas, sidebar, etapas 1–2 (perfil, dados faltantes, distribuições, 2x2 ainda bloqueada) e testes de API.

**Fase 2 — preparação:** etapas 3–5, imputação e escala corretas, partição estratificada com resumo e testes de leakage.

**Fase 3 — rede neural:** etapas 6–7, arquitetura configurável, treinamento Keras em tarefa assíncrona com polling, histórico verdadeiro e persistência.

**Fase 4 — avaliação:** etapas 8–9, regularização, comparação de experimentos, finalização do teste e matriz 2x2.

**Fase 5 — acabamento:** experiência responsiva, visualizações ricas, estados de erro, exportação de resumo/code snippets, testes end-to-end e documentação.

Implemente a Fase 1 primeiro, execute os testes, me mostre comandos de execução e explique o que foi concluído. Continue as demais fases quando eu solicitar; **não substitua funcionalidades reais por mocks**. Considere os critérios específicos em `docs/06-criterios-de-aceite.md` e os contratos em `docs/03-contrato-api.md`. Se você tiver capacidade/contexto suficiente para entregar mais sem comprometer a qualidade, mantenha fases independentes e verificáveis.
