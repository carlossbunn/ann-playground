# Origem do trabalho, obrigatoriedades e decisões adicionais

## O que está no PDF original (`data/ML-ANN-Classification.pdf`)

- Página 1: bibliotecas NumPy, Pandas, Matplotlib, Scikit-Learn, TensorFlow/Keras; etapa 1 problema supervisionado, classificação, features versus labels; etapa 2 EDA (histogramas, inconsistências, outliers por boxplot, correlação); início da etapa 3 tratamento de faltantes (remoção ou imputação).
- Página 2: imputação por média, mediana ou KNN; etapa 4 StandardScaler/MinMaxScaler; etapa 5 divisão 70/15/15 ou 80/20, com papéis de treino, validação e teste.
- Página 3: etapa 6 MLP `keras.Sequential` com entrada, camadas ocultas e saída; etapa 7 forward/backpropagation, learning rate, batch size, épocas, camadas, neurônios; otimizadores SGD, Adam, RMSprop.
- Página 4: etapa 8 validação (overfitting, underfitting, convergência), curvas de aprendizado, early stopping.
- Página 5: L2, etapa 9 métricas fundamentais e ilustrações de matriz de confusão.

## O PDF NÃO especifica

Não fornece arquitetura MLP obrigatória, função de ativação, limiar de classificação, quantidade mínima de épocas, métrica prioritária para selecionar modelos, nota alvo, benchmark obrigatório, interface gráfica, linguagem de frontend, nem formato de entrega final. Esses itens neste pacote são **opções didáticas e técnicas propostas por nós**, não exigências do professor.

**Atenção:** na página 5 há ilustração de matriz 10x10; o dataset disponibilizado é **binário** (`target` 0/1), então a implementação correta para estes dados é **2x2**.

## O que sabemos do CSV, sem inventar semântica

São 10.000 registros, 13 colunas; `sample_id` é identificador, `environment` contém `cold`, `mild`, `hot` e faltantes; `target` tem 5.000 valores 0 e 5.000 valores 1. Nomes de sensores e medições não demonstram unidade física, causa, contexto industrial ou significado das classes. A aplicação deve dizer explicitamente “não informado no enunciado” quando necessário.

## Fontes oficiais úteis

- OpenCode V2 — Instruções `AGENTS.md`: <https://opencode.ai/v2/docs/instructions>
- OpenCode V2 — Instalação/uso: <https://opencode.ai/v2/docs>
- OpenCode — Comandos personalizados: <https://docs.opencode.ai/docs/commands/>
- Documentação scikit-learn: <https://scikit-learn.org/stable/user_guide.html>
- Guia TensorFlow Keras: <https://www.tensorflow.org/guide/keras>
