# Prompt para o OpenCode — Fase 2: faltantes, escalonamento e split

Inspecione o código atual e leia `AGENTS.md`, `docs/04-metodologia-ml.md`, `docs/05-nove-etapas.md`, `docs/06-criterios-de-aceite.md`.

Implemente as etapas 3 a 5, conectadas ao backend e ao estado de configuração do frontend. Estratégias de faltantes: remover linhas, média, mediana e KNN (categorias tratadas separadamente). Escalonamento: StandardScaler e MinMaxScaler (+ sem escala, rotulado experimento adicional). Split estratificado 70/15/15; alternativa 80/20 com validação interna de 20% da porção de treino (efetivo 64/16/20). Mostrar cobertura, contagem, antes/depois e classes em cada partição.

Crie `POST /api/preprocessing/preview`. As transformações devem ser ajustadas SOMENTE em treino, com `transform` em validação e teste (teste não deve ser usado para decisões). Não use `sample_id` nem `target` em features; `environment` recebe imputação adequada e OneHotEncoder.

Escreva testes de não-vazamento e disjunção dos conjuntos, prove o funcionamento de cada estratégia com dados reais e execute suite. Documente escolhas específicas, especialmente KNN e remoção de linhas.
