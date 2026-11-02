# 10 — Direção visual e conteúdo educacional

## Personalidade visual

Uma ferramenta científica moderna, não painel administrativo genérico: **limpa, exploratória e focada em gráficos**.

- Fundo: `#F6F8FB`; superfícies `#FFFFFF`; borda `#E5EAF2`; texto `#17243B`.
- Azul principal `#315DDB` para controles; verde-petróleo `#159B8E` para conclusão/validação; âmbar `#DA8B2C` para alertas de interpretação.
- `font-family: Inter, system-ui, sans-serif` ou fallback nativo; títulos firmes, parágrafos curtos.
- 8 px base spacing; cards com raio ~12 px; layout fluido com max-width.
- Evitar encher telas de cards numéricos repetidos; usar gráfico como elemento principal, com legenda textual e contextualização.
- Gráficos de treino versus validação com cores consistentes e linhas distinguíveis também por tracejado, não só por cor.
- Não usar arte gerada artificialmente quando os dados reais são mais úteis.

## Cópias prontas (pt-BR)

- Cabeçalho: **“ANN Lab — Aprenda classificação binária experimentando”**
- Boas-vindas: “Você vai transformar um CSV em uma rede neural. A cada etapa, mude uma escolha e observe o resultado com dados reais.”
- Passo 1: “A rede aprende padrões das *features* para prever a *label* `target`. O que os valores 0 e 1 representam no mundo real não foi informado.”
- Passo 2: “Antes de treinar, conheça os dados. Procure faltantes, assimetrias, outliers e relações entre variáveis.”
- Passo 3: “Não existe um tratamento de ausentes universalmente melhor. Compare o que você ganha e perde.”
- Passo 4: “Escalonar evita que uma variável domine os cálculos apenas por ter números maiores.”
- Passo 5: “O teste é uma prova final. O modelo não deve usá-lo para estudar.”
- Passo 6: “Cada camada aprende uma transformação. Mais neurônios aumentam capacidade, mas podem elevar risco de overfitting.”
- Passo 7: “Treine o modelo e observe se a perda diminui no treino e na validação.”
- Passo 8: “Quando o treino melhora e a validação piora, talvez você esteja memorizando em vez de generalizar.”
- Passo 9: “A matriz de confusão mostra *como* o modelo erra. Um único número de acurácia não conta toda a história.”
- Sem resultado: “Ainda não há treino para esta configuração. Clique em ‘Treinar’ para produzir resultados reais.”
- Mudança pendente: “Você alterou os parâmetros. Os gráficos anteriores pertencem a outro experimento.”
- Erro de execução: “Não foi possível concluir este experimento. Veja os detalhes e tente novamente.”
- Teste bloqueado: “Escolha um modelo pela validação antes de consultar o teste.”
- Alerta de leakage: “Este valor deve ser calculado só a partir dos dados de treino. Usar o teste aqui compromete a avaliação.”

## Comportamento do painel “Entenda o porquê”

Mostrar um conceito de 2–4 frases, uma fórmula ou exemplo simples, um “erro comum” e um link para expandir detalhes. A ajuda deve mudar com a etapa, e não repetir sempre texto introdutório.

## Regras para microtextos automáticos

Montar frases a partir de medições reais **exclusivamente depois** que a API devolver dados. Não usar aproximações inventadas. Ex.: “O método X reteve Y amostras” somente após existir `Y` real na resposta.
