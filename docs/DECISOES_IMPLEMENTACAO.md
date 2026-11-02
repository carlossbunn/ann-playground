# Decisões de implementação — ANN Lab

## Versão 1.0 — Fases 3, 4 e 5 concluídas (estado atual)

### Treinamento e contratos

- TensorFlow 2.20.0 e Keras 3.10.0 instalados e executados com Python 3.12 / NumPy 2.2.5, em CPU. Dois threads intra-op e um inter-op; os datasets de treinamento e validação têm threadpool privado de tamanho 1 para evitar criação excessiva de threads.
- `PreparationSettings` separa as escolhas de modelagem dos parâmetros de visualização de `PreprocessingConfig`. Experimentos não armazenam coluna/bins/subconjunto da prévia como hiperparâmetros.
- Contratos imutáveis validados para modelo, treinamento e experimento. Camadas 1–4, neurônios 4–256, épocas 1–200, batch 8–256, LR >0 e ≤0,1, L2 0–0,1, sigmoid binária e entropia cruzada. Patience precisa ser menor que épocas quando early stopping está ativo.
- A seed do split e a seed de treinamento são independentes e explicitadas na interface e no registro.
- O resumo calcula dimensões da preparação real e os parâmetros densos por `(entradas + 1) × saídas`. O treinamento confere com `model.count_params()` do Keras. O preview não inicializa pesos nem altera a seed global de um treino em execução.
- O callback grava histórico verdadeiro a cada época. Validação de números finitos impede um experimento divergente de parecer concluído. Configuração enviada permanece imutável.

### Persistência, fila e recuperação

- SQLite local, em WAL, armazena documentos de experimentos e o ciclo de estudo. Configuração, estados, históricos, hashes, IDs ordenados, features e versões são persistidos. As transações de escrita usam `BEGIN IMMEDIATE`.
- Modelo `.keras` e pipeline `.joblib` são escritos primeiro com nome temporário e renomeados; o status só muda para `completed` após salvar os dois e seus hashes SHA-256.
- Um worker em thread executa no máximo um treino; a fila aceita um aguardando, além do ativo. Um lock de processo Linux (`flock`) impede duas APIs de treinarem sobre o mesmo armazenamento. Uma API deve ser iniciada com um único worker.
- O shutdown solicita parada na próxima época e preserva histórico parcial. Ao reiniciar, `queued`/`running` remanescentes passam a `interrupted`. Não há retomada silenciosa de pesos.
- O GET de progresso lê o SQLite e não depende do lock de treinamento. Há teste controlado que mantém o construtor ocupado e comprova que health continua respondendo e a fila rejeita o terceiro trabalho.

### Comparação e ciclo científico

- O primeiro experimento reserva um fingerprint da fonte e dos IDs ordenados de todas as partições. Mudanças que troquem a coorte ou as partições são rejeitadas nos próximos treinos desse ciclo (409).
- Assim, média/mediana/KNN, tratamento categórico, escala e MLP podem ser comparados com as mesmas amostras; trocar para remoção ou mudar seed/split depois do primeiro treino não coloca no treino um holdout antes reservado.
- A interface compara até quatro curvas, ordena pela perda de validação do modelo salvo ou pela perda preditiva sem L2 e exige seleção explícita para o teste. Não escolhe hiperparâmetros pelo teste.
- EarlyStopping monitora `val_loss` e restaura os melhores pesos quando habilitado. Sem early stopping, os pesos são da última época. Melhor época observada e época salva são metadados diferentes. A avaliação da validação é refeita sobre os pesos salvos.
- `loss` inclui L2; `predictive_loss` é a BinaryCrossentropy sem regularização. As duas são mostradas e a interface explica que comparar penalizações diferentes exige atenção a essa distinção.
- Avisos de overfitting, underfitting e estabilização são heurísticas qualificadas: só após pelo menos cinco épocas; afastamento de val_loss em mais de 8% do melhor anterior enquanto treino melhora, ou ambas as acurácias abaixo de 60%, ou variação das últimas três perdas preditivas de validação abaixo de 0,005. Não são exigências do professor nem diagnósticos definitivos.
- A detecção de rascunho alterado compara configurações por serialização canônica, independente da ordem das propriedades retornadas pela API. Alterar o nome não invalida pesos. “Usar como base” cria um rascunho, sem editar o experimento original.

### Avaliação final e exportação

- `/api/study/select` aceita apenas run concluído, enquanto o ciclo ainda está aberto. Seleção pode mudar pela validação até o teste ser aberto.
- A avaliação exige confirmação e congela a seleção em transação antes de carregar o holdout. Avaliações simultâneas recebem conflito, e a consulta de um resultado concluído é idempotente, sem nova inferência.
- Fonte e hashes dos artefatos são conferidos; o pipeline é recarregado sem fit; os IDs persistidos recuperam exatamente o teste daquele run. Só nesse momento ele passa por transform e inferência.
- Matriz na ordem `[[TN, FP], [FN, TP]]`, positivo=1 e rótulo por `score >= threshold`, com limiar definido no experimento. Fórmulas das métricas e casos de denominador zero são explícitos. A perda preditiva do teste usa `sklearn.metrics.log_loss`; métricas históricas vêm do Keras.
- Após abrir o teste, novos treinos e trocas de seleção são bloqueados. Falha operacional preserva seleção e permite retry apenas desse mesmo run; reinício durante avaliação marca `evaluation_failed`, sem reabrir a seleção.
- Relatório Markdown, JSON com manifesto completo, histórico CSV e Python estão disponíveis por run. Python verifica a identidade do CSV e reproduz treino/validação, mantendo avaliação de teste como ação explícita no laboratório.
- O teste ponta a ponta usa outro armazenamento e outras portas (8001/5174), para não consumir o holdout da aplicação do usuário.

### Resultado verificado

- `make check`: Ruff, ESLint, TypeScript e build aprovados.
- `make test`: **102 testes pytest e 5 testes Vitest aprovados**. Inclui treinamento real, regularização, early stopping que realmente interrompe e restaura, igualdade de matriz após recarregar modelo/pipeline, fila, erro, recuperação/reavaliação após reinício e bloqueio transacional/idempotente de teste.
- `make e2e`: **17 testes aprovados**. O fluxo integrado configurou e treinou duas MLPs com Keras, comparou curvas, selecionou pela validação, abriu o teste, conferiu 1.500 amostras na matriz, baixou o relatório e recarregou a página com resultado preservado.
- Capturas reais: `treinamento-real.png`, `comparacao-real.png`, `avaliacao-real.png` e `experimentos-mobile.png`, em `frontend/test-results/`.
- Testes axe cobrem as nove etapas, início e Experimentos. As etapas novas foram verificadas em 390 px; etapas de preparação também em 320 px. Não se afirma auditoria humana completa de acessibilidade.
- Avisos de depreciação do TestClient/AnyIO e NumPy/Keras foram emitidos no ambiente validado; testes passaram. GPU, cancelamento individual e retomada de pesos são extensões opcionais.

---

## Histórico — Fase 2

### Escopo e arquitetura

- Etapas 3–5 implementadas e conectadas a `POST /api/preprocessing/preview`. Etapas 1–2 preservadas; passos 6–9 continuam como conteúdo futuro explícito.
- scikit-learn 1.6.1 adicionado às dependências. `PreprocessingConfig` é imutável, valida enums e limites, rejeita campos extras e proíbe prévias do subconjunto de teste.
- `PreprocessingService` mantém até oito pipelines em memória, por configuração de preparação. Trocar coluna, subconjunto ou limite de exemplos reutiliza o mesmo pipeline já ajustado. Os cálculos são serializados por lock para evitar várias imputações KNN simultâneas.
- O cache é efêmero e não salva modelos ou experimentos no disco. SQLite e treinamento Keras continuam destinados à Fase 3. O health informa Fase 2 e treinamento indisponível.

### Ordem de cálculo e prevenção de leakage

1. Se `drop`, selecionar registros completos **somente pelos 11 atributos**, antes da divisão. Informar cobertura e perda por classe. Nenhum valor negativo ou outlier é descartado por essa regra.
2. Separar teste estratificado (15% ou 20%), com seed. Separar a validação na parte restante, também estratificada e com a mesma seed: 15/85 para 70/15/15; 20% dos 80% para 64/16/20. Em coortes não divisíveis exatamente, os tamanhos inteiros seguem o arredondamento do scikit-learn e a interface mostra as proporções efetivas.
3. Criar `ColumnTransformer` com a lista explícita dos atributos: dez numéricos e `environment`. `sample_id` e `target` nunca chegam ao tensor de features.
4. Chamar `fit_transform` **apenas com treino** e `transform` com validação. O serviço armazena a partição bruta de teste, mas **não a transforma nesta fase**. Isso segue a restrição mais forte de `docs/04-metodologia-ml.md`, que reserva até mesmo essa transformação para a avaliação final.
5. Prévias e estatísticas antes/depois usam exclusivamente treino ou validação, conforme a seleção. A EDA global da etapa 2 não é reutilizada como parâmetro dos transformadores.

### Imputação, escala e categorias

- Média e mediana vêm de `SimpleImputer` ajustado no treino. Remoção usa passagem direta dos numéricos já completos.
- KNN usa `ScaledKNNImputer`: `StandardScaler` interno com NaNs preservados, seguido de `KNNImputer`, ambos ajustados somente no treino. O resultado é reconvertido à escala original antes do StandardScaler/MinMaxScaler escolhido pelo usuário. A escolha “sem escala” desliga apenas o scaler final; a padronização interna das distâncias KNN permanece e é explicada na interface.
- A memória de trabalho dos blocos de distâncias do KNN é configurada para 32 MiB via `sklearn.config_context`. Isso não limita toda a memória do processo; limita o tamanho de trabalho usado pelo cálculo de distâncias.
- `environment` tem pipeline próprio: imputação por moda do treino ou marcador constante `missing`; depois `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`. Categorias desconhecidas em validação produzem zeros sem reajuste.
- Nomes e dimensões de saída vêm do pipeline real. Na base original são 13 features com moda, ou 14 quando o marcador `missing` é aprendido como categoria de treino. Na coorte completa, o marcador pode não aparecer e a dimensão é recalculada.
- Prévias numéricas incluem dados brutos, dados imputados nas unidades originais e dados escalonados. Os arrays imputados são recuperados pela inversa do scaler final, sem refazer vizinhanças KNN a cada seleção de coluna; diferenças residuais de ponto flutuante são esperadas.
- Estatísticas usam desvio padrão populacional (`ddof=0`) para corresponder ao StandardScaler. Histograma e estatísticas consideram o subconjunto inteiro; só a tabela é limitada. Até 12 exemplos reais são mostrados com ausentes primeiro, de maneira determinística e explicitamente não aleatória.
- Comparar duas estratégias mantém fonte, seed, coluna e subconjunto. Remover linhas altera a coorte e portanto os IDs selecionados; a comparação explica esse limite. Não há métricas de classificação nessas prévias.

### Estado da interface e rastreabilidade

- O rascunho compartilhado contém estratégia numérica, scaler, divisão, seed, vizinhos e tratamento categórico. É validado ao carregar de `localStorage`; em indisponibilidade de armazenamento, continua em memória.
- Trocar configuração altera a chave de consulta. Resultados de uma configuração anterior não são apresentados como se pertencessem à nova configuração. Erros oferecem retry e retiram as visualizações correspondentes à consulta que falhou.
- O painel da divisão informa contagens, classes, proporções reais e SHA-256 dos IDs na ordem de cada partição. A API inclui SHA-256 da fonte e versão do scikit-learn. O código Python reproduz a preparação e imprime os IDs exatos para registro local.
- O código apresentado usa o construtor real `build_pipeline`, explicita a divisão, `fit_transform` de treino e `transform` de validação; não transforma o teste. Há teste que executa esse código e compara arrays e IDs com o serviço.

### Verificação executada na Fase 2

- `make check`: Ruff, ESLint, TypeScript e build aprovados.
- `make test`: **81 testes aprovados**, incluindo os 45 da Fase 1. Casos adicionais cobrem os quatro tratamentos reais, escalas, one-hot, sentinelas de holdout, espionagem do fit/transform, partições disjuntas/estratificadas/reproduzíveis e entradas inválidas.
- `make e2e`: **15 testes aprovados** com Chrome e API reais. Incluem comparação, efeito dos vizinhos KNN, remoção, categoria ausente, MinMax, seed/divisão, persistência após reload, erro recuperável e etapas futuras honestas.
- axe aprovado no início e nas etapas 1–5; ausência de transbordamento da página verificada em 320 e 390 px nas novas etapas.
- Capturas inspecionadas: `frontend/test-results/etapa-3-desktop.png`, `etapa-4-desktop.png`, `etapa-5-desktop.png` e `divisao-mobile.png`.
- Serviços reiniciados com `python3 scripts/dev.py`; a suite final de navegador usou a aplicação atualizada em execução.

---

## Histórico — Fase 1

## Escopo da entrega

O primeiro incremento implementa o MVP de dados e EDA previsto em `docs/07-roadmap.md`. Os passos 1–2 são interativos e calculados sobre o CSV. Os passos 3–9 têm navegação e explicações específicas, mas mostram explicitamente “A implementar”. Não há treinamento, experimentos persistidos ou avaliação do teste nesta fase.

## Stack e execução local

- Frontend: React 18, TypeScript 5, Vite 6, TanStack Query 5, Recharts 2, ícones Lucide e CSS próprio.
- API: FastAPI, pandas e NumPy em ambiente Python 3.12. Versões diretas fixadas em `backend/requirements.txt`; frontend com versões fixadas e `package-lock.json`.
- O Vite 6 foi escolhido para compatibilidade com o Node.js 18.19.1 disponível. O script de instalação também aceita versões mais novas do Node.js; Node.js 22 LTS é a recomendação para instalações novas.
- `scripts/setup.py` funciona mesmo quando há Node.js sem npm, usando npm 10.9.2 temporário do registro oficial. `scripts/dev.py` inicia e encerra os dois processos e dispensa npm durante a execução.
- API e frontend vinculados a `127.0.0.1`. Vite faz proxy de `/api`; CORS direto aceita somente as origens locais configuradas.
- SQLite, scikit-learn e TensorFlow/Keras entram nas fases que realmente usam seus recursos. O endpoint de saúde informa que o motor de treinamento não está inicializado, sem declarar sucesso fictício de TensorFlow.

## Fonte e cálculos

- Fonte única: CSV original em `data/`, carregado na inicialização da API e mantido em memória somente para leitura. Alterações no arquivo exigem reiniciar o backend.
- O perfil JSON de referência só é lido nos testes. Perfil, prévia, distribuições, faltantes e correlação da API são calculados a partir do DataFrame.
- O SHA-256 da fonte é exposto no perfil para rastreabilidade.
- Ausências viram `null` na prévia. Estatísticas indefinidas e correlações sem dados suficientes também retornam `null`, sem `NaN` em JSON.
- Histogramas usam `numpy.histogram`, com 10/20/30 intervalos de mesma largura no domínio observado; o último intervalo inclui a extremidade direita. Ausências não entram nas barras numéricas.
- Boxplot: quartis com interpolação padrão do pandas; limites em Q1−1,5×IQR e Q3+1,5×IQR; bigodes são os extremos observados dentro desses limites. A contagem de outliers considera todos os registros; o SVG exibe no máximo 80 valores reais amostrados uniformemente na lista ordenada de outliers. Esse limite é informado no gráfico.
- Correlação Pearson por pares válidos, com matriz das contagens de observações conjuntas. `sample_id`, `target` e `environment` não entram na matriz dos atributos numéricos.
- `environment` preserva as categorias textuais. “(ausente)” é apenas marcador diagnóstico, não categoria de domínio. `target` pode aparecer em contagens descritivas, mas nunca é tratado como atributo preditor.

## Integridade experimental e apresentação

- Toda a EDA desta fase é global e descritiva. A interface explica que essas estatísticas não são parâmetros de imputação/escala nem resultados de validação.
- A sequência visual das etapas não autoriza ajustar preprocessadores antes do split: na Fase 2, as prévias de tratamento devem usar uma divisão já criada e fit somente no treino.
- Não são inventadas semânticas para sensores, unidades ou classes. Valores negativos e outliers são mantidos.
- Não existem resultados de treinamento ou matriz fictícia. Etapa 9 mostra teste reservado e explica as dependências para sua futura liberação.
- Trechos “Ver o Python” reproduzem os cálculos com os controles atuais e usam o caminho relativo à raiz do projeto, como informado na própria interface.
- Navegação por hash permite abrir diretamente uma etapa e usar voltar/avançar do navegador. Mudanças de etapa focam o conteúdo. A ajuda vira painel recolhível nas telas pequenas.

## Verificação

- Pytest compara perfil e contagens com a referência, verifica paginação, serialização de ausências, histogramas, IQR, correlações, entradas inválidas, CORS e indisponibilidade real do CSV.
- Playwright utiliza o Chrome real e a API real para testar navegação, mudança de atributo/bins, tabela, categorias, faltantes, correlações, quizzes, snippets, erros recuperáveis e layout mobile.
- A verificação de indisponibilidade de rede interrompe uma requisição no navegador; nenhum teste de gráfico substitui a API por métricas simuladas.
- axe verifica automaticamente regras WCAG A/AA nas páginas implementadas. Isso complementa as verificações de teclado e responsividade; não equivale a uma auditoria humana completa de acessibilidade.
- Ruff, ESLint e TypeScript verificam o código. Capturas funcionais ficam em `frontend/test-results/`, sem modificar os dados originais.

### Resultado da execução desta entrega

- `python3 scripts/setup.py`: instalação concluída, incluindo a alternativa de npm temporário nesta máquina.
- `make check`: Ruff, ESLint, TypeScript e build Vite aprovados.
- `make test`: **45 testes aprovados**. A dependência Starlette emite um aviso de depreciação de alias do AnyIO no TestClient; não houve falhas.
- `make e2e`: **9 testes aprovados** no Google Chrome, incluindo o teste axe nas três páginas implementadas (início e etapas 1–2).
- `python3 scripts/inspecionar_base.py`: 10.000 linhas, 13 colunas, 5.000 rótulos de cada classe e contagens de ausências conferidas.
- `python3 scripts/dev.py`: API e interface iniciadas; health e frontend responderam HTTP 200. Os testes de navegador finais também passaram contra esses serviços em execução.
- Capturas: `frontend/test-results/inicio-desktop.png`, `eda-desktop.png` e `inicio-mobile.png`.
