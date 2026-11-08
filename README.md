# ANN Lab — Playground Web para aprender classificação binária

> **Versão 1.0 — as nove etapas implementadas, do CSV ao teste final.**
> O projeto começou como um kit de especificação. Agora inclui `frontend/`, `backend/`, testes e scripts de execução, além dos documentos e dados originais.

## Executar a aplicação

Requisitos: **Python 3.12**, módulo `venv`, **Node.js 18.18+** (recomendado: 22 LTS) e conexão com a internet na instalação. Validado em Linux com Python 3.12.3 e Node.js 18.19.1. Depois da instalação, o app usa apenas os serviços locais; fontes e ícones não dependem de CDNs.

Na raiz deste projeto — a pasta que contém este README:

```bash
python3 scripts/setup.py
python3 scripts/dev.py
```

- **Aplicação:** <http://127.0.0.1:5173>
- **Documentação da API:** <http://127.0.0.1:8000/docs>
- `Ctrl+C` no terminal encerra os dois serviços.
- Se as dependências já estiverem instaladas, execute apenas `python3 scripts/dev.py`.
- Equivalentes com Make: `make setup` e `make dev`.

O instalador cria `backend/.venv` e instala o frontend conforme `package-lock.json`. Se `npm` não estiver instalado, usa uma cópia temporária do npm 10.9.2; não altera o Node.js global. Não é necessário ativar o ambiente virtual para os scripts acima.

O treinamento usa **TensorFlow 2.20.0 / Keras 3.10.0 em CPU**, com NumPy 2.2.5. A primeira instalação baixa o pacote TensorFlow (aproximadamente 620 MB) e suas dependências. Não é necessário instalar CUDA. Mensagens de detecção de CUDA não indicam uso de GPU; o modo validado e registrado nos experimentos é CPU.

### O que já funciona

- Boas-vindas com perfil calculado a partir do CSV original.
- Navegação pelas **nove etapas funcionais**, com contexto pedagógico e dados reais.
- **Etapa 1:** atributos versus alvo versus identificador, exploração do papel das colunas, balanceamento das classes e tabela paginada com ausências identificadas.
- **Etapa 2:** histogramas com 10/20/30 intervalos, boxplot e outliers 1,5 IQR, estatísticas, categorias de `environment`, ausências ordenáveis e matriz de correlação Pearson com número de pares válidos.
- **Etapa 3:** remoção, média, mediana e KNN (3/5/7 vizinhos); comparação de duas estratégias; histogramas, estatísticas e valores antes/depois; cobertura da coorte por classe; imputação separada e one-hot de `environment`.
- **Etapa 4:** StandardScaler, MinMaxScaler e opção didática sem escala; histogramas, estatísticas e valores transformados de treino ou validação.
- **Etapa 5:** divisão estratificada 70/15/15 ou 80/20 com validação interna (efetivo 64/16/20); seed configurável; contagem de classes; fingerprints das partições.
- **Etapa 6:** construtor de 1–4 camadas ocultas, 4–256 neurônios, ReLU/tanh, dimensão pós-encoding e contagem real de parâmetros conferida pelo Keras.
- **Etapa 7:** treino assíncrono real, Adam/SGD/RMSprop, learning rate, épocas, batch, seed de treinamento, limiar e curvas loss/accuracy por época.
- **Etapa 8:** early stopping, restauração de pesos, L2, comparação de até quatro curvas, ordenação por validação e seleção explícita para teste final.
- **Etapa 9:** confirmação do holdout, matriz 2×2, acurácia/precisão/recall/F1/especificidade, fórmulas e cálculos, com resultado persistido.
- **Experimentos:** configurações imutáveis, histórico SQLite, recuperação após reiniciar, modelos `.keras`, pipelines ajustados e exportação de relatório Markdown, JSON, histórico CSV e Python.
- Rascunho de preparação compartilhado entre etapas e salvo no navegador. Restaurar padrões altera as escolhas, sem modificar o CSV.
- Explicações contextuais, perguntas de aprendizagem e trechos Python que reproduzem os cálculos apresentados.
- Estados reais de carregamento/erro, botão para tentar novamente, layout responsivo, acesso por teclado e resumos textuais dos gráficos.

Todos os agregados vêm de cálculos do backend sobre `data/base_classificacao_binaria.csv`. O arquivo `perfil_base.json` é usado pelos testes, não como fonte de resultados da interface. O CSV não é modificado.

### APIs disponíveis

| Rota | Finalidade |
|---|---|
| `GET /api/health` | Disponibilidade real do CSV e estágio da implementação |
| `GET /api/dataset/profile` | Perfil, faltantes, estatísticas e SHA-256 da fonte |
| `GET /api/dataset/preview?offset=0&limit=30` | Prévia paginada; máximo de 100 registros |
| `GET /api/dataset/eda?column=sensor_a&bins=20` | Distribuição e boxplot ou contagens categóricas |
| `GET /api/dataset/correlations` | Correlação dos dez atributos numéricos e contagens de pares válidos |
| `POST /api/preprocessing/preview` | Preparação ajustada só no treino, comparações e resumo da divisão |
| `POST /api/model/preview` | Dimensão e parâmetros da arquitetura; código correspondente |
| `POST /api/experiments` | Cria experimento imutável e retorna 202 com ID e estado queued |
| `GET /api/experiments` / `GET /api/experiments/{id}` | Configuração, estado e histórico real |
| `GET /api/experiments/{id}/export?format=json` | `json`, `markdown`, `csv` ou `python` |
| `GET /api/study` / `POST /api/study/select` | Ciclo de estudo e escolha explícita pela validação |
| `GET /api/evaluation/final` / `POST /api/evaluation/final` | Consulta e avaliação final persistida |

A EDA da etapa 2 é **global e descritiva**. Nas etapas 3–5, a divisão acontece **antes de qualquer fit**: imputador, scaler e encoder aprendem somente com o treino; validação recebe apenas `transform`. O teste continua bruto e reservado, sem prévias dos seus valores transformados. A ordem das telas é pedagógica, não a ordem interna dos cálculos.

O app completo informa `phase=5`, `ml_engine="tensorflow"` e `training_available=true` quando o motor foi iniciado. A fábrica de testes permite desativar treinamento para testar só as APIs de dados.

### Como percorrer o laboratório

1. Explore o CSV nas etapas 1–2.
2. Compare tratamentos e escalas nas etapas 3–4 e defina a divisão/seed na etapa 5.
3. Construa a MLP na etapa 6. A saída é sempre `Dense(1, sigmoid)`.
4. Configure e execute o treino na etapa 7. Um teste rápido pode usar 2 épocas, batch 128 e patience 1; isso verifica o fluxo, não estabelece um benchmark de desempenho.
5. Altere a arquitetura ou os hiperparâmetros e treine outro experimento. Experimentos anteriores mantêm a configuração original.
6. Na etapa 8 ou em **Experimentos**, marque curvas para comparar e use **Selecionar para teste** na linha do modelo escolhido.
7. Na etapa 9, confirme a escolha pela validação e abra o teste final. Depois exporte o relatório.

### Use seu modelo

Na opção **Use seu modelo** do menu, escolha um experimento concluído, carregue uma amostra de treino ou validação, altere as leituras e compare a previsão com a referência. A API recarrega e confere os artefatos salvos e aplica o pipeline original sem novo ajuste; o teste reservado não é fornecido como amostra. A tabela mostra os valores após imputação e escala, inclusive média e divisor quando o scaler é StandardScaler. O contexto de máquina de furação é uma analogia didática: o CSV não documenta sensores físicos, unidades nem o significado das classes.

**Ciclo de estudo:** o primeiro experimento reserva a coorte e as partições (fonte, seed e proporções). Média/mediana/KNN, scaler, arquitetura e hiperparâmetros podem variar sem trocar as amostras de avaliação. Mudar de coorte por remoção, seed ou proporções após o primeiro treino retorna 409. Isso evita usar no treino de um experimento as amostras de teste de outro.

Ao abrir o teste, a seleção é congelada e novos treinos desse ciclo são encerrados. Recarregar a página ou reiniciar o app mantém a reserva e o resultado. Consultar novamente o mesmo resultado não reexecuta a avaliação. Uma falha operacional permite repetir somente o modelo já selecionado. Um estudo independente exige uma nova reserva de dados; a interface não oferece botão para reabrir o mesmo holdout como se fosse novo.

**Perdas e pesos:** `val_loss` inclui a penalização L2, enquanto `predictive_loss` mede entropia cruzada sem penalização. A tabela permite ordenar pelas duas. Com early stopping, o modelo salvo usa os melhores pesos por val_loss; sem ele, usa a última época. Melhor época observada e época dos pesos salvos são apresentadas separadamente. Métricas de validação são recalculadas sobre os pesos efetivamente salvos.

### Dados persistidos e execução

Por padrão, tudo fica em `local_storage/` (ignorado pelo versionamento): SQLite em `experiments.sqlite3`, e modelos/pipelines em `artifacts/{id}/`. Cada experimento guarda o SHA-256 da fonte, IDs ordenados das partições, nomes das features, versões, seeds, histórico e hashes dos artefatos. A avaliação final recarrega os arquivos desse experimento, confere sua identidade e só então transforma o teste.

Há **um treino em execução e no máximo um aguardando**. Uma segunda instância da API não pode usar o mesmo armazenamento. Use um único worker Uvicorn. Ao encerrar o app, treinos incompletos ficam `interrupted`, com histórico parcial; eles não são declarados concluídos nem retomados silenciosamente.

Para um diretório alternativo de armazenamento, defina `ANN_LAB_STORAGE` antes de iniciar o app. O CSV continua sendo o original do projeto. O relatório Python verifica o hash da fonte e reproduz preparação e treino; ele mantém o teste reservado para a ação explícita de avaliação.

### Contrato da preparação

Exemplo de configuração aceita por `POST /api/preprocessing/preview`:

```json
{
  "missing_strategy": "median",
  "scaler": "standard",
  "split_strategy": "70_15_15",
  "seed": 42,
  "knn_neighbors": 5,
  "categorical_strategy": "most_frequent",
  "column": "sensor_b",
  "subset": "train",
  "sample_limit": 12,
  "bins": 20
}
```

- `missing_strategy`: `drop`, `mean`, `median` ou `knn`.
- `scaler`: `standard`, `minmax` ou `none` (experimento didático).
- `split_strategy`: `70_15_15` ou `80_20`; seed inteira de 0 a 4294967295.
- `categorical_strategy`: `most_frequent` ou `constant` (marcador `missing`).
- `subset`: `train` ou `val`; `test` é rejeitado. Coluna deve ser um dos dez atributos numéricos.
- `sample_limit`: 1–100; `bins`: 10, 20 ou 30.

A resposta contém configuração aplicada, cobertura, partições/classes, hashes, versão do scikit-learn, nomes reais das features, tratamento categórico e `numeric_before_after`. Este último inclui `samples`, estatísticas e histogramas antes da imputação, após imputação e após escala. O código Python retornado reproduz o pipeline e imprime os IDs das partições para registro local.

**KNN:** padroniza internamente os atributos numéricos com estatísticas do treino para calcular distâncias e devolve as estimativas à escala original; só então aplica o scaler escolhido. **Remoção:** seleciona registros completos nos atributos antes de dividir, informa perda por classe e altera a população do estudo. **MinMax:** valores de validação podem ficar fora de [0, 1], pois os extremos vêm apenas do treino.

### Gráficos com Matplotlib

O Matplotlib está incluído nas dependências do backend. Na raiz do projeto, gere histogramas, boxplots e correlação descritiva do CSV completo:

```bash
backend/.venv/bin/python scripts/graficos_matplotlib.py
```

Para incluir curvas reais de um experimento, baixe o **JSON** na área de exportação e informe o caminho do arquivo:

```bash
backend/.venv/bin/python scripts/graficos_matplotlib.py --experimento /caminho/para/experimento.json
```

Os PNGs ficam em `local_storage/graficos/`; use `--saida /caminho` para outro diretório. O JSON de um experimento concluído também gera um gráfico das métricas de validação e, se disponível, do teste. Se o JSON contiver um teste final já realizado, a matriz de confusão 2×2 também será gerada; o script não abre nem reexecuta o teste. O SHA-256 do CSV é conferido antes de usar os resultados. Reexporte o JSON após a avaliação final para incluir a matriz. As figuras descritivas usam a base inteira, sem ajustar o pipeline de treinamento.

### Verificações

```bash
make check  # Ruff, ESLint, TypeScript e build de produção
make test   # dados/API/ML (pytest) + estado do frontend (Vitest)
make e2e    # testes reais de navegador com Playwright e verificações axe
python3 scripts/inspecionar_base.py
```

Validação da versão completa: **102 testes de backend, 5 testes Vitest e 17 testes de navegador aprovados**, além de Ruff, ESLint, TypeScript e build. Inclui Keras real, artefatos recarregados, early stopping real, recuperação após reiniciar, fila limitada, API responsiva, holdout transacional e métricas recalculadas da matriz.

Os testes de navegador usam **Google Chrome instalado** e sobem API em **8001** e frontend em **5174**, com armazenamento temporário isolado. Eles não consomem o teste do seu laboratório em 8000/5173. O teste ponta a ponta treina duas MLPs, compara, seleciona, avalia e baixa o relatório. Capturas reais ficam em `frontend/test-results/`, incluindo `treinamento-real.png`, `comparacao-real.png` e `avaliacao-real.png`. Responsividade e acessibilidade automática foram verificadas nas telas implementadas; isso não substitui uma auditoria humana completa.

### Serviços separados (opcional)

```bash
# Terminal 1, a partir da raiz
backend/.venv/bin/python -m uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000

# Terminal 2, a partir da raiz
node frontend/node_modules/vite/bin/vite.js frontend --host 127.0.0.1 --port 5173 --strictPort
```

O frontend encaminha `/api` ao backend local pelo proxy do Vite. Para acesso direto à API por outra origem local de desenvolvimento, configure `CORS_ORIGINS` com origens separadas por vírgula. A leitura do CSV é resolvida pela localização do serviço Python, independentemente do diretório do terminal.

### Notas operacionais

- A primeira execução do motor pode levar alguns segundos para importar TensorFlow.
- Se uma porta já estiver ocupada, encerre a instância anterior antes de iniciar outra.
- Para reproduzir artefatos, preserve o CSV e as versões registradas. Seeds não garantem identidade bit a bit entre equipamentos.
- Valores negativos e outliers não são excluídos automaticamente. Classes, sensores e unidades não têm semântica de domínio informada.
- Interrupções e erros aparecem no experimento; detalhes operacionais ficam no log do backend. Dependências de teste emitem avisos de depreciação do AnyIO e NumPy/Keras no ambiente validado, sem falhas na suite.
- Aceleração GPU, retomada de pesos de um treino interrompido e cancelamento individual de jobs são extensões opcionais; o percurso acadêmico completo funciona em CPU.

Decisões técnicas: [`docs/DECISOES_IMPLEMENTACAO.md`](docs/DECISOES_IMPLEMENTACAO.md). O material original de planejamento e os prompts por fase estão preservados abaixo e em `docs/` e `prompts/`.

---

## Material de referência do kit original

## Objetivo

Criar uma aplicação web local, em português do Brasil, que ensina **cada uma das 9 etapas da atividade de Inteligência Artificial**, enquanto permite alterar escolhas reais de processamento e de uma MLP (rede neural multicamadas) e observar seus efeitos sobre dados reais.

O estudante deve conseguir responder: *o que estou fazendo, por que isso é necessário, o que muda se eu alterar uma escolha e como comprovo o resultado?*

### Continuar a implementação com OpenCode

1. Extraia este ZIP e abra o terminal **na pasta `ann-playground-opencode`**.
2. Instale/abra o OpenCode conforme a documentação oficial: <https://opencode.ai/v2/docs>. Se já o utiliza, apenas execute `opencode`.
3. Abra `PROMPT_INICIAL_OPENCODE.md` para consultar o plano original; as fases previstas já foram implementadas.
4. Para evoluir o laboratório, considere o código e as decisões atuais antes de reutilizar os prompts históricos de `prompts/`.
5. Ao fim de cada fase, peça que ele **execute testes, inicie os serviços e abra o playground** para verificar o comportamento. Não aceite gráficos fictícios nem números inventados.

```bash
unzip ann-playground-opencode.zip
cd ann-playground-opencode
opencode
```

**Exemplo para solicitar uma evolução:**

```text
Leia AGENTS.md e PROMPT_INICIAL_OPENCODE.md, além de docs/ e data/.
Leia o código atual e docs/DECISOES_IMPLEMENTACAO.md.
Implemente a melhoria solicitada preservando as nove etapas, o fit só no treino
e a reserva persistida do teste final.
Execute verificações e explique o que foi implementado e como rodar.
```

### Conteúdo do pacote

| Arquivo / pasta | Finalidade |
|---|---|
| `AGENTS.md` | Regras permanentes para o OpenCode; evita dados simulados e leakage. |
| `PROMPT_INICIAL_OPENCODE.md` | Briefing único, pronto para colar. |
| `docs/01-produto-e-ux.md` | Layout, navegação e interação. |
| `docs/02-arquitetura.md` | Stack, estrutura, execução e persistência. |
| `docs/03-contrato-api.md` | Endpoints, JSON de exemplo, estados e falhas. |
| `docs/04-metodologia-ml.md` | Pipeline de ML correto e decisões pedagógicas. |
| `docs/05-nove-etapas.md` | O que cada uma das nove telas mostra, ensina e permite mudar. |
| `docs/06-criterios-de-aceite.md` | Testes de conclusão, funcionais e científicos. |
| `docs/07-roadmap.md` | Sequência de construção por entregas verificáveis. |
| `docs/08-dicionario-dados.md` | Análise do CSV original, sem inferir significados não informados. |
| `docs/09-fonte-e-limites.md` | O que o PDF exige e o que é proposta de implementação. |
| `docs/10-estilo-e-copias.md` | Direção visual e textos didáticos. |
| `prompts/` | Pedidos independentes para o OpenCode, um por fase. |
| `.opencode/commands/` | Comandos locais para ajudar a repetir as fases. |
| `data/base_classificacao_binaria.csv` | **CSV original (10 mil registros)**. |
| `data/ML-ANN-Classification.pdf` | **PDF original da atividade**. |
| `data/perfil_base.json` | Perfil extraído do CSV, para conferir análises na implementação. |
| `data/amostra_30_linhas.csv` | Amostra pequena para diagnósticos. |
| `scripts/inspecionar_base.py` | Verifica estrutura sem precisar de dependências externas. |

### Visão da experiência de aprendizagem

Um painel com **menu lateral de 9 etapas**, visualização central interativa e um **painel contextual “Entenda o porquê”**. Cada etapa tem conceitos, controles manipuláveis, prévia antes/depois, mini-experimento e explicação em português. Uma aba **Experimentos** compara configurações reais de rede neural. Um botão **“Ver Python”** apresenta trechos reais correspondentes à configuração atual.

**Preferências arquiteturais:** `React + TypeScript + Vite` no frontend; `FastAPI + pandas + scikit-learn + TensorFlow/Keras` no backend; `SQLite` local para experimentos; `Recharts` (ou biblioteca equivalente estável) para gráficos. Não é necessário login nem serviços em nuvem.

**Separação de papéis:** o frontend exibe e ensina; o backend calcula tudo, divide os dados e treina a rede neural. Nunca treinar uma MLP usando valores aleatórios de demonstração ao se apresentar como resultado real.

**Escopo:** experimento acadêmico local, não plataforma de inferência em produção.

## Validação rápida deste kit

```bash
python3 scripts/inspecionar_base.py
```

O resultado deve apontar **10.000 linhas, 13 colunas e 5.000 exemplos de cada classe**. Isso **não** valida o playground: é apenas uma referência da fonte de dados.

## Ponto crítico

O PDF lista os tópicos e ferramentas, mas **não define hiperparâmetros, nota mínima, desenho de tela, nem formato de entrega final**. Esses elementos neste ZIP são propostas para aprendizagem e implementação, e devem ser tratados como tal.
