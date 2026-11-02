# 08 — Dicionário dos dados e observações verificadas

**Arquivo analisado:** `data/base_classificacao_binaria.csv`. Resumo derivado diretamente da leitura do arquivo (detalhe no `data/perfil_base.json`).

| Coluna | Papel | Valores ausentes | Observações seguras |
|---|---|---:|---|
| `sample_id` | ID, não-feature | 0 (0%) | Identificador textual por amostra, ex. `S08268` |
| `sensor_a` | Numérica | 414 (4,14%) | Unidade física desconhecida |
| `sensor_b` | Numérica | 1192 (11,92%) | Unidade física desconhecida |
| `sensor_c` | Numérica | 693 (6,93%) | Unidade física desconhecida |
| `measurement_d` | Numérica | 911 (9,11%) | Unidade física desconhecida |
| `measurement_e` | Numérica | 468 (4,68%) | Unidade física desconhecida |
| `measurement_f` | Numérica | 1011 (10,11%) | Unidade física desconhecida |
| `measurement_g` | Numérica | 603 (6,03%) | Alguns valores negativos são válidos como dado bruto até prova em contrário |
| `measurement_h` | Numérica | 855 (8,55%) | Unidade física desconhecida |
| `measurement_i` | Numérica | 1526 (15,26%) | Variável com maior número de faltantes |
| `ratio_j` | Numérica | 1061 (10,61%) | Razão indicada pelo nome, sem definição documentada |
| `environment` | Categórica | 598 (5,98%) | `cold`: 3141; `mild`: 3131; `hot`: 3130 |
| `target` | Label | 0 (0%) | `0`: 5000; `1`: 5000 |

Total: **10.000 linhas, 13 colunas**, sendo **11 features** (10 numéricas + 1 categórica), 1 identificador e 1 rótulo.

O nome das colunas sugere sensores e medições, mas o arquivo não especifica **qual equipamento**, o **significado de `target`**, as **unidades**, nem se valores negativos são anômalos. Não inventar esse contexto no aplicativo.

## Observações para testes

- 0 IDs duplicados no arquivo original.
- Todas as features, incluindo `environment`, precisam de tratamento adequado de valores ausentes.
- A categoria ausente deve ser incluída no diagnóstico EDA, mas não é uma “quarta categoria real” de ambiente.
- O rótulo é perfeitamente balanceado, mas isso não prova que a classificação seja fácil.

## Script de auditoria

Execute `python3 scripts/inspecionar_base.py` a partir da raiz do kit. Compare saída com `data/perfil_base.json`. O frontend não deve ler esse JSON como se fossem resultados de treino; serve para verificação.
