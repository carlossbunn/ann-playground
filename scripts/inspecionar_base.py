#!/usr/bin/env python3
"""Audita o CSV original sem depender de bibliotecas externas."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / 'data' / 'base_classificacao_binaria.csv'
PROFILE_PATH = ROOT / 'data' / 'perfil_base.json'


def main() -> None:
    with CSV_PATH.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    assert rows and columns, 'CSV vazio'
    classes = dict(sorted(Counter(row['target'] for row in rows).items()))
    missing = {col: sum(not row[col].strip() for row in rows) for col in columns}
    env = dict(sorted(Counter(row['environment'] or '(ausente)' for row in rows).items()))
    profile = json.loads(PROFILE_PATH.read_text(encoding='utf-8'))

    assert len(rows) == profile['row_count'], 'Total de linhas divergente'
    assert columns == profile['columns'], 'Colunas divergentes'
    assert classes == profile['target_distribution'], 'Classes divergentes'
    assert missing == {k: v['count'] for k, v in profile['missing_by_column'].items()}, 'Faltantes divergentes'

    print(f'Dataset: {CSV_PATH.name}')
    print(f'Linhas: {len(rows)} | Colunas: {len(columns)}')
    print('Rótulos:', classes)
    print('Ambientes:', env)
    print('Faltantes por coluna:', missing)
    print('Perfil OK: dados e contagens conferem.')


if __name__ == '__main__':
    main()
