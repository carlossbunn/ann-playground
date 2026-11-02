"""Análise descritiva global. Nenhum transformador ou modelo é ajustado aqui."""

from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV = ROOT / "data" / "base_classificacao_binaria.csv"
NUMERIC_FEATURES = [
    "sensor_a", "sensor_b", "sensor_c", "measurement_d", "measurement_e",
    "measurement_f", "measurement_g", "measurement_h", "measurement_i", "ratio_j",
]
FEATURES = [*NUMERIC_FEATURES, "environment"]
EXPECTED_COLUMNS = ["sample_id", *FEATURES, "target"]


class DatasetService:
    def __init__(self, path: Path = DEFAULT_CSV):
        self.path = path
        self.df = pd.read_csv(path)
        if list(self.df.columns) != EXPECTED_COLUMNS:
            raise ValueError("As colunas do CSV não correspondem ao contrato do ANN Lab.")
        if self.df["target"].isna().any() or set(self.df["target"].unique()) != {0, 1}:
            raise ValueError("A coluna target deve conter apenas as classes 0 e 1.")
        for column in NUMERIC_FEATURES:
            self.df[column] = pd.to_numeric(self.df[column], errors="raise")
            if np.isinf(self.df[column]).any():
                raise ValueError("O CSV contém valores numéricos infinitos.")
        self.file_hash = sha256(path.read_bytes()).hexdigest()

    def categories(self, column: str) -> list[dict]:
        series = self.df[column].fillna("(ausente)").astype(str)
        return [
            {"category": category, "count": int(count)}
            for category, count in series.value_counts().sort_index().items()
        ]

    def profile(self) -> dict:
        df = self.df
        summary = {}
        for column in NUMERIC_FEATURES:
            values = df[column].dropna()
            summary[column] = {
                "n": int(values.count()),
                "min": float(values.min()) if len(values) else None,
                "max": float(values.max()) if len(values) else None,
                "mean": float(values.mean()) if len(values) else None,
                "median": float(values.median()) if len(values) else None,
                "std": float(values.std()) if len(values) > 1 else None,
                "negative_count": int((values < 0).sum()),
            }
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "target_distribution": {
                str(label): int(count)
                for label, count in df.target.value_counts().sort_index().items()
            },
            "missing_by_column": {
                col: {"count": int(df[col].isna().sum()),
                      "percent": float(df[col].isna().mean() * 100)}
                for col in df.columns
            },
            "environment_distribution": self.categories("environment"),
            "numeric_columns": NUMERIC_FEATURES,
            "categorical_columns": ["environment"],
            "id_column": "sample_id",
            "label_column": "target",
            "numeric_summary": summary,
            "duplicate_ids": int(df.sample_id.duplicated().sum()),
            "sha256": self.file_hash,
            "scope": "global_descriptive",
        }

    def preview(self, offset: int, limit: int) -> dict:
        page = self.df.iloc[offset:offset + limit]
        # Convertir para object antes de trocar NaN evita que pandas o reintroduza.
        records = page.astype(object).where(pd.notna(page), None).to_dict(orient="records")
        return {
            "columns": list(self.df.columns), "rows": records,
            "total": len(self.df), "offset": offset, "limit": limit,
        }

    def eda(self, column: str, bins: int) -> dict:
        if column not in [*FEATURES, "target"]:
            raise ValueError("Escolha um atributo válido ou target; sample_id é um identificador.")
        base = {
            "column": column, "scope": "global_descriptive", "total": len(self.df),
            "missing": int(self.df[column].isna().sum()),
        }
        if column in ["environment", "target"]:
            return {**base, "kind": "categorical", "categories": self.categories(column)}
        values = self.df[column].dropna()
        if values.empty:
            return {**base, "kind": "numeric", "histogram": [], "boxplot": None}
        counts, edges = np.histogram(values, bins=bins)
        q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        inliers = values[values.between(lower_fence, upper_fence)]
        outliers = values[~values.between(lower_fence, upper_fence)].sort_values()
        # Pontos reais, amostrados uniformemente só para não sobrecarregar o SVG.
        indexes = np.linspace(0, len(outliers) - 1, min(80, len(outliers)), dtype=int)
        return {
            **base, "kind": "numeric", "bins": bins,
            "histogram": [
                {"bin_start": float(edges[i]), "bin_end": float(edges[i + 1]),
                 "count": int(count)}
                for i, count in enumerate(counts)
            ],
            "boxplot": {
                "min": float(values.min()), "max": float(values.max()),
                "q1": float(q1), "median": float(median), "q3": float(q3),
                "lower_whisker": float(inliers.min()), "upper_whisker": float(inliers.max()),
                "lower_fence": float(lower_fence), "upper_fence": float(upper_fence),
                "outlier_count": len(outliers),
                "outlier_points": outliers.iloc[indexes].tolist(),
            },
        }

    def correlations(self) -> dict:
        numeric = self.df[NUMERIC_FEATURES]
        matrix = numeric.corr(method="pearson")
        present = numeric.notna().astype(int)
        return {
            "columns": NUMERIC_FEATURES,
            "matrix": matrix.astype(object).where(pd.notna(matrix), None).values.tolist(),
            "sample_counts": (present.T @ present).values.tolist(),
            "method": "pearson", "scope": "global_descriptive",
            "excluded_columns": ["sample_id", "target", "environment"],
        }
