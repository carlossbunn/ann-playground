"""Preparação reprodutível: fit no treino, prévias de treino/val e teste reservado."""

from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from threading import Lock
from typing import Literal

import numpy as np
import pandas as pd
import sklearn
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from app.services.dataset import FEATURES, NUMERIC_FEATURES, DatasetService


class PreparationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    missing_strategy: Literal["drop", "mean", "median", "knn"] = "median"
    scaler: Literal["standard", "minmax", "none"] = "standard"
    split_strategy: Literal["70_15_15", "80_20", "custom"] = "70_15_15"
    train_percent: float = Field(default=70, gt=0, lt=100, allow_inf_nan=False)
    val_percent: float = Field(default=15, gt=0, lt=100, allow_inf_nan=False)
    test_percent: float = Field(default=15, gt=0, lt=100, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2**32 - 1, strict=True)
    knn_neighbors: Literal[3, 5, 7] = 5
    categorical_strategy: Literal["most_frequent", "constant"] = "most_frequent"

    @model_validator(mode="after")
    def valid_percentages(self):
        if self.split_strategy == "custom" and abs(
            self.train_percent + self.val_percent + self.test_percent - 100
        ) > 1e-8:
            raise ValueError("Os percentuais de treino, validação e teste devem somar 100%.")
        return self

    def split_percentages(self):
        if self.split_strategy == "custom":
            return self.train_percent, self.val_percent, self.test_percent
        return (70, 15, 15) if self.split_strategy == "70_15_15" else (64, 16, 20)

    def pipeline_settings(self) -> dict:
        return self.model_dump()


class PreprocessingConfig(PreparationSettings):
    column: str = "sensor_a"
    subset: Literal["train", "val"] = "train"
    sample_limit: int = Field(default=12, ge=1, le=100, strict=True)
    bins: Literal[10, 20, 30] = 20

    @field_validator("column")
    @classmethod
    def numeric_column(cls, value: str) -> str:
        if value not in NUMERIC_FEATURES:
            raise ValueError("Selecione um atributo numérico para a comparação.")
        return value

    def pipeline_settings(self) -> dict:
        return self.model_dump(exclude={"column", "subset", "sample_limit", "bins"})


class ScaledKNNImputer(TransformerMixin, BaseEstimator):
    """Calcula vizinhança em z-scores e devolve valores nas unidades originais."""

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler().fit(X)
        self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors).fit(self.scaler_.transform(X))
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        scaled = self.scaler_.transform(X)
        return self.scaler_.inverse_transform(self.imputer_.transform(scaled))

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)


def build_pipeline(config: PreprocessingConfig) -> ColumnTransformer:
    if config.missing_strategy == "drop":
        imputer = "passthrough"
    elif config.missing_strategy == "knn":
        imputer = ScaledKNNImputer(n_neighbors=config.knn_neighbors)
    else:
        imputer = SimpleImputer(strategy=config.missing_strategy, keep_empty_features=True)
    scaler = {
        "standard": StandardScaler(), "minmax": MinMaxScaler(), "none": "passthrough",
    }[config.scaler]
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", imputer), ("scaler", scaler)]), NUMERIC_FEATURES),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(
                strategy=config.categorical_strategy, fill_value="missing",
                keep_empty_features=True,
            )),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), ["environment"]),
    ], remainder="drop")


def split_cohort(df: pd.DataFrame, config: PreprocessingConfig) -> dict[str, pd.DataFrame]:
    cohort = df.dropna(subset=FEATURES) if config.missing_strategy == "drop" else df
    if len(cohort) < 20 or cohort.target.value_counts().min() < 5:
        raise ValueError("A coorte é pequena demais para três partições estratificadas.")
    _, val_percent, test_percent = config.split_percentages()
    test_size = test_percent / 100
    try:
        remaining, test = train_test_split(
            cohort, test_size=test_size, stratify=cohort.target, random_state=config.seed,
        )
        val_size = val_percent / (100 - test_percent)
        train, val = train_test_split(
            remaining, test_size=val_size, stratify=remaining.target, random_state=config.seed,
        )
    except ValueError as error:
        raise ValueError(
            "Não foi possível estratificar a coorte; confira o tamanho das classes."
        ) from error
    if train[FEATURES].isna().all().any():
        raise ValueError("Há um atributo inteiramente ausente no treino; não é possível estimá-lo.")
    return {"train": train, "val": val, "test": test}


@dataclass
class PreparedData:
    partitions: dict[str, pd.DataFrame]
    pipeline: ColumnTransformer
    transformed: dict[str, np.ndarray]
    imputed_numeric: dict[str, np.ndarray]
    categorical_values: dict[str, np.ndarray]


def statistics(values: np.ndarray) -> dict:
    observed = values[np.isfinite(values)]
    return {
        "observed": len(observed), "missing": int(np.isnan(values).sum()),
        **{name: float(operation(observed)) if len(observed) else None for name, operation in [
            ("mean", np.mean), ("median", np.median), ("std", np.std),
            ("min", np.min), ("max", np.max),
        ]},
    }


def histogram(values: np.ndarray, bins: int) -> list[dict]:
    observed = values[np.isfinite(values)]
    if not len(observed):
        return []
    counts, edges = np.histogram(observed, bins=bins)
    return [{"bin_start": float(edges[i]), "bin_end": float(edges[i + 1]), "count": int(n)}
            for i, n in enumerate(counts)]


class PreprocessingService:
    def __init__(self, dataset: DatasetService):
        self.dataset = dataset
        # Um cálculo por vez; no máximo oito pipelines em memória. Sem alterar o CSV.
        self._lock = Lock()
        self._cache: OrderedDict[tuple, PreparedData] = OrderedDict()

    def prepare(self, config: PreprocessingConfig) -> PreparedData:
        key = tuple(config.pipeline_settings().items())
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            partitions = split_cohort(self.dataset.df, config)
            pipeline = build_pipeline(config)
            # Limita os blocos de distâncias de KNN. O teste NÃO passa por transform nesta fase.
            with sklearn.config_context(working_memory=32):
                transformed = {"train": pipeline.fit_transform(partitions["train"][FEATURES])}
                transformed["val"] = pipeline.transform(partitions["val"][FEATURES])
            if any(not np.isfinite(array).all() for array in transformed.values()):
                raise ValueError("A preparação produziu valores indefinidos; confira os dados.")
            imputed = {}
            categorical = {}
            scaler = pipeline.named_transformers_["numeric"].named_steps["scaler"]
            categorical_imputer = pipeline.named_transformers_["categorical"].named_steps["imputer"]
            for name in ("train", "val"):
                numeric = transformed[name][:, :len(NUMERIC_FEATURES)]
                imputed[name] = (
                    numeric.copy() if scaler == "passthrough"
                    else scaler.inverse_transform(numeric)
                )
                categorical[name] = categorical_imputer.transform(partitions[name][["environment"]])
            result = PreparedData(partitions, pipeline, transformed, imputed, categorical)
            self._cache[key] = result
            if len(self._cache) > 8:
                self._cache.popitem(last=False)
            return result

    def preview(self, config: PreprocessingConfig) -> dict:
        prepared = self.prepare(config)
        df = self.dataset.df
        partitions = prepared.partitions
        retained = sum(len(frame) for frame in partitions.values())
        selected = partitions[config.subset]
        position = NUMERIC_FEATURES.index(config.column)
        raw = selected[config.column].to_numpy(dtype=float)
        imputed = prepared.imputed_numeric[config.subset][:, position]
        scaled = prepared.transformed[config.subset][:, position]
        # Ausentes primeiro, depois observados; escolha determinística e rotulada na UI.
        indexes = np.concatenate([np.flatnonzero(np.isnan(raw)), np.flatnonzero(~np.isnan(raw))])
        indexes = indexes[:config.sample_limit]
        samples = [{
            "sample_id": str(selected.iloc[i].sample_id),
            "before": None if np.isnan(raw[i]) else float(raw[i]),
            "imputed": float(imputed[i]), "scaled": float(scaled[i]),
        } for i in indexes]
        numeric_pipeline = prepared.pipeline.named_transformers_["numeric"]
        fitted_imputer = numeric_pipeline.named_steps["imputer"]
        fitted_scaler = numeric_pipeline.named_steps["scaler"]
        replacement = None
        if config.missing_strategy in ("mean", "median"):
            replacement = float(fitted_imputer.statistics_[position])
        environment = prepared.pipeline.named_transformers_["categorical"]
        categories = environment.named_steps["encoder"].categories_[0].tolist()
        warnings = []
        if config.missing_strategy == "drop":
            warnings.append(
                "A remoção altera a população: as partições usam só registros completos."
            )
        if config.missing_strategy == "knn":
            warnings.append(
                "KNN usa padronização interna ajustada só no treino para medir distâncias; "
                "a imputação é devolvida à escala original antes do scaler escolhido."
            )
        if config.scaler == "none":
            warnings.append("Sem escalonamento é um experimento didático adicional.")
        if config.split_strategy == "80_20":
            warnings.append("80/20 reserva 20% dos 80% para validação: divisão efetiva 64/16/20.")
        if config.scaler == "minmax":
            warnings.append(
                "MinMax usa extremos do treino; valores de validação podem sair de [0, 1]."
            )
        return {
            "config": config.model_dump(), "dataset_rows": len(df),
            "rows_after_filter": retained, "removed_rows": len(df) - retained,
            "coverage_percent": retained / len(df) * 100,
            "cohort_by_class": {
                str(label): {
                    "original": int((df.target == label).sum()),
                    "retained": sum(
                        int((frame.target == label).sum()) for frame in partitions.values()
                    ),
                } for label in (0, 1)
            },
            "split_counts": {name: len(frame) for name, frame in partitions.items()},
            "split_classes": {
                name: {str(label): int((frame.target == label).sum()) for label in (0, 1)}
                for name, frame in partitions.items()
            },
            "partition_fingerprints": {
                name: sha256("\n".join(frame.sample_id.astype(str)).encode()).hexdigest()
                for name, frame in partitions.items()
            },
            "dataset_sha256": self.dataset.file_hash, "sklearn_version": sklearn.__version__,
            "fitted_on": "train_only", "preview_subset": config.subset,
            "test_transformed": False, "input_features": FEATURES,
            "output_features": prepared.pipeline.get_feature_names_out().tolist(),
            "output_feature_count": prepared.transformed["train"].shape[1],
            "numeric_before_after": [{
                "column": config.column, "samples": samples,
                "before_stats": statistics(raw), "imputed_stats": statistics(imputed),
                "scaled_stats": statistics(scaled), "fitted_replacement": replacement,
                "standard_scaler": {
                    "mean": float(fitted_scaler.mean_[position]),
                    "std": float(np.sqrt(fitted_scaler.var_[position])),
                    "scale": float(fitted_scaler.scale_[position]),
                } if config.scaler == "standard" else None,
                "before_histogram": histogram(raw, config.bins),
                "imputed_histogram": histogram(imputed, config.bins),
                "scaled_histogram": histogram(scaled, config.bins),
            }],
            "missing_before_after": {
                **{
                    column: {
                        "before": int(selected[column].isna().sum()),
                        "after": int(np.isnan(prepared.imputed_numeric[config.subset][:, i]).sum()),
                    }
                    for i, column in enumerate(NUMERIC_FEATURES)
                },
                "environment": {
                    "before": int(selected.environment.isna().sum()),
                    "after": int(pd.isna(prepared.categorical_values[config.subset]).sum()),
                },
            },
            "environment": {
                "strategy": config.categorical_strategy, "categories": categories,
                "fill_value": str(environment.named_steps["imputer"].statistics_[0]),
                "before_missing": int(selected.environment.isna().sum()),
                "after_missing": int(pd.isna(prepared.categorical_values[config.subset]).sum()),
                "unknown_handling": "ignore",
                "encoded_counts": {
                    str(category): int(
                        (prepared.categorical_values[config.subset] == category).sum()
                    )
                    for category in categories
                },
            },
            "warnings": warnings, "python_code": python_code(config),
        }


def python_code(config: PreprocessingConfig) -> str:
    _, val_percent, test_percent = config.split_percentages()
    test_size = test_percent / 100
    val_size = val_percent / (100 - test_percent)
    filtering = "df.dropna(subset=FEATURES)" if config.missing_strategy == "drop" else "df"
    return f'''from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path("backend").resolve()))
from app.services.dataset import FEATURES
from app.services.preprocessing import PreprocessingConfig, build_pipeline

df = pd.read_csv("data/base_classificacao_binaria.csv")
config = PreprocessingConfig(**{config.pipeline_settings()!r})
cohort = {filtering}
remaining, test = train_test_split(
    cohort, test_size={test_size}, stratify=cohort.target, random_state=config.seed,
)
train, val = train_test_split(
    remaining, test_size={val_size}, stratify=remaining.target, random_state=config.seed,
)
pipeline = build_pipeline(config)
X_train = pipeline.fit_transform(train[FEATURES])  # ÚNICO fit: treino
X_val = pipeline.transform(val[FEATURES])
# test permanece bruto e reservado até a avaliação final.
print(pipeline)  # exibe imputador, scaler e OneHotEncoder reais
print(X_train.shape, X_val.shape)
print({{name: part.sample_id.tolist() for name, part in
       [("train", train), ("val", val), ("test", test)]}})
'''
