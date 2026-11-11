from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.impute import KNNImputer

from app.main import create_app
from app.services.dataset import DEFAULT_CSV, FEATURES, NUMERIC_FEATURES, ROOT, DatasetService
from app.services.preprocessing import (
    PreprocessingConfig,
    PreprocessingService,
    ScaledKNNImputer,
    build_pipeline,
    split_cohort,
)


@pytest.fixture(scope="module")
def service():
    return PreprocessingService(DatasetService())


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.mark.parametrize("strategy,counts", [
    ("70_15_15", {"train": 7000, "val": 1500, "test": 1500}),
    ("80_20", {"train": 6400, "val": 1600, "test": 2000}),
])
def test_split_is_stratified_reproducible_and_disjoint(service, strategy, counts):
    config = PreprocessingConfig(split_strategy=strategy)
    first = split_cohort(service.dataset.df, config)
    second = split_cohort(service.dataset.df, config)
    assert {name: len(frame) for name, frame in first.items()} == counts
    ids = {name: set(frame.sample_id) for name, frame in first.items()}
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]
    assert len(set.union(*ids.values())) == 10000
    for name, frame in first.items():
        assert frame.sample_id.tolist() == second[name].sample_id.tolist()
        assert frame.target.mean() == pytest.approx(0.5, abs=0.001)
    changed = split_cohort(service.dataset.df, config.model_copy(update={"seed": 43}))
    assert set(changed["train"].sample_id) != ids["train"]


@pytest.mark.parametrize("percentages", [(60, 20, 20), (75, 10, 15), (65.5, 14.5, 20)])
def test_custom_split_and_reproduction(service, percentages, monkeypatch):
    from app.services.preprocessing import python_code

    config = PreprocessingConfig(
        split_strategy="custom", train_percent=percentages[0],
        val_percent=percentages[1], test_percent=percentages[2],
    )
    parts = split_cohort(service.dataset.df, config)
    repeated = split_cohort(service.dataset.df, config)
    ids = [set(frame.sample_id) for frame in parts.values()]
    assert len(set.union(*ids)) == sum(map(len, ids)) == 10000
    for (key, frame), percentage in zip(parts.items(), percentages, strict=True):
        assert len(frame) == pytest.approx(100 * percentage, abs=1)
        assert frame.target.mean() == pytest.approx(0.5, abs=0.001)
        assert frame.sample_id.tolist() == repeated[key].sample_id.tolist()
    namespace = {}
    monkeypatch.chdir(ROOT)
    with redirect_stdout(StringIO()):
        exec(compile(python_code(config), "custom_split.py", "exec"), namespace)
    for key in parts:
        assert namespace[key].sample_id.tolist() == parts[key].sample_id.tolist()


@pytest.mark.parametrize("percentages", [(70, 20, 20), (0, 80, 20), (-10, 90, 20),
                                         (100, 0, 0), (float("inf"), 15, 15)])
def test_invalid_custom_percentages(percentages):
    with pytest.raises(ValueError):
        PreprocessingConfig(split_strategy="custom", train_percent=percentages[0],
                            val_percent=percentages[1], test_percent=percentages[2])


@pytest.mark.parametrize("strategy", ["drop", "mean", "median", "knn"])
def test_real_strategies_produce_complete_features_without_changing_original(service, strategy):
    before = sha256(DEFAULT_CSV.read_bytes()).hexdigest()
    config = PreprocessingConfig(missing_strategy=strategy)
    prepared = service.prepare(config)
    assert prepared.transformed["train"].shape[1] == 13
    assert set(prepared.transformed) == {"train", "val"}
    for array in prepared.transformed.values():
        assert np.isfinite(array).all()
    numeric = prepared.partitions["train"][NUMERIC_FEATURES].to_numpy()
    observed = ~np.isnan(numeric)
    assert np.allclose(prepared.imputed_numeric["train"][observed], numeric[observed])
    assert sha256(DEFAULT_CSV.read_bytes()).hexdigest() == before


def test_mean_and_median_use_training_statistics_and_differ(service):
    results = {}
    for strategy in ["mean", "median"]:
        prepared = service.prepare(PreprocessingConfig(missing_strategy=strategy))
        series = prepared.partitions["train"].sensor_b
        expected = getattr(series, strategy)()
        index = NUMERIC_FEATURES.index("sensor_b")
        imputed = prepared.imputed_numeric["train"][:, index]
        assert np.allclose(imputed[series.isna()], expected)
        results[strategy] = expected
    assert results["mean"] != pytest.approx(results["median"])


def test_drop_filters_features_before_split_and_reports_class_coverage(service):
    expected = service.dataset.df.dropna(subset=FEATURES)
    result = service.preview(PreprocessingConfig(missing_strategy="drop"))
    assert result["rows_after_filter"] == len(expected)
    assert result["removed_rows"] == 10000 - len(expected)
    assert sum(result["split_counts"].values()) == len(expected)
    assert result["coverage_percent"] == pytest.approx(len(expected) / 100)
    for label in (0, 1):
        assert result["cohort_by_class"][str(label)]["retained"] == int(
            (expected.target == label).sum()
        )
    prepared = service.prepare(PreprocessingConfig(missing_strategy="drop"))
    assert all(not frame[FEATURES].isna().any().any() for frame in prepared.partitions.values())


@pytest.mark.parametrize("strategy", ["mean", "median", "knn"])
def test_preprocessors_fit_train_only_with_holdout_sentinels(service, strategy):
    # Mesmos índices e y; mudar val/test para extremos não pode mudar imputador, scaler ou encoder.
    df = service.dataset.df.iloc[:200].copy()
    config = PreprocessingConfig(missing_strategy=strategy)
    parts = split_cohort(df, config)
    altered = df.copy()
    holdout = parts["val"].index.union(parts["test"].index)
    altered.loc[holdout, NUMERIC_FEATURES] = 1e12
    altered.loc[holdout, "environment"] = "holdout-only"
    normal_parts = split_cohort(df, config)
    altered_parts = split_cohort(altered, config)
    normal = build_pipeline(config)
    changed = build_pipeline(config)
    first = normal.fit_transform(normal_parts["train"][FEATURES])
    second = changed.fit_transform(altered_parts["train"][FEATURES])
    assert np.allclose(first, second)
    normal_scaler = normal.named_transformers_["numeric"].named_steps["scaler"]
    changed_scaler = changed.named_transformers_["numeric"].named_steps["scaler"]
    assert np.allclose(normal_scaler.mean_, changed_scaler.mean_)
    assert np.all(normal_scaler.n_samples_seen_ == len(parts["train"]))
    categorical = changed.named_transformers_["categorical"]
    assert "holdout-only" not in categorical.named_steps["encoder"].categories_[0]
    unknown = categorical.transform(altered_parts["val"][["environment"]])
    assert not unknown.any()
    if strategy == "knn":
        knn = changed.named_transformers_["numeric"].named_steps["imputer"]
        expected = parts["train"][NUMERIC_FEATURES].mean().to_numpy()
        assert np.allclose(knn.scaler_.mean_, expected)
    else:
        imputer = changed.named_transformers_["numeric"].named_steps["imputer"]
        expected = getattr(parts["train"][NUMERIC_FEATURES], strategy)().to_numpy()
        assert np.allclose(imputer.statistics_, expected)


def test_service_never_transforms_test_and_fit_uses_only_training_rows(monkeypatch):
    # Espiona a integração real, não apenas o construtor do pipeline.
    from sklearn.compose import ColumnTransformer

    dataset = DatasetService()
    dataset.df = dataset.df.iloc[:200].copy()
    config = PreprocessingConfig()
    partitions = split_cohort(dataset.df, config)
    calls = []
    original_fit = ColumnTransformer.fit_transform
    original_transform = ColumnTransformer.transform

    def fit(self, X, *args, **kwargs):
        calls.append(("fit", set(X.index)))
        return original_fit(self, X, *args, **kwargs)

    def transform(self, X, *args, **kwargs):
        calls.append(("transform", set(X.index)))
        return original_transform(self, X, *args, **kwargs)

    monkeypatch.setattr(ColumnTransformer, "fit_transform", fit)
    monkeypatch.setattr(ColumnTransformer, "transform", transform)
    PreprocessingService(dataset).preview(config)
    assert calls == [
        ("fit", set(partitions["train"].index)),
        ("transform", set(partitions["val"].index)),
    ]


@pytest.mark.parametrize("categorical_strategy", ["most_frequent", "constant"])
def test_environment_one_hot_excludes_id_target_and_handles_absence(service, categorical_strategy):
    config = PreprocessingConfig(categorical_strategy=categorical_strategy)
    prepared = service.prepare(config)
    names = prepared.pipeline.get_feature_names_out().tolist()
    assert not any("sample_id" in name or "target" in name for name in names)
    assert names[:10] == [f"numeric__{column}" for column in NUMERIC_FEATURES]
    categories = prepared.pipeline.named_transformers_["categorical"].named_steps["encoder"]
    assert set(categories.categories_[0]) == (
        {"cold", "hot", "mild"} if categorical_strategy == "most_frequent"
        else {"cold", "hot", "mild", "missing"}
    )
    assert np.all(prepared.transformed["train"][:, 10:].sum(axis=1) == 1)


@pytest.mark.parametrize("scaler", ["standard", "minmax", "none"])
def test_scalers_have_real_distinct_effects(service, scaler):
    prepared = service.prepare(PreprocessingConfig(scaler=scaler))
    values = prepared.transformed["train"][:, :10]
    if scaler == "standard":
        assert np.allclose(values.mean(axis=0), 0, atol=1e-12)
        assert np.allclose(values.std(axis=0), 1)
    elif scaler == "minmax":
        assert np.allclose(values.min(axis=0), 0)
        assert np.allclose(values.max(axis=0), 1)
    else:
        assert np.allclose(values, prepared.imputed_numeric["train"])


def test_knn_standardizes_distances_and_returns_original_units():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 10.0], "b": [10.0, 900.0, 1000.0, np.nan]})
    imputer = ScaledKNNImputer(n_neighbors=3).fit(train)
    result = imputer.transform(train)
    expected = KNNImputer(n_neighbors=3).fit_transform(imputer.scaler_.transform(train))
    assert np.allclose(result, imputer.scaler_.inverse_transform(expected))
    assert np.allclose(result[:3], train.to_numpy()[:3])
    assert result[3, 1] == pytest.approx((10 + 900 + 1000) / 3)


def test_preview_reports_training_stats_samples_and_reproducible_manifest(client):
    response = client.post("/api/preprocessing/preview", json={"column": "sensor_b"})
    assert response.status_code == 200
    result = response.json()
    assert result["fitted_on"] == "train_only"
    assert result["test_transformed"] is False
    assert set(result["partition_fingerprints"]) == {"train", "val", "test"}
    assert all(len(digest) == 64 for digest in result["partition_fingerprints"].values())
    comparison = result["numeric_before_after"][0]
    assert len(comparison["samples"]) == 12
    assert comparison["samples"][0]["before"] is None
    assert comparison["imputed_stats"]["missing"] == 0
    assert comparison["scaled_stats"]["std"] == pytest.approx(1)
    assert comparison["scaled_stats"]["mean"] == pytest.approx(0, abs=1e-12)
    assert comparison["standard_scaler"]["mean"] == pytest.approx(
        comparison["imputed_stats"]["mean"]
    )
    assert comparison["standard_scaler"]["std"] == pytest.approx(comparison["imputed_stats"]["std"])
    for sample in comparison["samples"]:
        scaler = comparison["standard_scaler"]
        assert sample["scaled"] == pytest.approx(
            (sample["imputed"] - scaler["mean"]) / scaler["scale"]
        )
    assert sum(bar["count"] for bar in comparison["imputed_histogram"]) == 7000
    assert "pipeline.fit_transform(train[FEATURES])" in result["python_code"]
    assert "pipeline.transform(test" not in result["python_code"]


def test_preview_validation_uses_same_fitted_pipeline(client):
    train = client.post("/api/preprocessing/preview", json={}).json()
    val = client.post("/api/preprocessing/preview", json={"subset": "val"}).json()
    assert train["partition_fingerprints"] == val["partition_fingerprints"]
    assert (train["numeric_before_after"][0]["standard_scaler"]
            == val["numeric_before_after"][0]["standard_scaler"])
    assert val["numeric_before_after"][0]["scaled_stats"]["observed"] == 1500
    assert val["preview_subset"] == "val"


@pytest.mark.parametrize("invalid", [
    {"seed": -1}, {"seed": 2**32}, {"seed": True}, {"seed": 1.5},
    {"knn_neighbors": 0}, {"knn_neighbors": 10000}, {"sample_limit": 101},
    {"missing_strategy": "random"}, {"scaler": "fake"}, {"split_strategy": "50_50"},
    {"subset": "test"}, {"column": "target"}, {"column": "sample_id"},
    {"categorical_strategy": "mean"}, {"unrecognized": 1},
])
def test_invalid_config_is_rejected_before_computation(client, invalid):
    response = client.post("/api/preprocessing/preview", json=invalid)
    assert response.status_code == 422
    assert "Configuração inválida" in response.json()["detail"]


def test_exported_python_code_is_executable_and_matches_real_split(service, monkeypatch):
    config = PreprocessingConfig(split_strategy="80_20")
    result = service.preview(config)
    monkeypatch.chdir(ROOT)
    namespace = {}
    with redirect_stdout(StringIO()):
        exec(compile(result["python_code"], "snippet", "exec"), namespace)
    prepared = service.prepare(config)
    assert np.allclose(namespace["X_train"], prepared.transformed["train"])
    assert np.allclose(namespace["X_val"], prepared.transformed["val"])
    for name in ("train", "val", "test"):
        assert namespace[name].sample_id.tolist() == prepared.partitions[name].sample_id.tolist()
