import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.dataset import DEFAULT_CSV, NUMERIC_FEATURES, ROOT, DatasetService


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


def test_profile_matches_original_csv_and_reference(client):
    response = client.get("/api/dataset/profile")
    assert response.status_code == 200
    profile = response.json()
    reference = json.loads((ROOT / "data/perfil_base.json").read_text())
    assert profile["rows"] == reference["row_count"] == 10000
    assert profile["columns"] == reference["column_count"] == 13
    assert profile["column_names"] == reference["columns"]
    assert profile["target_distribution"] == reference["target_distribution"]
    assert len(profile["numeric_columns"]) == 10
    assert profile["categorical_columns"] == ["environment"]
    for column, expected in reference["missing_by_column"].items():
        actual = profile["missing_by_column"][column]
        assert actual["count"] == expected["count"]
        assert actual["percent"] == pytest.approx(expected["percent"], abs=0.01)
    for column, expected in reference["numeric_summary"].items():
        for key in ("min", "max", "mean", "median"):
            assert profile["numeric_summary"][column][key] == pytest.approx(expected[key], abs=1e-6)
    assert len(profile["sha256"]) == 64


def test_preview_is_paginated_and_missing_values_are_json_null(client):
    page = client.get("/api/dataset/preview?offset=30&limit=30").json()
    original = pd.read_csv(DEFAULT_CSV)
    assert len(page["rows"]) == 30
    assert page["rows"][0]["sample_id"] == original.iloc[30].sample_id
    first_page = client.get("/api/dataset/preview?limit=1")
    assert first_page.json()["rows"][0]["measurement_d"] is None
    assert "NaN" not in first_page.text
    assert client.get("/api/dataset/preview?offset=10000").json()["rows"] == []


@pytest.mark.parametrize("column", NUMERIC_FEATURES)
@pytest.mark.parametrize("bins", [10, 20, 30])
def test_histograms_account_for_every_observed_value(client, column, bins):
    response = client.get(f"/api/dataset/eda?column={column}&bins={bins}")
    assert response.status_code == 200
    result = response.json()
    assert len(result["histogram"]) == bins
    assert sum(bar["count"] for bar in result["histogram"]) + result["missing"] == 10000
    box = result["boxplot"]
    assert box["lower_whisker"] <= box["q1"] <= box["median"] <= box["q3"]
    assert box["q3"] <= box["upper_whisker"]


def test_boxplot_uses_iqr_and_real_outlier_values(tmp_path):
    df = pd.read_csv(DEFAULT_CSV).iloc[:6].copy()
    df["sensor_a"] = [1, 2, 3, 4, 100, np.nan]
    path = tmp_path / "small.csv"
    df.to_csv(path, index=False)
    result = DatasetService(path).eda("sensor_a", 10)
    assert result["missing"] == 1
    box = result["boxplot"]
    assert (box["q1"], box["median"], box["q3"]) == (2, 3, 4)
    assert (box["lower_whisker"], box["upper_whisker"]) == (1, 4)
    assert box["outlier_count"] == 1
    assert box["outlier_points"] == [100]


def test_categories_include_missing_as_a_diagnostic(client):
    data = client.get("/api/dataset/eda?column=environment").json()
    assert data["kind"] == "categorical"
    assert {item["category"]: item["count"] for item in data["categories"]} == {
        "(ausente)": 598, "cold": 3141, "hot": 3130, "mild": 3131,
    }


def test_correlations_exclude_id_and_label_and_report_pairwise_counts(client):
    result = client.get("/api/dataset/correlations").json()
    assert result["columns"] == NUMERIC_FEATURES
    matrix = np.array(result["matrix"])
    assert matrix.shape == (10, 10)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1)
    original = pd.read_csv(DEFAULT_CSV)
    assert result["sample_counts"][0][1] == len(original[["sensor_a", "sensor_b"]].dropna())


def test_undefined_statistics_are_null(tmp_path):
    df = pd.read_csv(DEFAULT_CSV).iloc[:6].copy()
    df["sensor_a"] = np.nan
    path = tmp_path / "empty-column.csv"
    df.to_csv(path, index=False)
    with TestClient(create_app(path)) as client:
        summary = client.get("/api/dataset/profile").json()["numeric_summary"]
        assert summary["sensor_a"]["mean"] is None
        assert client.get("/api/dataset/eda").json()["boxplot"] is None
        assert client.get("/api/dataset/correlations").json()["matrix"][0][1] is None


@pytest.mark.parametrize("path", [
    "/api/dataset/preview?limit=101", "/api/dataset/preview?offset=-1",
    "/api/dataset/preview?limit=0", "/api/dataset/eda?bins=13",
    "/api/dataset/eda?column=sample_id", "/api/dataset/eda?column=inexistente",
])
def test_invalid_inputs_return_portuguese_errors(client, path):
    response = client.get(path)
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], str)


def test_health_does_not_claim_a_training_engine(client):
    health = client.get("/api/health").json()
    assert health["dataset_loaded"] is True
    assert health["training_available"] is False
    assert health["ml_engine"] == "not_initialized"


def test_missing_csv_returns_real_unavailability(tmp_path):
    with TestClient(create_app(tmp_path / "absent.csv")) as client:
        assert client.get("/api/health").status_code == 503
        assert client.get("/api/dataset/profile").status_code == 503


def test_cors_only_allows_local_development_origins(client):
    local = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert local.headers["access-control-allow-origin"] == "http://localhost:5173"
    external = client.get("/api/health", headers={"Origin": "https://example.com"})
    assert "access-control-allow-origin" not in external.headers
