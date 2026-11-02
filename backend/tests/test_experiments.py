import json
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.dataset import FEATURES
from app.services.experiments import Repository
from app.services.neural import (
    ModelConfig,
    build_model,
    classification_metrics,
    describe_model,
    probabilities,
    tf,
)


def wait(client, run_id):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        response = client.get(f"/api/experiments/{run_id}")
        assert response.status_code == 200
        run = response.json()
        if run["status"] not in ("queued", "running"):
            assert run["status"] == "completed", run
            return run
        time.sleep(0.1)
    pytest.fail("O treino real excedeu 60 segundos.")


@pytest.fixture(scope="module")
def laboratory(tmp_path_factory):
    root = tmp_path_factory.mktemp("learning")
    with TestClient(create_app(storage_path=root, enable_training=True)) as client:
        config = {
            "name": "Teste real A", "model": {"hidden_layers": [8], "l2": 0.001},
            "training": {"epochs": 2, "early_stopping": False, "batch_size": 128},
        }
        created = client.post("/api/experiments", json=config)
        assert created.status_code == 202, created.text
        first = wait(client, created.json()["id"])
        config["name"] = "Teste real B"
        config["model"]["hidden_layers"] = [12, 4]
        created = client.post("/api/experiments", json=config)
        assert created.status_code == 202
        second = wait(client, created.json()["id"])
        yield client, first, second, root


def test_model_count_optimizers_and_l2_are_real():
    for optimizer in ("adam", "sgd", "rmsprop"):
        config = ModelConfig(hidden_layers=[8, 4], optimizer=optimizer, l2=0.01)
        model = build_model(13, config, 42)
        assert model.count_params() == describe_model(13, config)["total_parameters"]
        assert model.output_shape == (None, 1)
        assert model.layers[-1].activation.__name__ == "sigmoid"
        assert model.optimizer.__class__.__name__.lower() == optimizer
        assert len(model.losses) == 2
        assert float(tf.add_n(model.losses)) > 0


def test_real_mlp_history_artifacts_and_reload(laboratory):
    client, first, second, root = laboratory
    assert client.get("/api/health").json()["ml_engine"] == "tensorflow"
    for run in (first, second):
        assert len(run["history"]) == 2
        assert run["saved_epoch"] == 2
        assert run["test_metrics"] is None
        assert run["validation_metrics"]["samples"] == 1500
        for epoch in run["history"]:
            assert all(np.isfinite(epoch[key]) for key in (
                "loss", "val_loss", "accuracy", "val_accuracy",
            ))
        model = tf.keras.models.load_model(root / "artifacts" / run["id"] / "model.keras")
        assert model.count_params() == run["model_summary"]["total_parameters"]
        instance = client.app.state.experiments
        saved = instance.repo.get(run["id"])
        validation = instance.dataset.df.set_index("sample_id", drop=False).loc[
            saved["partition_ids"]["val"]
        ]
        pipeline = joblib.load(root / "artifacts" / run["id"] / "pipeline.joblib")
        recalculated = classification_metrics(
            validation.target, probabilities(model, pipeline.transform(validation[FEATURES])),
            run["config"]["threshold"],
        )
        assert recalculated["confusion_matrix"] == run["validation_metrics"]["confusion_matrix"]
        assert recalculated["accuracy"] == run["validation_metrics"]["accuracy"]
    assert first["model_summary"]["total_parameters"] != second["model_summary"]["total_parameters"]
    assert first["history"] != second["history"]
    assert first["partition_key"] == second["partition_key"]


def test_exports_are_actual_saved_results(laboratory):
    client, first, _, _ = laboratory
    base = f"/api/experiments/{first['id']}/export"
    data = client.get(base).json()
    assert data["history"] == first["history"]
    assert set(data["partition_ids"]) == {"train", "val", "test"}
    assert client.get(base + "?format=csv").text.count("\n") == 3
    assert "Validação do modelo salvo" in client.get(base + "?format=markdown").text
    code = client.get(base + "?format=python").text
    compile(code, "reproduce.py", "exec")
    assert "model.fit(" in code
    assert "transform(test" not in code


def test_model_preview_matches_training(laboratory):
    client, first, _, _ = laboratory
    response = client.post("/api/model/preview", json=first["config"])
    assert response.status_code == 200
    assert response.json()["total_parameters"] == first["model_summary"]["total_parameters"]


@pytest.mark.parametrize("patch", [
    {"model": {"hidden_layers": [0]}}, {"model": {"hidden_layers": [8] * 5}},
    {"model": {"learning_rate": 0}}, {"model": {"optimizer": "fake"}},
    {"training": {"epochs": 201}}, {"training": {"epochs": 2, "patience": 8}},
    {"training": {"batch_size": 1}}, {"threshold": 0},
])
def test_bad_config_never_starts_a_job(laboratory, patch):
    client, _, _, _ = laboratory
    assert client.post("/api/experiments", json=patch).status_code == 422


def test_partition_reservation_prevents_test_reuse(laboratory):
    client, first, _, _ = laboratory
    altered = {**first["config"], "preprocess": {"seed": 99}}
    response = client.post("/api/experiments", json=altered)
    assert response.status_code == 409
    assert "partições" in response.json()["detail"]
    assert client.get("/api/experiments/missing").status_code == 404


def test_confusion_metrics_from_known_counts():
    result = classification_metrics([0, 0, 0, 1, 1], [0.1, 0.2, 0.8, 0.2, 0.9], 0.5)
    assert result["confusion_matrix"] == [[2, 1], [1, 1]]
    assert result["accuracy"] == pytest.approx(3 / 5)
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5
    assert result["specificity"] == pytest.approx(2 / 3)
    no_positive = classification_metrics([0, 1], [0.1, 0.1], 0.5)
    assert no_positive["precision"] == 0
    assert "precision" in no_positive["undefined_metrics"]


def test_final_lock_concurrency_idempotence_and_persistence(laboratory):
    client, first, second, root = laboratory
    final = {"experiment_id": first["id"], "confirm": True}
    assert client.post("/api/evaluation/final", json=final).status_code == 409
    assert client.post("/api/study/select", json={"experiment_id": first["id"]}).status_code == 200
    with ThreadPoolExecutor(2) as pool:
        responses = list(pool.map(
            lambda _: client.post("/api/evaluation/final", json=final), range(2),
        ))
    assert any(response.status_code == 200 for response in responses)
    assert all(response.status_code in (200, 409) for response in responses)
    result = client.get("/api/evaluation/final").json()
    assert result["status"] == "completed"
    metrics = result["result"]
    assert sum(sum(row) for row in metrics["confusion_matrix"]) == 1500
    assert metrics["accuracy"] == pytest.approx((metrics["tn"] + metrics["tp"]) / 1500)
    assert client.post("/api/evaluation/final", json=final).json() == result
    assert client.post("/api/study/select", json={"experiment_id": second["id"]}).status_code == 409
    assert client.post("/api/experiments", json=first["config"]).status_code == 409
    assert Repository(root).study() == result


def test_restart_marks_unfinished_runs_interrupted(tmp_path):
    repo = Repository(tmp_path)
    for status in ("queued", "running"):
        with repo.connect(write=True) as db:
            repo.put(db, {"id": status, "status": status, "history": [{"epoch": 1}]})
    repo.recover()
    assert all(run["status"] == "interrupted" for run in repo.list())
    assert all(run["history"] == [{"epoch": 1}] for run in repo.list())


def test_failure_is_persisted_and_does_not_claim_completion(tmp_path, monkeypatch):
    import app.services.experiments as module

    def failure(*args, **kwargs):
        raise RuntimeError("Falha controlada no teste do worker")

    monkeypatch.setattr(module, "build_model", failure)
    with TestClient(create_app(storage_path=tmp_path, enable_training=True)) as client:
        created = client.post("/api/experiments", json={})
        run_id = created.json()["id"]
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            run = client.get(f"/api/experiments/{run_id}").json()
            if run["status"] == "failed":
                break
            time.sleep(0.1)
        assert run["status"] == "failed"
        assert run["validation_metrics"] is None
        assert "Não foi possível" in run["error"]


def test_study_snapshot_is_json_and_not_an_editable_config(laboratory):
    client, first, _, _ = laboratory
    assert "column" not in first["config"]["preprocess"]
    json.dumps(client.get("/api/study").json(), allow_nan=False)


def test_early_stopping_restores_real_best_weights(tmp_path):
    with TestClient(create_app(storage_path=tmp_path, enable_training=True)) as client:
        response = client.post("/api/experiments", json={
            "model": {"hidden_layers": [4], "learning_rate": 1e-12},
            "training": {"epochs": 5, "early_stopping": True, "patience": 1, "batch_size": 256},
        })
        assert response.status_code == 202
        run = wait(client, response.json()["id"])
        assert run["completed_epochs"] < 5
        assert run["saved_epoch"] == run["best_epoch"] == 1
        assert run["validation_metrics"]["loss"] == pytest.approx(
            run["history"][0]["val_loss"], rel=1e-5,
        )


def test_job_queue_is_bounded_and_api_remains_responsive(tmp_path, monkeypatch):
    import app.services.experiments as module

    entered, release = Event(), Event()
    original = module.build_model

    def slow_build(*args, **kwargs):
        entered.set()
        assert release.wait(15)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "build_model", slow_build)
    with TestClient(create_app(storage_path=tmp_path, enable_training=True)) as client:
        config = {"training": {"epochs": 1, "early_stopping": False, "batch_size": 256}}
        first = client.post("/api/experiments", json=config)
        assert first.status_code == 202
        try:
            assert entered.wait(10)
            started = time.monotonic()
            assert client.get("/api/health").status_code == 200
            assert time.monotonic() - started < 2
            second = client.post("/api/experiments", json=config)
            assert second.status_code == 202
            assert client.post("/api/experiments", json=config).status_code == 409
            queued = client.get(f"/api/experiments/{second.json()['id']}").json()
            assert queued["status"] == "queued"
        finally:
            release.set()
        wait(client, first.json()["id"])
        wait(client, second.json()["id"])


def test_restart_keeps_completed_run_and_can_evaluate_saved_artifacts(tmp_path):
    with TestClient(create_app(storage_path=tmp_path, enable_training=True)) as client:
        response = client.post("/api/experiments", json={
            "training": {"epochs": 1, "early_stopping": False, "batch_size": 256},
        })
        first = wait(client, response.json()["id"])
    with TestClient(create_app(storage_path=tmp_path, enable_training=True)) as restarted:
        run = restarted.get(f"/api/experiments/{first['id']}").json()
        assert run["history"] == first["history"]
        assert run["status"] == "completed"
        selection = restarted.post("/api/study/select", json={"experiment_id": run["id"]})
        assert selection.status_code == 200
        result = restarted.post("/api/evaluation/final", json={
            "experiment_id": run["id"], "confirm": True,
        })
        assert result.status_code == 200
        assert result.json()["result"]["samples"] == 1500
