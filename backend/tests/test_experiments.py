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
from app.services.preprocessing import PreparationSettings


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
    assert "Relatório de classificação" in client.get(base + "?format=markdown").text
    code = client.get(base + "?format=python").text
    compile(code, "reproduce.py", "exec")
    assert "model.fit(" in code
    assert "transform(test" not in code


def test_model_preview_matches_training(laboratory):
    client, first, _, _ = laboratory
    response = client.post("/api/model/preview", json=first["config"])
    assert response.status_code == 200
    assert response.json()["total_parameters"] == first["model_summary"]["total_parameters"]


def test_saved_model_inference_is_real_and_never_loads_test_samples(laboratory):
    client, first, _, root = laboratory
    url = f"/api/experiments/{first['id']}"
    sample = client.get(url + "/sample?subset=val&offset=0").json()
    assert client.get(url + "/sample?subset=test").status_code == 422
    scenario = {"numeric": sample["numeric"], "environment": sample["environment"]}
    response = client.post(url + "/predict", json={"scenarios": [scenario, scenario]})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["scenarios"][0]["score"] == result["scenarios"][1]["score"]
    saved = client.app.state.experiments.repo.get(first["id"])
    row = client.app.state.experiments.dataset.df.set_index("sample_id", drop=False).loc[
        [saved["partition_ids"]["val"][0]]
    ]
    pipeline = joblib.load(root / "artifacts" / first["id"] / "pipeline.joblib")
    model = tf.keras.models.load_model(root / "artifacts" / first["id"] / "model.keras")
    expected = probabilities(model, pipeline.transform(row[FEATURES]))[0]
    assert result["scenarios"][0]["score"] == pytest.approx(expected, abs=1e-6)
    assert result["scenarios"][0]["predicted_class"] == int(expected >= result["threshold"])


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
    report = result["classification_report"]
    assert report["0"]["support"] == 3
    assert report["1"]["support"] == 2
    assert report["0"]["precision"] == pytest.approx(2 / 3)
    assert report["0"]["recall"] == pytest.approx(2 / 3)
    assert report["macro avg"]["f1-score"] == pytest.approx((2 / 3 + 0.5) / 2)
    assert report["weighted avg"]["f1-score"] == pytest.approx((3 * 2 / 3 + 2 * 0.5) / 5)
    no_positive = classification_metrics([0, 1], [0.1, 0.1], 0.5)
    assert no_positive["precision"] == 0
    assert "precision" in no_positive["undefined_metrics"]
    assert no_positive["classification_report"]["1"]["precision"] == 0
    assert result["negative_predictive_value"] == pytest.approx(2 / 3)
    no_negative = classification_metrics([0, 1], [0.9, 0.9], 0.5)
    assert no_negative["negative_predictive_value"] == 0
    assert "negative_predictive_value" in no_negative["undefined_metrics"]


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


def test_custom_split_saved_weights_metrics_and_formatted_reports(tmp_path):
    from app.services.neural import batches

    with TestClient(create_app(storage_path=tmp_path, enable_training=True)) as client:
        config = {
            "name": '<script>alert("teste")</script>',
            "preprocess": {"split_strategy": "custom", "train_percent": 60,
                           "val_percent": 20, "test_percent": 20},
            "model": {"hidden_layers": [4], "l2": 0.01},
            "training": {"epochs": 2, "early_stopping": False, "batch_size": 256},
        }
        bad = {**config, "preprocess": {**config["preprocess"], "train_percent": 70}}
        assert client.post("/api/experiments", json=bad).status_code == 422
        created = client.post("/api/experiments", json=config)
        assert created.status_code == 202
        run = wait(client, created.json()["id"])
        assert run["split_counts"] == {"train": 6000, "val": 2000, "test": 2000}
        base = f"/api/experiments/{run['id']}/export"
        pending = client.get(base + "?format=html").text
        pending_markdown = client.get(base + "?format=markdown").text
        for content in (pending, pending_markdown):
            assert "Relatório parcial" not in content
            assert "Teste reservado" not in content
            assert "Selecionar para teste" not in content
            assert "ANN Lab" not in content
        assert "<h1>" not in pending
        assert pending.count("<svg ") == 2
        assert pending.count("<table>") == 2
        for content in (pending, pending_markdown):
            assert "Relatório de classificação — validação" in content
            assert "Matriz de confusão — validação" in content
            for label in ("Precision", "Recall", "F1-score", "Support", "Acurácia",
                          "Média macro", "Média ponderada"):
                assert label in content
        val = run["validation_metrics"]
        accuracy = f"{val['accuracy'] * 100:.2f}%".replace(".", ",")
        assert f"| Acurácia | — | — | {accuracy} | {val['samples']} |" in pending_markdown
        assert f"| Classe 0 | {val['tn']} | {val['fp']} |" in pending_markdown
        assert f"| Classe 1 | {val['fn']} | {val['tp']} |" in pending_markdown
        assert client.get("/api/study").json()["status"] == "open"
        assert client.get(f"/api/experiments/{run['id']}").json()["test_metrics"] is None
        client.post("/api/study/select", json={"experiment_id": run["id"]})
        response = client.post("/api/evaluation/final", json={
            "experiment_id": run["id"], "confirm": True,
        })
        assert response.status_code == 200
        run = client.get(f"/api/experiments/{run['id']}").json()
        instance = client.app.state.experiments
        model = tf.keras.models.load_model(tmp_path / "artifacts" / run["id"] / "model.keras")
        prepared = instance.preparation.prepare(
            PreparationSettings.model_validate(run["config"]["preprocess"])
        )
        for part, key in [("train", "train_metrics"), ("val", "validation_metrics"),
                          ("test", "test_metrics")]:
            frame = prepared.partitions[part]
            transformed = prepared.pipeline.transform(frame[FEATURES])
            measured = model.evaluate(batches(transformed, frame.target, 256, 42),
                                      return_dict=True, verbose=0)
            metrics = run[key]
            assert metrics["loss"] == pytest.approx(measured["loss"])
            assert metrics["accuracy"] == pytest.approx(measured["accuracy"])
            assert metrics["loss"] > metrics["predictive_loss"]
            assert metrics["classification_report"]["macro avg"]["support"] == len(frame)
        html = client.get(base + "?format=html")
        assert html.headers["content-type"].startswith("text/html")
        assert "<script>" not in html.text
        assert "alert(" not in html.text
        assert html.text.count("<svg ") == 2
        assert "Relatório parcial" not in html.text
        assert "Relatório de classificação — teste final" in html.text
        assert "Matriz de confusão — teste final" in html.text
        assert "Relatório de classificação — validação" not in html.text
        for label in ["Categoria / Métrica", "Média macro", "Média ponderada",
                      "Support", "Matriz de confusão",
                      "Acurácia × Épocas", "Loss × Épocas"]:
            assert label in html.text
            assert label in client.get(base + "?format=markdown").text
        markdown = client.get(base + "?format=markdown").text
        assert html.text.count("<h2>") == 4
        assert html.text.count("<table>") == 2
        assert sum(line.startswith("## ") for line in markdown.splitlines()) == 4
        assert "| Acurácia | — | — |" in markdown
        for content in (html.text, markdown):
            for extra in ("Resumo do desempenho", "Divisão dos dados", "Configuração",
                          "SHA-256", "Especificidade", "Perda preditiva", "Status:"):
                assert extra not in content
