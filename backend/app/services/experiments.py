"""Experimentos imutáveis, fila única e avaliação holdout transacional."""

import csv
import fcntl
import io
import json
import logging
import platform
import sqlite3
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from threading import Event, Lock, Thread
from uuid import uuid4

import joblib
import numpy as np
import sklearn

from app.services.dataset import FEATURES, DatasetService
from app.services.neural import (
    ExperimentConfig,
    batches,
    build_model,
    classification_metrics,
    describe_model,
    probabilities,
    tf,
)
from app.services.preprocessing import PreprocessingService, python_code, split_cohort

logger = logging.getLogger(__name__)


def now():
    return datetime.now(UTC).isoformat()


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class Conflict(ValueError):
    pass


class Repository:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "experiments.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS runs "
                       "(id TEXT PRIMARY KEY, document TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS study "
                       "(id INTEGER PRIMARY KEY, document TEXT NOT NULL)")
            db.execute("INSERT OR IGNORE INTO study VALUES (1, ?)", (encode({
                "status": "open", "partition_key": None, "preprocess": None,
                "selected_experiment_id": None, "result": None,
            }),))

    @contextmanager
    def connect(self, write=False):
        db = sqlite3.connect(self.path, timeout=15)
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def get(self, run_id, db=None):
        if db is None:
            with self.connect() as connection:
                return self.get(run_id, connection)
        row = db.execute("SELECT document FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError("Experimento não encontrado.")
        return json.loads(row[0])

    def put(self, db, document):
        db.execute("INSERT OR REPLACE INTO runs VALUES (?, ?)",
                   (document["id"], encode(document)))

    def update(self, run_id, **changes):
        with self.connect(write=True) as db:
            document = self.get(run_id, db)
            document.update(changes)
            self.put(db, document)

    def list(self):
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute(
                "SELECT document FROM runs ORDER BY rowid DESC"
            )]

    def study(self, db=None):
        if db is None:
            with self.connect() as connection:
                return self.study(connection)
        return json.loads(db.execute("SELECT document FROM study WHERE id=1").fetchone()[0])

    def put_study(self, db, study):
        db.execute("UPDATE study SET document=? WHERE id=1", (encode(study),))

    def recover(self):
        with self.connect(write=True) as db:
            for row in db.execute("SELECT document FROM runs").fetchall():
                run = json.loads(row[0])
                if run["status"] in ("queued", "running"):
                    run.update(status="interrupted", ended_at=now(),
                               error="A aplicação foi encerrada durante este experimento.")
                    self.put(db, run)
            study = self.study(db)
            if study["status"] == "evaluating":
                study["status"] = "evaluation_failed"
                self.put_study(db, study)


class ExperimentService:
    def __init__(self, dataset: DatasetService, root: Path):
        self.dataset = dataset
        self.preparation = PreprocessingService(dataset)
        self.repo = Repository(root)
        self.process_lock = (root / "trainer.lock").open("a+")
        try:
            fcntl.flock(self.process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.process_lock.close()
            raise RuntimeError(
                "Já existe um motor usando este armazenamento. Inicie apenas um worker da API."
            ) from error
        self.repo.recover()
        self.stop = Event()
        self.wake = Event()
        self.operation_lock = Lock()
        self.worker = Thread(target=self.work, name="ann-lab-trainer", daemon=True)
        self.worker.start()

    def close(self):
        self.stop.set()
        self.wake.set()
        self.worker.join(timeout=20)
        if not self.worker.is_alive():
            fcntl.flock(self.process_lock, fcntl.LOCK_UN)
            self.process_lock.close()

    def create(self, config: ExperimentConfig):
        parts = split_cohort(self.dataset.df, config.preprocess)
        manifest = {name: frame.sample_id.astype(str).tolist() for name, frame in parts.items()}
        key = sha256(encode([self.dataset.file_hash, manifest]).encode()).hexdigest()
        with self.repo.connect(write=True) as db:
            study = self.repo.study(db)
            if study["status"] != "open":
                raise Conflict("O teste deste ciclo já foi aberto. Novos treinos estão encerrados.")
            if study["partition_key"] and study["partition_key"] != key:
                raise Conflict(
                    "Este ciclo já reservou suas partições. Mantenha seed, divisão e coorte "
                    "do primeiro experimento; imputação e modelo podem variar."
                )
            active = sum(json.loads(row[0])["status"] in ("queued", "running")
                         for row in db.execute("SELECT document FROM runs"))
            if active >= 2:
                raise Conflict("Já há um treino e uma espera na fila. Aguarde a conclusão.")
            run_id = str(uuid4())
            run = {
                "id": run_id, "config": config.model_dump(), "status": "queued",
                "created_at": now(), "started_at": None, "ended_at": None,
                "history": [], "completed_epochs": 0, "best_epoch": None,
                "saved_epoch": None, "validation_metrics": None, "test_metrics": None,
                "dataset_sha256": self.dataset.file_hash, "partition_key": key,
                "partition_ids": manifest,
                "split_counts": {name: len(frame) for name, frame in parts.items()},
                "versions": {"python": platform.python_version(), "tensorflow": tf.__version__,
                             "keras": tf.keras.__version__, "sklearn": sklearn.__version__,
                             "numpy": np.__version__},
                "device": "CPU", "model_summary": None, "output_features": [],
                "artifact_hashes": {}, "error": None,
            }
            self.repo.put(db, run)
            if not study["partition_key"]:
                study.update(partition_key=key, preprocess=config.preprocess.pipeline_settings(),
                             dataset_sha256=self.dataset.file_hash)
                self.repo.put_study(db, study)
        self.wake.set()
        return run

    def work(self):
        while not self.stop.is_set():
            queued = [run for run in self.repo.list() if run["status"] == "queued"]
            if not queued:
                self.wake.wait(0.5)
                self.wake.clear()
                continue
            run = queued[-1]
            with self.operation_lock:
                try:
                    self.train(run)
                except Exception:
                    logger.exception("Falha no treino %s", run["id"])
                    self.repo.update(run["id"], status="failed", ended_at=now(), error=(
                        "Não foi possível concluir o treino. Confira os dados e hiperparâmetros; "
                        f"detalhes no log do experimento {run['id']}."
                    ))

    def train(self, run):
        run_id = run["id"]
        config = ExperimentConfig.model_validate(run["config"])
        self.repo.update(run_id, status="running", started_at=now())
        prepared = self.preparation.prepare(config.preprocess)
        tf.keras.backend.clear_session()
        model = build_model(
            prepared.transformed["train"].shape[1], config.model, config.training.seed,
            config.threshold,
        )
        description = describe_model(prepared.transformed["train"].shape[1], config.model)
        if model.count_params() != description["total_parameters"]:
            raise ValueError("Contagem de parâmetros inconsistente.")
        self.repo.update(run_id, model_summary=description,
                         output_features=prepared.pipeline.get_feature_names_out().tolist())
        train = batches(prepared.transformed["train"], prepared.partitions["train"].target,
                        config.training.batch_size, config.training.seed, shuffle=True)
        val = batches(prepared.transformed["val"], prepared.partitions["val"].target,
                      config.training.batch_size, config.training.seed)
        history = []
        service = self

        class Progress(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                values = {name: float(value) for name, value in (logs or {}).items()}
                if not all(np.isfinite(value) for value in values.values()):
                    raise ValueError("A perda divergiu para valores indefinidos.")
                history.append({"epoch": epoch + 1, **values})
                service.repo.update(run_id, history=list(history), completed_epochs=epoch + 1)
                if service.stop.is_set():
                    self.model.stop_training = True

        callbacks = [Progress()]
        if config.training.early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=config.training.patience, restore_best_weights=True,
            ))
        started = time.monotonic()
        model.fit(train, validation_data=val, epochs=config.training.epochs,
                  callbacks=callbacks, verbose=0)
        if self.stop.is_set():
            self.repo.update(run_id, status="interrupted", ended_at=now(),
                             error="Aplicação encerrada; histórico parcial preservado.")
            return
        measured = model.evaluate(val, return_dict=True, verbose=0)
        metrics = classification_metrics(
            prepared.partitions["val"].target,
            probabilities(model, prepared.transformed["val"]), config.threshold,
        )
        metrics.update(loss=float(measured["loss"]),
                       predictive_loss=float(measured["predictive_loss"]))
        directory = self.repo.root / "artifacts" / run_id
        directory.mkdir(parents=True, exist_ok=True)
        model.save(directory / "model.pending.keras")
        (directory / "model.pending.keras").replace(directory / "model.keras")
        joblib.dump(prepared.pipeline, directory / "pipeline.pending.joblib")
        (directory / "pipeline.pending.joblib").replace(directory / "pipeline.joblib")
        hashes = {name: sha256((directory / name).read_bytes()).hexdigest()
                  for name in ("model.keras", "pipeline.joblib")}
        best_epoch = min(history, key=lambda epoch: epoch["val_loss"])["epoch"]
        self.repo.update(
            run_id, status="completed", ended_at=now(), duration_seconds=time.monotonic() - started,
            validation_metrics=metrics, best_epoch=best_epoch,
            saved_epoch=best_epoch if config.training.early_stopping else len(history),
            artifact_hashes=hashes,
        )

    def select(self, run_id):
        with self.repo.connect(write=True) as db:
            run = self.repo.get(run_id, db)
            study = self.repo.study(db)
            if study["status"] != "open":
                raise Conflict("A seleção está encerrada: o teste deste ciclo já foi aberto.")
            if run["status"] != "completed":
                raise Conflict("Selecione um experimento concluído, usando a validação.")
            study["selected_experiment_id"] = run_id
            self.repo.put_study(db, study)
        return study

    def evaluate(self, run_id):
        # Reserva persistida antes de ler o holdout. Falhas permitem retry só do mesmo modelo.
        with self.repo.connect(write=True) as db:
            run = self.repo.get(run_id, db)
            study = self.repo.study(db)
            if study["selected_experiment_id"] != run_id:
                raise Conflict("Selecione este experimento pela validação antes de abrir o teste.")
            if study["status"] == "completed":
                return study
            if study["status"] == "evaluating":
                raise Conflict("A avaliação final já está em execução.")
            if run["status"] != "completed":
                raise Conflict("O experimento precisa estar concluído.")
            if any(json.loads(row[0])["status"] in ("queued", "running")
                   for row in db.execute("SELECT document FROM runs")):
                raise Conflict("Aguarde todos os treinos antes de abrir o teste final.")
            study["status"] = "evaluating"
            self.repo.put_study(db, study)
        try:
            with self.operation_lock:
                if self.dataset.file_hash != run["dataset_sha256"] or sha256(
                    self.dataset_path_bytes()
                ).hexdigest() != run["dataset_sha256"]:
                    raise ValueError("O CSV mudou desde o treino. Restaure o arquivo original.")
                directory = self.repo.root / "artifacts" / run_id
                for name, digest in run["artifact_hashes"].items():
                    if sha256((directory / name).read_bytes()).hexdigest() != digest:
                        raise ValueError("Um artefato não corresponde ao experimento salvo.")
                model = tf.keras.models.load_model(directory / "model.keras")
                pipeline = joblib.load(directory / "pipeline.joblib")
                test = self.dataset.df.set_index("sample_id", drop=False).loc[
                    run["partition_ids"]["test"]
                ]
                X_test = pipeline.transform(test[FEATURES])
                result = classification_metrics(
                    test.target, probabilities(model, X_test), run["config"]["threshold"],
                )
                result.update(experiment_id=run_id, evaluated_at=now(), positive_class=1,
                              dataset_sha256=run["dataset_sha256"])
            with self.repo.connect(write=True) as db:
                stored = self.repo.get(run_id, db)
                stored["test_metrics"] = result
                self.repo.put(db, stored)
                study = self.repo.study(db)
                study.update(status="completed", result=result)
                self.repo.put_study(db, study)
            return study
        except Exception:
            with self.repo.connect(write=True) as db:
                study = self.repo.study(db)
                study["status"] = "evaluation_failed"
                self.repo.put_study(db, study)
            raise

    def dataset_path_bytes(self):
        return self.dataset.path.read_bytes()


def reproduction_code(run):
    config = ExperimentConfig.model_validate(run["config"])
    source_check = f'''from pathlib import Path
from hashlib import sha256
assert sha256(Path("data/base_classificacao_binaria.csv").read_bytes()).hexdigest() == \\
    "{run['dataset_sha256']}", "O CSV não corresponde à fonte deste experimento."
'''
    return source_check + python_code(config.preprocess) + f'''
from app.services.neural import ModelConfig, TrainingConfig, build_model, batches, probabilities
import tensorflow as tf

model_config = ModelConfig(**{config.model.model_dump()!r})
training = TrainingConfig(**{config.training.model_dump()!r})
model = build_model(X_train.shape[1], model_config, training.seed, threshold={config.threshold})
callbacks = []
if training.early_stopping:
    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=training.patience, restore_best_weights=True,
    ))
history = model.fit(
    batches(X_train, train.target, training.batch_size, training.seed, shuffle=True),
    validation_data=batches(X_val, val.target, training.batch_size, training.seed),
    epochs=training.epochs, callbacks=callbacks,
)
print(model.evaluate(batches(X_val, val.target, training.batch_size, training.seed),
                     return_dict=True))
# Não abrir test durante a reprodução ou seleção de hiperparâmetros.
# Teste final: apenas pela ação explícita de avaliação no laboratório.
# Fonte SHA-256: {run['dataset_sha256']}
# Seed não garante determinismo bit a bit entre hardwares e versões.
'''


def export_run(run, format_name):
    if format_name == "python":
        return reproduction_code(run), "text/x-python", "py"
    if format_name == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            "epoch", "loss", "val_loss", "accuracy", "val_accuracy",
            "predictive_loss", "val_predictive_loss",
        ])
        writer.writeheader()
        writer.writerows(run["history"])
        return output.getvalue(), "text/csv", "csv"
    if format_name == "markdown":
        content = f"# ANN Lab — {run['config']['name']}\n\n"
        content += f"Experimento: {run['id']}\n\nStatus: {run['status']}\n\n"
        content += (
            f"Épocas realizadas: {run['completed_epochs']} / "
            f"{run['config']['training']['epochs']}\n\n"
            f"Melhor época: {run['best_epoch']}; pesos salvos da época: {run['saved_epoch']}.\n\n"
            f"Partições: {encode(run['split_counts'])}\n\n"
            f"Arquitetura: {encode(run['model_summary'])}\n\n"
            f"Features: {', '.join(run['output_features'])}\n\n"
        )
        for title, value in [("Configuração", run["config"]), ("Versões", run["versions"]),
                             ("Validação do modelo salvo", run["validation_metrics"]),
                             ("Teste final", run["test_metrics"])]:
            formatted = json.dumps(value, indent=2, ensure_ascii=False)
            content += f"## {title}\n\n```json\n{formatted}\n```\n\n"
        content += (
            f"Fonte SHA-256: `{run['dataset_sha256']}`\n\n"
            "Classes e unidades não têm significado de domínio informado. Fit apenas no treino. "
            "Perda loss inclui L2; predictive_loss mede a entropia cruzada sem penalização. "
            "Métricas de teste null indicam teste ainda reservado. "
            "Reprodutibilidade aproximada entre versões e equipamentos.\n"
        )
        return content, "text/markdown", "md"
    return json.dumps(run, ensure_ascii=False, indent=2), "application/json", "json"
