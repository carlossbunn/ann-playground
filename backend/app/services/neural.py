"""Motor Keras em CPU e contratos de experimentos."""

import os
from typing import Annotated, Literal

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
import tensorflow as tf
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.metrics import confusion_matrix, log_loss

from app.services.preprocessing import PreparationSettings


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ModelConfig(StrictModel):
    hidden_layers: list[Annotated[int, Field(ge=4, le=256, strict=True)]] = Field(
        default_factory=lambda: [32, 16], min_length=1, max_length=4,
    )
    activation: Literal["relu", "tanh"] = "relu"
    optimizer: Literal["adam", "sgd", "rmsprop"] = "adam"
    learning_rate: float = Field(default=0.001, gt=0, le=0.1)
    l2: float = Field(default=0, ge=0, le=0.1)


class TrainingConfig(StrictModel):
    epochs: int = Field(default=50, ge=1, le=200, strict=True)
    batch_size: int = Field(default=32, ge=8, le=256, strict=True)
    early_stopping: bool = True
    patience: int = Field(default=8, ge=1, le=199, strict=True)
    seed: int = Field(default=42, ge=0, le=2**31 - 1, strict=True)

    @model_validator(mode="after")
    def valid_patience(self):
        if self.early_stopping and self.patience >= self.epochs:
            raise ValueError("Com early stopping, patience deve ser menor que épocas.")
        return self


class ExperimentConfig(StrictModel):
    name: str = Field(default="Meu experimento", min_length=1, max_length=100)
    preprocess: PreparationSettings = Field(default_factory=PreparationSettings)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    threshold: float = Field(default=0.5, gt=0, lt=1)


class FinalRequest(StrictModel):
    experiment_id: str
    confirm: Literal[True]


class SelectRequest(StrictModel):
    experiment_id: str


def describe_model(inputs: int, config: ModelConfig) -> dict:
    layers = []
    previous = inputs
    for index, units in enumerate([*config.hidden_layers, 1]):
        count = (previous + 1) * units
        layers.append({
            "name": f"dense_{index + 1}", "inputs": previous, "units": units,
            "activation": config.activation if index < len(config.hidden_layers) else "sigmoid",
            "parameters": count,
        })
        previous = units
    return {"input_features": inputs, "layers": layers,
            "total_parameters": sum(layer["parameters"] for layer in layers)}


def build_model(inputs: int, config: ModelConfig, seed: int, threshold: float = 0.5):
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([tf.keras.Input(shape=(inputs,))])
    for units in config.hidden_layers:
        model.add(tf.keras.layers.Dense(
            units, activation=config.activation,
            kernel_regularizer=tf.keras.regularizers.l2(config.l2),
        ))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    optimizers = {"adam": tf.keras.optimizers.Adam, "sgd": tf.keras.optimizers.SGD,
                  "rmsprop": tf.keras.optimizers.RMSprop}
    model.compile(
        optimizer=optimizers[config.optimizer](learning_rate=config.learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=threshold),
                 tf.keras.metrics.BinaryCrossentropy(name="predictive_loss")],
    )
    return model


def batches(X, y, batch_size: int, seed: int, shuffle: bool = False):
    dataset = tf.data.Dataset.from_tensor_slices((
        np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32).reshape(-1, 1),
    ))
    if shuffle:
        dataset = dataset.shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return dataset.batch(batch_size).with_options(options)


def probabilities(model, X):
    result = np.concatenate([
        model(np.asarray(X[start:start + 512], dtype=np.float32), training=False).numpy().ravel()
        for start in range(0, len(X), 512)
    ])
    if not np.isfinite(result).all():
        raise ValueError("A rede produziu probabilidades indefinidas.")
    return result


def classification_metrics(y, probability, threshold: float) -> dict:
    predicted = np.asarray(probability) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel().tolist()

    def divide(numerator, denominator):
        return numerator / denominator if denominator else 0.0

    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    return {
        "confusion_matrix": [[tn, fp], [fn, tp]], "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": divide(tp + tn, tn + fp + fn + tp), "precision": precision,
        "recall": recall, "specificity": divide(tn, tn + fp),
        "f1": divide(2 * precision * recall, precision + recall),
        "predictive_loss": float(log_loss(y, np.asarray(probability, dtype=float), labels=[0, 1])),
        "zero_division": 0, "threshold": threshold, "samples": len(y),
        "undefined_metrics": [name for name, denominator in [
            ("precision", tp + fp), ("recall", tp + fn), ("specificity", tn + fp),
            ("f1", 2 * tp + fp + fn),
        ] if denominator == 0],
    }
