#!/usr/bin/env python3
"""Gera figuras reais do CSV e, opcionalmente, de um experimento exportado."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/base_classificacao_binaria.csv"


def save(fig, directory, name):
    fig.tight_layout()
    fig.savefig(directory / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(directory / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimento", type=Path, help="JSON exportado pelo ANN Lab")
    parser.add_argument("--saida", type=Path, default=ROOT / "local_storage/graficos")
    args = parser.parse_args()
    run = None
    if args.experimento:
        run = json.loads(args.experimento.read_text(encoding="utf-8"))
        if run.get("dataset_sha256") != sha256(CSV.read_bytes()).hexdigest():
            raise SystemExit("O CSV não corresponde à fonte do experimento exportado.")
    df = pd.read_csv(CSV)
    numeric = df.drop(columns=["sample_id", "target"]).select_dtypes(include="number")
    args.saida.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    for ax, column in zip(axes.flat, numeric.columns, strict=True):
        ax.hist(numeric[column].dropna(), bins=20, color="#315ddb", edgecolor="white")
        ax.set_title(column)
        ax.set_ylabel("Frequência")
    fig.suptitle("Distribuições dos atributos — CSV completo (EDA descritiva)")
    save(fig, args.saida, "histogramas.png")

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    for ax, column in zip(axes.flat, numeric.columns, strict=True):
        ax.boxplot(numeric[column].dropna(), vert=True)
        ax.set_title(column)
    fig.suptitle("Boxplots — CSV completo (pontos extremos não são necessariamente erros)")
    save(fig, args.saida, "boxplots.png")

    fig, ax = plt.subplots(figsize=(10, 8))
    matrix = numeric.corr(method="pearson")
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrix)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix)), matrix.columns)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="Correlação de Pearson")
    ax.set_title("Correlação — CSV completo, pares disponíveis")
    save(fig, args.saida, "correlacao.png")

    if run and run.get("history"):
        history = pd.DataFrame(run["history"])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, metric, label in zip(
            axes, ["loss", "accuracy"], ["Perda", "Acurácia"], strict=True,
        ):
            ax.plot(history.epoch, history[metric], label="Treino")
            ax.plot(history.epoch, history[f"val_{metric}"], "--", label="Validação")
            ax.set(xlabel="Época", ylabel=label, title=label)
            ax.legend()
            ax.grid(alpha=0.25)
        fig.suptitle(f"Curvas reais — {run['config']['name']}")
        save(fig, args.saida, "curvas_aprendizado.png")

    if run and run.get("test_metrics"):
        matrix = run["test_metrics"]["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(image, ax=ax, label="Amostras")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i][j]), ha="center", va="center",
                        color="white" if matrix[i][j] > max(map(max, matrix)) / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Classe prevista", ylabel="Classe real",
               title="Matriz de confusão — teste final")
        save(fig, args.saida, "matriz_confusao.png")
    elif run:
        print("Teste final ainda não disponível neste JSON; matriz não gerada.")

    if run and run.get("validation_metrics"):
        metrics = ["accuracy", "precision", "recall", "specificity", "f1"]
        labels = ["Acurácia", "Precisão", "Recall", "Especificidade", "F1"]
        validation = run["validation_metrics"]
        test = run.get("test_metrics")
        positions = range(len(metrics))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar([i - 0.2 for i in positions], [validation[k] for k in metrics],
               width=0.4, label="Validação")
        if test:
            ax.bar([i + 0.2 for i in positions], [test[k] for k in metrics],
                   width=0.4, label="Teste final")
        ax.set_xticks(list(positions), labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Proporção")
        ax.set_title("Métricas do modelo salvo")
        ax.legend()
        save(fig, args.saida, "metricas.png")


if __name__ == "__main__":
    main()
