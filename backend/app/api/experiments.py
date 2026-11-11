from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

from app.services.experiments import ExperimentService, export_run, reproduction_code
from app.services.neural import ExperimentConfig, FinalRequest, SelectRequest, describe_model

router = APIRouter(prefix="/api")


def service(request: Request) -> ExperimentService:
    instance = request.app.state.experiments
    if instance is None:
        raise HTTPException(
            503, "O motor de treinamento não está disponível. Confira a instalação."
        )
    return instance


Service = Annotated[ExperimentService, Depends(service)]


def public(run):
    return {key: value for key, value in run.items() if key != "partition_ids"}


@router.post("/model/preview")
def model_preview(config: ExperimentConfig, instance: Service):
    prepared = instance.preparation.prepare(config.preprocess)
    return {
        **describe_model(prepared.transformed["train"].shape[1], config.model),
        "output_features": prepared.pipeline.get_feature_names_out().tolist(),
        "python_code": reproduction_code({
            "config": config.model_dump(), "dataset_sha256": instance.dataset.file_hash,
        }),
    }


@router.post("/experiments", status_code=202)
def create_experiment(config: ExperimentConfig, instance: Service):
    return public(instance.create(config))


@router.get("/experiments")
def list_experiments(instance: Service):
    return [public(run) for run in instance.repo.list()]


@router.get("/experiments/{run_id}")
def get_experiment(run_id: str, instance: Service):
    return public(instance.repo.get(run_id))


@router.get("/experiments/{run_id}/export")
def export_experiment(
    run_id: str, instance: Service,
    format: Literal["json", "markdown", "html", "csv", "python"] = "json",
):
    run = instance.repo.get(run_id)
    content, media_type, extension = export_run(run, format)
    return Response(content, media_type=media_type, headers={
        "Content-Disposition": f'attachment; filename="ann-lab-{run["id"]}.{extension}"',
    })


@router.get("/study")
def study(instance: Service):
    return instance.repo.study()


@router.post("/study/select")
def select_experiment(config: SelectRequest, instance: Service):
    return instance.select(config.experiment_id)


@router.get("/evaluation/final")
def final_status(instance: Service):
    return instance.repo.study()


@router.post("/evaluation/final")
def evaluate(config: FinalRequest, instance: Service):
    return instance.evaluate(config.experiment_id)
