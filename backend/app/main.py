import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.experiments import router as experiment_router
from app.services.dataset import DEFAULT_CSV, ROOT, DatasetService
from app.services.experiments import Conflict, ExperimentService
from app.services.preprocessing import PreprocessingConfig, PreprocessingService

logger = logging.getLogger(__name__)


def create_app(
    csv_path: Path = DEFAULT_CSV, storage_path: Path | None = None, enable_training: bool = False,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.experiments = None
        try:
            app.state.dataset = DatasetService(csv_path)
            app.state.preprocessing = PreprocessingService(app.state.dataset)
            if enable_training:
                app.state.experiments = ExperimentService(
                    app.state.dataset,
                    storage_path or Path(os.getenv("ANN_LAB_STORAGE", str(ROOT / "local_storage"))),
                )
        except (OSError, ValueError):
            logger.exception("Não foi possível carregar a base original")
            app.state.dataset = None
        yield
        if app.state.experiments:
            app.state.experiments.close()

    app = FastAPI(title="ANN Lab", version="1.0.0", lifespan=lifespan)
    app.include_router(experiment_router)
    origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    app.add_middleware(
        CORSMiddleware, allow_origins=[value.strip() for value in origins.split(",")],
        allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
    )

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request: Request, _error: RequestValidationError):
        if request.url.path == "/api/preprocessing/preview":
            return JSONResponse(status_code=422, content={
                "detail": "Configuração inválida. Confira estratégia, escala, divisão, seed "
                          "(inteiro de 0 a 4294967295), vizinhos (3/5/7), coluna e subconjunto.",
            })
        paths = ("experiments", "model", "study", "evaluation")
        if any(part in request.url.path for part in paths):
            return JSONResponse(status_code=422, content={
                "detail": "Configuração inválida. Confira camadas (1–4), neurônios (4–256), "
                          "épocas (1–200), batch (8–256), LR (0–0,1) e patience menor que épocas.",
                "fields": [".".join(str(part) for part in item["loc"])
                           for item in _error.errors()],
            })
        return JSONResponse(status_code=422, content={
            "detail": "Parâmetros inválidos. Use offset ≥ 0, limit de 1 a 100 e bins 10, 20 ou 30.",
        })

    @app.exception_handler(Conflict)
    async def conflict(_request: Request, error: Conflict):
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.exception_handler(KeyError)
    async def missing(_request: Request, _error: KeyError):
        return JSONResponse(status_code=404, content={"detail": "Experimento não encontrado."})

    @app.exception_handler(ValueError)
    async def invalid_operation(_request: Request, error: ValueError):
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.exception_handler(Exception)
    async def unexpected(_request: Request, _error: Exception):
        logger.exception("Falha operacional", exc_info=_error)
        return JSONResponse(status_code=500, content={
            "detail": "Não foi possível concluir a operação. Confira o log e tente novamente.",
        })

    def dataset(request: Request) -> DatasetService:
        service = request.app.state.dataset
        if service is None:
            raise HTTPException(
                503, "A base não pôde ser carregada. Confira o CSV e reinicie a API."
            )
        return service

    service_dep = Annotated[DatasetService, Depends(dataset)]

    @app.get("/api/health")
    def health(request: Request):
        loaded = request.app.state.dataset is not None
        return JSONResponse(status_code=200 if loaded else 503, content={
            "status": "ok" if loaded else "error", "dataset_loaded": loaded,
            "phase": 5,
            "ml_engine": "tensorflow" if request.app.state.experiments else "not_initialized",
            "training_available": request.app.state.experiments is not None,
        })

    @app.get("/api/dataset/profile")
    def profile(service: service_dep):
        return service.profile()

    @app.get("/api/dataset/preview")
    def preview(
        service: service_dep,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=100)] = 30,
    ):
        return service.preview(offset, limit)

    @app.get("/api/dataset/eda")
    def eda(
        service: service_dep,
        column: str = "sensor_a",
        bins: Annotated[int, Query(ge=10, le=30)] = 20,
    ):
        if bins not in (10, 20, 30):
            raise HTTPException(422, "Escolha 10, 20 ou 30 intervalos (bins).")
        try:
            return service.eda(column, bins)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/api/dataset/correlations")
    def correlations(service: service_dep):
        return service.correlations()

    @app.post("/api/preprocessing/preview")
    def preprocessing_preview(config: PreprocessingConfig, request: Request, service: service_dep):
        try:
            return request.app.state.preprocessing.preview(config)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except Exception as error:
            logger.exception("Falha na preparação dos dados")
            raise HTTPException(
                500, "Não foi possível preparar os dados. Tente novamente."
            ) from error

    return app


app = create_app(enable_training=True)
