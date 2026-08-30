from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.ai_routes import router as ai_router
from app.api.routes.health_routes import router as health_router
from app.api.routes.webhook_routes import router as webhook_router
from app.core.config import get_settings
from app.db.mongo import close_mongo_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_mongo_client()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(ai_router, prefix=settings.api_v1_prefix)
    app.include_router(webhook_router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
