# TODO: Use .env for configuration management

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.main import router as api_router
from src.core.config import settings

app = FastAPI(
    title="Server Application",
    openapi_url=f"{settings.SERVER_PREFIX}/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.SERVER_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.SERVER_PREFIX)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.SERVER_PORT, log_level="info")
