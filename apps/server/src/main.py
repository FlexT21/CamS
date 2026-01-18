# TODO: Use .env for configuration management

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.main import router as api_router
from src.core.constants import API_PREFIX

app = FastAPI(
    title="Server Application",
    openapi_url=f"{API_PREFIX}/openapi.json",
)

# TODO: Refine CORS settings for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=API_PREFIX)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
