from fastapi import APIRouter

from src.api.routes.healthcheck import router as healthcheck_router
from src.api.routes.websocket import router as websocket_router

router = APIRouter()

router.include_router(healthcheck_router, prefix="/healthcheck", tags=["Healthcheck"])
router.include_router(websocket_router, prefix="/ws", tags=["WebSocket"])
