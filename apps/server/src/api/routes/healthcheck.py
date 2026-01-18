from fastapi import APIRouter

from src.schemas import HealthcheckResponse

router = APIRouter()


@router.get("/")
async def healthcheck() -> HealthcheckResponse:
    """Healthcheck endpoint to verify that the server is running."""
    # TODO: Implement logic to check connection to message broker service.
    return HealthcheckResponse(
        status="ok",
        message="Server is running",
        extra={"Broker connection": "ok"},
    )
