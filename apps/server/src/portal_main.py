"""React portal entrypoint, independent of the legacy dlib client."""

from contextlib import asynccontextmanager
from pathlib import Path
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / '.env')
load_dotenv(Path(__file__).resolve().parents[1] / '.env')
os.environ.setdefault('CAMS_MODELS_DIR', str(PROJECT_ROOT / '.runtime' / 'models'))


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.portal import initialize
    portal = initialize()
    portal.bridge.start()
    try:
        yield
    finally:
        portal.bridge.close()
        portal.faces.close()
        for session in portal.monitor.sessions.values():
            if session.tracker is not None and hasattr(session.tracker, 'close'):
                session.tracker.close()


app = FastAPI(title='CamS · Panel de acceso', lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
    allow_credentials=False,
    allow_methods=['GET', 'POST'],
    allow_headers=['Authorization', 'Content-Type'],
)

from src.portal import router
app.include_router(router, prefix='/api')


@app.get('/', include_in_schema=False)
def home():
    return RedirectResponse('http://localhost:5173')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=int(os.getenv('SERVER_PORT', '8765')))
