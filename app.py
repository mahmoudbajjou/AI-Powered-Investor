from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config.settings import ROOT
from database.create_table import create_tables
from database.metrics import get_metrics
from routes.chat import router as chat_router
from routes.dashboard import router as dashboard_router
from routes.health import router as health_router
from routes.ingestion import router as ingestion_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    yield


app = FastAPI(
    title="AI-Powered Investor Intelligence Platform",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)
app.include_router(ingestion_router, prefix="/api", tags=["Ingestion"])
app.include_router(chat_router, prefix="/api", tags=["Chat"])
app.include_router(dashboard_router, prefix="/api", tags=["Dashboard"])
app.include_router(health_router, tags=["Health"])
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

templates = Jinja2Templates(directory=ROOT / "templates")


@app.get("/")
def dashboard(request: Request):
    metrics = get_metrics()
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "metrics": metrics,
            "total_companies": len({row["company"] for row in metrics}),
            "total_reports": len(metrics),
        },
    )


if __name__ == "__main__":
    import uvicorn

    host = "127.0.0.1"
    port = 8000
    print(f"Ouvre le tableau de bord dans ton navigateur\nhttp://{host}:{port}/", flush=True)
    uvicorn.run(app, host=host, port=port)
