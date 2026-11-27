from fastapi import APIRouter

from database.metrics import get_metrics

router = APIRouter()
router.add_api_route("/metrics", get_metrics, methods=["GET"])
