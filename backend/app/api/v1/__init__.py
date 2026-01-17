"""API v1 routes."""

from fastapi import APIRouter

from app.api.v1 import projects, logs, analytics, evaluations, recommendations, scheduler

api_router = APIRouter()

api_router.include_router(projects.router, prefix="/projects", tags=["projects"])
api_router.include_router(logs.router, prefix="/logs", tags=["logs"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
api_router.include_router(evaluations.router, prefix="/evaluations", tags=["evaluations"])
api_router.include_router(recommendations.router, prefix="/recommendations", tags=["recommendations"])
api_router.include_router(scheduler.router, prefix="/scheduler", tags=["scheduler"])
