from __future__ import annotations

from typing import Any, List, Dict, Optional
from pydantic import BaseModel, Field


class CheckpointChampion(BaseModel):
    genome: Optional[Dict[str, float]] = None
    score: Optional[float] = None
    behavior: Optional[List[float]] = None
    metrics: Optional[Dict[str, float]] = None


class CheckpointArchiveEntry(BaseModel):
    idx: int = Field(..., ge=0)
    best_score: float
    behavior: List[float]
    genome: Optional[Dict[str, float]] = None


class CheckpointPayload(BaseModel):
    format_version: int = Field(default=1, ge=1)
    cycle: int = Field(..., ge=1)
    rng_state: Any
    population: List[Dict[str, float]]
    archive: List[CheckpointArchiveEntry]
    novelty_size: int = Field(..., ge=0)
    champion: CheckpointChampion
