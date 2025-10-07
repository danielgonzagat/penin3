"""
REST API for Fibonacci Engine using FastAPI

Provides HTTP endpoints for controlling and monitoring the engine.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uvicorn
from pathlib import Path

from fibonacci_engine.core.motor_fibonacci import FibonacciEngine, FibonacciConfig
from fibonacci_engine.adapters.rl_synthetic import RLSyntheticAdapter
from fibonacci_engine.adapters.supervised_synthetic import SupervisedSyntheticAdapter
from fibonacci_engine.adapters.tool_pipeline import ToolPipelineAdapter


# API Models
class StartEngineRequest(BaseModel):
    """Request to start the engine."""
    config: Optional[Dict[str, Any]] = None
    adapter: str = Field(default="rl", pattern="^(rl|supervised|tool)$")


class RunEngineRequest(BaseModel):
    """Request to run the engine."""
    generations: Optional[int] = None


class StepEngineRequest(BaseModel):
    """Request to step the engine."""
    n: int = Field(default=1, ge=1)


class SnapshotRequest(BaseModel):
    """Request to save a snapshot."""
    filepath: str = Field(default="fibonacci_engine/persistence/snapshot.json")


class LoadSnapshotRequest(BaseModel):
    """Request to load a snapshot."""
    filepath: str


# Global engine instance
_engine: Optional[FibonacciEngine] = None
_engine_running: bool = False


# Create FastAPI app
app = FastAPI(
    title="Fibonacci Engine API",
    description="REST API for the Universal AI Optimization Engine",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Fibonacci Engine API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "engine_initialized": _engine is not None,
        "engine_running": _engine_running,
    }


@app.post("/engine/start")
async def start_engine(request: StartEngineRequest):
    """
    Initialize and start the engine.
    
    Args:
        request: Configuration and adapter selection.
        
    Returns:
        Status message and engine info.
    """
    global _engine
    
    if _engine is not None:
        raise HTTPException(status_code=400, detail="Engine already initialized")
    
    # Parse config
    if request.config:
        config = FibonacciConfig.from_dict(request.config)
    else:
        config = FibonacciConfig()
    
    # Select adapter
    if request.adapter == 'rl':
        adapter_obj = RLSyntheticAdapter()
    elif request.adapter == 'supervised':
        adapter_obj = SupervisedSyntheticAdapter()
    else:  # tool
        adapter_obj = ToolPipelineAdapter()
    
    # Create engine
    _engine = FibonacciEngine(
        config=config,
        evaluate_fn=adapter_obj.evaluate,
        descriptor_fn=adapter_obj.descriptor,
        mutate_fn=adapter_obj.mutate,
        cross_fn=adapter_obj.crossover,
        task_sampler=adapter_obj.task_sampler,
    )
    
    return {
        "status": "success",
        "message": "Engine initialized",
        "config": config.to_dict(),
        "adapter": request.adapter,
    }


@app.post("/engine/step")
async def step_engine(request: StepEngineRequest):
    """
    Execute N generation steps.
    
    Args:
        request: Number of steps to execute.
        
    Returns:
        Step statistics.
    """
    global _engine
    
    if _engine is None:
        raise HTTPException(status_code=400, detail="Engine not initialized")
    
    results = []
    for i in range(request.n):
        result = _engine.step()
        results.append(result)
    
    return {
        "status": "success",
        "steps_executed": request.n,
        "results": results,
    }


@app.post("/engine/run")
async def run_engine(request: RunEngineRequest, background_tasks: BackgroundTasks):
    """
    Run the engine for multiple generations.
    
    Args:
        request: Number of generations to run.
        background_tasks: FastAPI background tasks.
        
    Returns:
        Run statistics.
    """
    global _engine, _engine_running
    
    if _engine is None:
        raise HTTPException(status_code=400, detail="Engine not initialized")
    
    if _engine_running:
        raise HTTPException(status_code=400, detail="Engine already running")
    
    # Run synchronously for now (could be made async)
    _engine_running = True
    try:
        result = _engine.run(generations=request.generations)
        return {
            "status": "success",
            "message": "Run complete",
            "result": result,
        }
    finally:
        _engine_running = False


@app.post("/engine/stop")
async def stop_engine():
    """
    Stop the engine gracefully.
    
    Returns:
        Status message.
    """
    global _engine, _engine_running
    
    if _engine is None:
        raise HTTPException(status_code=400, detail="Engine not initialized")
    
    _engine.stop()
    _engine_running = False
    
    return {
        "status": "success",
        "message": "Engine stopped",
    }


@app.get("/engine/status")
async def get_status():
    """
    Get current engine status.
    
    Returns:
        Comprehensive status information.
    """
    global _engine
    
    if _engine is None:
        raise HTTPException(status_code=400, detail="Engine not initialized")
    
    status = _engine.get_status()
    return {
        "status": "success",
        "engine_status": status,
    }


@app.get("/elites")
async def get_elites():
    """
    Get all elite candidates from the archive.
    
    Returns:
        List of elite candidates.
    """
    global _engine
    
    if _engine is None:
        raise HTTPException(status_code=400, detail="Engine not initialized")
    
    elites = _engine.archive.get_all_elites()
    
    return {
        "status": "success",
        "n_elites": len(elites),
        "elites": [
            {
                "fitness": e.fitness,
                "descriptor": e.descriptor,
                "generation": e.generation,
                "metadata": e.metadata,
            }
            for e in elites
        ],
    }


@app.get("/elites/best")
async def get_best_elite():
    """
    Get the best elite candidate.
    
    Returns:
        Best elite candidate.
    """
    global _engine
    
    if _engine is None:
        raise HTTPException(status_code=400, detail="Engine not initialized")
    
    best = _engine.archive.get_best()
    
    if best is None:
        raise HTTPException(status_code=404, detail="No elites found")
    
    return {
        "status": "success",
        "best": {
            "fitness": best.fitness,
            "descriptor": best.descriptor,
            "generation": best.generation,
            "metadata": best.metadata,
        },
    }


@app.get("/ledger")
async def get_ledger():
    """
    Get ledger statistics and recent entries.
    
    Returns:
        Ledger information.
    """
    global _engine
    
    if _engine is None:
        raise HTTPException(status_code=400, detail="Engine not initialized")
    
    stats = _engine.ledger.get_statistics()
    recent_entries = _engine.ledger.get_entries(start_index=max(0, len(_engine.ledger) - 10))
    
    return {
        "status": "success",
        "statistics": stats,
        "recent_entries": [e.to_dict() for e in recent_entries],
    }


@app.get("/ledger/verify")
async def verify_ledger():
    """
    Verify ledger integrity.
    
    Returns:
        Verification result.
    """
    global _engine
    
    if _engine is None:
        raise HTTPException(status_code=400, detail="Engine not initialized")
    
    is_valid = _engine.ledger.verify()
    
    return {
        "status": "success",
        "is_valid": is_valid,
        "message": "Ledger is valid" if is_valid else "Ledger integrity check failed!",
    }


@app.post("/snapshot")
async def save_snapshot(request: SnapshotRequest):
    """
    Save engine snapshot.
    
    Args:
        request: Filepath for snapshot.
        
    Returns:
        Status message.
    """
    global _engine
    
    if _engine is None:
        raise HTTPException(status_code=400, detail="Engine not initialized")
    
    _engine.snapshot(request.filepath)
    
    return {
        "status": "success",
        "message": "Snapshot saved",
        "filepath": request.filepath,
    }


@app.get("/meta-controller")
async def get_meta_controller():
    """
    Get meta-controller statistics.
    
    Returns:
        Meta-controller information.
    """
    global _engine
    
    if _engine is None:
        raise HTTPException(status_code=400, detail="Engine not initialized")
    
    stats = _engine.meta_controller.get_all_statistics()
    recommendation = _engine.meta_controller.get_recommendation()
    
    return {
        "status": "success",
        "statistics": stats,
        "recommendation": recommendation,
    }


@app.get("/curriculum")
async def get_curriculum():
    """
    Get curriculum statistics.
    
    Returns:
        Curriculum information.
    """
    global _engine
    
    if _engine is None:
        raise HTTPException(status_code=400, detail="Engine not initialized")
    
    stats = _engine.curriculum.get_statistics()
    
    return {
        "status": "success",
        "curriculum": stats,
    }


@app.delete("/engine")
async def reset_engine():
    """
    Reset/delete the engine instance.
    
    Returns:
        Status message.
    """
    global _engine, _engine_running
    
    if _engine_running:
        _engine.stop()
    
    _engine = None
    _engine_running = False
    
    return {
        "status": "success",
        "message": "Engine reset",
    }


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": str(exc),
        },
    )


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Start the FastAPI server.
    
    Args:
        host: Host address.
        port: Port number.
    """
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
