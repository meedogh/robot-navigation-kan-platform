import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.live_sim import LiveSimulator
from backend.explain import build_kan_explanation


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "experiments" / "results"
CHECKPOINT_DIR = BASE_DIR / "experiments" / "checkpoints"

app = FastAPI(
    title="Robot Navigation Simulation Platform",
    description="RL + KAN robot navigation comparison dashboard backend",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_csv_as_records(file_path: Path):
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path.name}")
    df = pd.read_csv(file_path)
    return df.to_dict(orient="records")


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "project": "Robot Navigation Simulation Platform using RL and KAN",
    }


@app.get("/api/results/training/mlp")
def get_mlp_training_log():
    return read_csv_as_records(RESULTS_DIR / "custom_dqn_mlp_train_log.csv")


@app.get("/api/results/training/kan")
def get_kan_training_log():
    return read_csv_as_records(RESULTS_DIR / "custom_dqn_kan_train_log.csv")


@app.get("/api/results/training")
def get_training_logs():
    return {
        "mlp": read_csv_as_records(RESULTS_DIR / "custom_dqn_mlp_train_log.csv"),
        "kan": read_csv_as_records(RESULTS_DIR / "custom_dqn_kan_train_log.csv"),
    }


@app.get("/api/results/final")
def get_final_evaluation():
    return read_csv_as_records(RESULTS_DIR / "final_saved_model_evaluation.csv")


@app.get("/api/results/comparison-image")
def get_comparison_image():
    path = RESULTS_DIR / "mlp_vs_kan_comparison.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Comparison image not found")
    return FileResponse(path)


@app.get("/api/comparison/summary")
def get_comparison_summary():
    final_path = RESULTS_DIR / "final_saved_model_evaluation.csv"
    if final_path.exists():
        return pd.read_csv(final_path).to_dict(orient="records")

    mlp_path = RESULTS_DIR / "custom_dqn_mlp_train_log.csv"
    kan_path = RESULTS_DIR / "custom_dqn_kan_train_log.csv"
    if not mlp_path.exists() or not kan_path.exists():
        raise HTTPException(status_code=404, detail="No result files found")

    return {
        "mlp_final": pd.read_csv(mlp_path).iloc[-1].to_dict(),
        "kan_final": pd.read_csv(kan_path).iloc[-1].to_dict(),
    }


@app.get("/api/explain/kan")
def explain_kan():
    """Return KAN explainability data: feature importance + learned basis functions."""
    model_path = CHECKPOINT_DIR / "custom_dqn_kan_best.pt"
    if not model_path.exists():
        model_path = CHECKPOINT_DIR / "custom_dqn_kan.pt"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="No KAN checkpoint found")
    return build_kan_explanation(model_path)


@app.websocket("/ws/live")
async def live_simulation(websocket: WebSocket):
    """Stream a live robot navigation episode. Client sends {'model': 'mlp'|'kan'}."""
    await websocket.accept()

    try:
        first_message = await websocket.receive_json()
        model_type = first_message.get("model", "kan")
    except Exception:
        model_type = "kan"

    try:
        sim = LiveSimulator(model_type=model_type)
    except Exception as exc:
        await websocket.send_json({"error": str(exc)})
        await websocket.close()
        return

    try:
        while True:
            frame = sim.step()
            await websocket.send_json(frame)
            if frame.get("done"):
                sim.reset()
            await asyncio.sleep(0.05)  # ~20 fps
    except WebSocketDisconnect:
        pass
    except Exception:
        await websocket.close()