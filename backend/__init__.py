from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "experiments" / "results"

app = FastAPI(
    title="Robot Navigation Simulation Platform",
    description="Backend API for RL + KAN robot navigation comparison"
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
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {file_path.name}"
        )

    df = pd.read_csv(file_path)
    return df.to_dict(orient="records")


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "project": "Robot Navigation Simulation Platform using RL and KAN"
    }


@app.get("/api/results/training/mlp")
def get_mlp_training_log():
    path = RESULTS_DIR / "custom_dqn_mlp_train_log.csv"
    return read_csv_as_records(path)


@app.get("/api/results/training/kan")
def get_kan_training_log():
    path = RESULTS_DIR / "custom_dqn_kan_train_log.csv"
    return read_csv_as_records(path)


@app.get("/api/results/training")
def get_training_logs():
    mlp_path = RESULTS_DIR / "custom_dqn_mlp_train_log.csv"
    kan_path = RESULTS_DIR / "custom_dqn_kan_train_log.csv"

    return {
        "mlp": read_csv_as_records(mlp_path),
        "kan": read_csv_as_records(kan_path)
    }


@app.get("/api/results/final")
def get_final_evaluation():
    path = RESULTS_DIR / "final_saved_model_evaluation.csv"
    return read_csv_as_records(path)


@app.get("/api/results/comparison-image")
def get_comparison_image():
    path = RESULTS_DIR / "mlp_vs_kan_comparison.png"

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Comparison image not found"
        )

    return FileResponse(path)


@app.get("/api/comparison/summary")
def get_comparison_summary():
    final_path = RESULTS_DIR / "final_saved_model_evaluation.csv"

    if final_path.exists():
        df = pd.read_csv(final_path)
        return df.to_dict(orient="records")

    # Fallback: use final row from training logs
    mlp_path = RESULTS_DIR / "custom_dqn_mlp_train_log.csv"
    kan_path = RESULTS_DIR / "custom_dqn_kan_train_log.csv"

    if not mlp_path.exists() or not kan_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No result files found"
        )

    mlp_df = pd.read_csv(mlp_path)
    kan_df = pd.read_csv(kan_path)

    return {
        "mlp_final": mlp_df.iloc[-1].to_dict(),
        "kan_final": kan_df.iloc[-1].to_dict()
    }