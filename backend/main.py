import asyncio
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.live_sim import LiveSimulator
from backend.explain import build_kan_explanation
from backend import training
from rl import config_io
from simulation import env_factory


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


@app.get("/api/training/config")
def get_training_config():
    """Default tuning knobs plus preset runs offered by the Setup page."""
    return {
        "defaults": training.DEFAULT_TRAINING_CONFIG,
        "presets": training.presets(),
    }


@app.get("/api/training/last-config/{model_type}")
def get_last_training_config(model_type: str):
    """The config used for the most recent training run of this model."""
    config = training.last_run_config(model_type)
    if config is None:
        raise HTTPException(
            status_code=404,
            detail="No previous run config for this model"
        )
    return config


@app.post("/api/training/start")
def start_training_job(payload: dict):
    """Start a background training job with the given hyperparameters."""
    try:
        return training.start_training(payload)
    except training.JobError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


def _run_config_response(run: dict, filename: str) -> Response:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename).strip("_") or "run"
    return Response(
        content=json.dumps(run, indent=2) + "\n",
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}.json"'
        },
    )


@app.post("/api/config/export")
def export_run_config(payload: dict):
    """Turn a flat training config (e.g. from the Setup page form) into a
    portable run-config JSON file describing training params + environment."""
    config = payload.get("config") or {}
    if not isinstance(config, dict):
        raise HTTPException(status_code=400, detail="config must be a JSON object")

    name = str(payload.get("name") or "robotnav-run")
    description = str(payload.get("description") or "")

    try:
        run = config_io.run_config_from_flat(config, name=name, description=description)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid config: {exc}")

    return _run_config_response(run, name)


@app.post("/api/config/import")
def import_run_config(payload: dict):
    """Validate an uploaded run config JSON and return the flattened training
    config (ready for the Setup page form or /api/training/start)."""
    try:
        flat = config_io.flat_config_from_run(payload)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    warnings = []
    environment = payload.get("environment") or {}
    if (environment.get("source") or "builtin") == "module":
        module_path = environment.get("module")
        try:
            env_factory.resolve_env_class("module", None, module_path)
        except Exception as exc:
            warnings.append(
                f"The external environment '{module_path}' could not be loaded "
                f"on this machine: {exc}"
            )
        else:
            warnings.append(
                f"External environment '{module_path}' will be used for "
                "training, evaluation and live simulation on this backend."
            )

    return {
        "ok": True,
        "name": payload.get("name") or "unnamed-run",
        "config": flat,
        "warnings": warnings,
    }


@app.get("/api/config/export/{model_type}")
def export_last_run_config(model_type: str):
    """Download the config of the most recent run of a model as a portable
    run-config JSON file."""
    flat = training.last_run_config(model_type)
    if flat is None:
        raise HTTPException(
            status_code=404,
            detail=f"No previous run config for this model ({model_type})",
        )
    try:
        run = config_io.run_config_from_flat(
            flat,
            name=f"last-run-{model_type}",
            description="Config of the most recent training run, exported from the backend",
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid saved config: {exc}")

    return _run_config_response(run, f"robotnav-last-run-{model_type}")


@app.get("/api/model/export/{model_type}")
def export_model_onnx(model_type: str):
    """Download the best checkpoint of a model as an ONNX file.

    The exported graph (input 'observations' -> output 'q_values') can be run
    inside external simulators - e.g. Unity with Microsoft Sentis - so a model
    trained here can drive a simulation built there ("train here, run it
    there").  Works for the MLP and the KAN networks alike.
    """
    if model_type not in ("mlp", "kan"):
        raise HTTPException(
            status_code=400,
            detail=f"model_type must be 'mlp' or 'kan', got {model_type!r}",
        )

    from rl.model_export import (
        ONNX_INPUT_NAME,
        ONNX_OUTPUT_NAME,
        export_checkpoint_to_onnx,
        resolve_checkpoint_path,
    )

    try:
        resolve_checkpoint_path(model_type, CHECKPOINT_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    try:
        out_path = export_checkpoint_to_onnx(model_type, checkpoint_dir=CHECKPOINT_DIR)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=f"ONNX export failed: {exc}")

    data = out_path.read_bytes()
    filename = f"custom_dqn_{model_type}_best.onnx"
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-ONNX-Input": ONNX_INPUT_NAME,
            "X-ONNX-Output": ONNX_OUTPUT_NAME,
        },
    )


@app.get("/api/training/status")
def get_training_status():
    return training.status()


@app.get("/api/training/progress")
def get_training_progress():
    data = training.progress()
    if data is None:
        return {"rows": [], "state": None, "message": "No training job."}
    return data


@app.post("/api/training/stop")
def stop_training_job():
    return training.stop()


@app.post("/api/evaluate/start")
def start_evaluation_job(payload: dict):
    """Evaluate every saved checkpoint and refresh the final evaluation table."""
    try:
        return training.start_evaluation(payload)
    except training.JobError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/api/evaluate/status")
def get_evaluation_status():
    return training.status()


@app.get("/api/checkpoints")
def get_checkpoints():
    return training.checkpoint_info()


@app.post("/api/env/test")
def test_environment(payload: dict):
    """Instantiate an environment from a flat config and probe it.

    Body: a flat training config (the same shape the Setup page submits) or a
    ready ``environment`` section.  Returns the observed spec (obs dim, action
    count, sample obstacle layout) so the UI can validate a custom environment
    source (builtin v1/v2 or an external module) before training on it.
    """
    from rl.frames import serialize_obstacles

    source = payload.get("env_source") or payload.get("source") or "builtin"
    module = payload.get("env_module") or payload.get("module") or None

    if source == "module" and not module:
        raise HTTPException(
            status_code=400,
            detail="External environment selected but no module path given.",
        )

    try:
        env = env_factory.create_env(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to create env: {exc}")

    try:
        obs, info = env.reset(seed=0)
        sample_steps = []
        for _ in range(3):
            action = int(env.action_space.sample())
            obs, reward, terminated, truncated, info = env.step(action)
            sample_steps.append({
                "action": action,
                "reward": float(reward),
                "terminated": bool(terminated),
            })
            if terminated or truncated:
                obs, info = env.reset(seed=1)

        action_space = env.action_space
        n_actions = getattr(action_space, "n", None)
        obs_shape = list(getattr(env.observation_space, "shape", []))

        obstacles = serialize_obstacles(getattr(env, "obstacles", None))

        return {
            "ok": True,
            "source": source,
            "module": module,
            "env_class": type(env).__name__,
            "obs_dim": int(obs_shape[0]) if obs_shape else int(np.asarray(obs).size),
            "obs_shape": obs_shape,
            "n_actions": int(n_actions) if n_actions is not None else None,
            "world_size": float(getattr(env, "world_size", 0.0) or 0.0),
            "num_obstacles": len(obstacles),
            "obstacle_shapes": sorted({o["shape"] for o in obstacles}) or [],
            "obstacles": obstacles,
            "sample_steps": sample_steps,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Env probe failed: {exc}")


# ─── Saved Custom Environment Management ─────────────────────────────────


@app.get("/api/environments")
def list_environments():
    """List all saved custom environments."""
    from simulation.env_manager import list_saved_environments
    return list_saved_environments()


@app.get("/api/environments/{name}")
def get_environment(name: str):
    """Get a specific saved environment's full spec (metadata + env section)."""
    from simulation.env_manager import get_environment_spec
    try:
        return get_environment_spec(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Environment '{name}' not found")


@app.post("/api/environments")
def save_environment(payload: dict):
    """Save a new custom environment from the UI."""
    from simulation.env_manager import save_custom_environment

    name = payload.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="Environment name is required")

    try:
        metadata = save_custom_environment(
            name=name,
            layout=payload.get("layout"),
            world_size=payload.get("world_size", 20.0),
            max_steps=payload.get("max_steps", 300),
            frame_skip=payload.get("frame_skip", 3),
            sensor_range=payload.get("sensor_range", 12.0),
            robot_radius=payload.get("robot_radius", 0.35),
            target_radius=payload.get("target_radius", 0.8),
            max_speed=payload.get("max_speed", 0.35),
            turn_angle_deg=payload.get("turn_angle_deg", 30.0),
            description=payload.get("description", ""),
            overwrite=payload.get("overwrite", False),
        )
        return {"status": "ok", "metadata": metadata}
    except FileExistsError:
        raise HTTPException(
            status_code=409,
            detail=f"Environment '{name}' already exists. Set overwrite=true to replace.",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/api/environments/{name}")
def delete_environment(name: str):
    """Delete a saved custom environment."""
    from simulation.env_manager import delete_custom_environment
    try:
        delete_custom_environment(name)
        return {"status": "ok", "message": f"Environment '{name}' deleted"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Environment '{name}' not found")


@app.post("/api/live/toggle")
def toggle_live_view(payload: dict):
    """Enable or disable live frame streaming from the active training/eval job."""
    enabled = bool(payload.get("enabled", False))
    return training.set_live_view(enabled)


@app.get("/api/live/status")
def get_live_view_status():
    return training.live_view_status()


async def _stream_active_job(websocket: WebSocket) -> None:
    """Stream frames produced by a running training / evaluation job."""
    info = training.live_view_status()
    if not info["active"] or not info["live_enabled"]:
        await websocket.send_json({
            "error": "No active job with live view enabled. Start a job with live "
                     "view or toggle it on from the Results page."
        })
        await websocket.close()
        return

    last_ts = 0.0
    try:
        while True:
            frame, ts = training.peek_live_frame()
            if frame is not None and ts != last_ts:
                last_ts = ts
                await websocket.send_json(frame)
            await asyncio.sleep(0.05)  # ~20 fps (frames are produced at ~30 fps)
    except WebSocketDisconnect:
        pass
    except Exception:
        await websocket.close()


@app.websocket("/ws/live")
async def live_simulation(websocket: WebSocket):
    """Stream a live robot navigation episode.

    The client sends one of:
      - {"live": true}              -> stream frames from the active training/eval job
      - {"model": "mlp"|"kan"}      -> run a saved checkpoint (original behavior)
      - {"manual": true,            -> YOU drive: send {"action": 0..5} messages
         "model": "mlp"|"kan",         to step the env; frames include the model's
         "env_config": {...}}          suggested action + Q-values when a
                                     checkpoint exists (ghost advice)
    """
    await websocket.accept()

    try:
        first_message = await websocket.receive_json()
    except Exception:
        first_message = {}

    model_type = first_message.get("model", "kan")
    live_requested = bool(first_message.get("live", False))
    manual_requested = bool(first_message.get("manual", False))

    if live_requested:
        await _stream_active_job(websocket)
        return

    try:
        if manual_requested:
            # Manual play: env from an explicit env_config (e.g. the custom
            # map currently open on the Setup page) or the checkpoint's env.
            env_config = first_message.get("env_config") or None
            sim = LiveSimulator(
                model_type=model_type, env_config=env_config, auto=False
            )
        else:
            sim = LiveSimulator(model_type=model_type)
    except Exception as exc:
        await websocket.send_json({"error": str(exc)})
        await websocket.close()
        return

    try:
        if manual_requested:
            # Send the initial state so the canvas is not blank, then step
            # only when the user provides input.
            await websocket.send_json(sim.initial_frame())
            while True:
                message = await websocket.receive_json()

                if message.get("reset"):
                    sim.reset()
                    frame = sim.initial_frame()
                elif "action" in message:
                    frame = sim.step(int(message["action"]))
                else:
                    continue

                await websocket.send_json(frame)
        else:
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