import json
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from rl.train_custom_dqn import DEFAULT_TRAINING_CONFIG, train as train_impl, validate_training_config
from rl import evaluate_saved_models as evaluate_module


BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = BASE_DIR / "experiments" / "checkpoints"
RESULTS_DIR = BASE_DIR / "experiments" / "results"


class JobError(Exception):
    """Raised when a job cannot be started (e.g. another job is already running)."""


_state: Dict[str, Any] = {
    "lock": threading.RLock(),
    "job": None,
    "thread": None,
    "cancel_event": None,
}


def _make_job(job_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "job_id": uuid.uuid4().hex[:8],
        "job_type": job_type,
        "state": "running",
        "message": "starting",
        "error": None,
        "started_at": time.time(),
        "finished_at": None,
        "payload": payload,
        "log": [],
        "progress": 0.0,
        "progress_label": "starting",
        "live_enabled": bool(payload.get("live", False)),
        "latest_frame": None,
        "frame_ts": 0.0,
        "steps_done": 0,
        "total_steps": 0,
    }


def _finish_job(job: Dict[str, Any], state: str, message: str) -> None:
    job["state"] = state
    job["message"] = message
    job["finished_at"] = time.time()


def _training_worker(payload: Dict[str, Any]) -> None:
    job = None
    with _state["lock"]:
        job = _state["job"]
        cancel_event = _state["cancel_event"]

    config = payload.get("config", {})
    cancel_event = cancel_event or threading.Event()

    last_frame_time = {"t": 0.0}

    def on_progress(row: Dict[str, Any]) -> None:
        with _state["lock"]:
            if job is not None and job["state"] == "running":
                job["log"].append(row)

    def on_frame(frame: Dict[str, Any]) -> None:
        now = time.monotonic()
        if now - last_frame_time["t"] < 1.0 / 30.0:
            return
        last_frame_time["t"] = now
        with _state["lock"]:
            if job is None or not job["live_enabled"] or job["state"] != "running":
                return
            job["latest_frame"] = frame
            job["frame_ts"] = now

    def on_status(st: Dict[str, Any]) -> None:
        total = int(st.get("total_steps") or 0)
        step = int(st.get("training_step") or 0)
        value = (step / total) if total > 0 else 0.0
        with _state["lock"]:
            if job is not None:
                job["steps_done"] = step
                job["total_steps"] = total
                job["progress"] = max(0.0, min(1.0, value))
                job["progress_label"] = f"training step {step:,} / {total:,}"

    try:
        with _state["lock"]:
            if job is not None:
                job["message"] = "training started"

        result = train_impl(
            checkpoint_dir=CHECKPOINT_DIR,
            results_dir=RESULTS_DIR,
            cancel_event=cancel_event,
            config=config,
            progress_callback=on_progress,
            frame_callback=on_frame,
            status_callback=on_status,
        )

        status = result.get("status", "completed") if isinstance(result, dict) else "completed"

        with _state["lock"]:
            if job is not None:
                if status == "stopped":
                    _finish_job(job, "stopped", "Training stopped by user.")
                else:
                    job["progress"] = 1.0
                    job["progress_label"] = "completed"
                    _finish_job(job, "completed", "Training finished.")
    except Exception as exc:
        with _state["lock"]:
            if job is not None:
                _finish_job(job, "failed", str(exc))
                job["error"] = traceback.format_exc()
    finally:
        with _state["lock"]:
            _state["thread"] = None
            _state["cancel_event"] = None


def _evaluation_worker(payload: Dict[str, Any]) -> None:
    job = None
    with _state["lock"]:
        job = _state["job"]

    episodes = int(payload.get("episodes", 100))
    seed_base = int(payload.get("seed_base", 999_999))

    last_frame_time = {"t": 0.0}

    def on_progress(row: Dict[str, Any]) -> None:
        with _state["lock"]:
            if job is not None and job["state"] == "running":
                job["log"].append(row)

    def on_frame(frame: Dict[str, Any]) -> None:
        now = time.monotonic()
        if now - last_frame_time["t"] < 1.0 / 30.0:
            return
        last_frame_time["t"] = now
        with _state["lock"]:
            if job is None or not job["live_enabled"] or job["state"] != "running":
                return
            job["latest_frame"] = frame
            job["frame_ts"] = now

    def on_episode(done_count: int, total_count: int) -> None:
        value = (done_count / total_count) if total_count > 0 else 0.0
        with _state["lock"]:
            if job is not None:
                job["progress"] = max(0.0, min(1.0, value))
                job["progress_label"] = f"evaluating episode {done_count:,} / {total_count:,}"

    try:
        with _state["lock"]:
            if job is not None:
                job["message"] = "evaluating saved checkpoints"

        evaluate_module.main(
            checkpoint_dir=CHECKPOINT_DIR,
            results_dir=RESULTS_DIR,
            episodes=episodes,
            seed_base=seed_base,
            progress_callback=on_progress,
            frame_callback=on_frame,
            episode_callback=on_episode,
        )

        with _state["lock"]:
            if job is not None:
                job["progress"] = 1.0
                job["progress_label"] = "completed"
                _finish_job(job, "completed", "Evaluation finished.")
    except Exception as exc:
        with _state["lock"]:
            if job is not None:
                _finish_job(job, "failed", str(exc))
                job["error"] = traceback.format_exc()
    finally:
        with _state["lock"]:
            _state["thread"] = None
            _state["cancel_event"] = None


def _start_job(job_type: str, payload: Dict[str, Any], worker) -> Dict[str, Any]:
    with _state["lock"]:
        if _state["thread"] is not None and _state["thread"].is_alive():
            current = _state["job"]
            kind = current["job_type"] if current else "job"
            raise JobError(
                f"Another {kind} is already running (job {current['job_id']})."
            )

        job = _make_job(job_type, payload)
        cancel_event = threading.Event()
        thread = threading.Thread(target=worker, args=(payload,), daemon=True)

        _state["job"] = job
        _state["cancel_event"] = cancel_event
        _state["thread"] = thread

        thread.start()

    return job


def start_training(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Start a background training job. Raises JobError if another job is active."""
    raw_config = payload.get("config") or {}
    if not isinstance(raw_config, dict):
        raise JobError("config must be a JSON object")

    # Fail fast with a helpful message instead of dying in the worker thread.
    try:
        validate_training_config(raw_config)
    except (ValueError, TypeError) as exc:
        raise JobError(f"Invalid training config: {exc}")

    return _start_job(
        "train",
        {"config": dict(raw_config), "live": bool(payload.get("live", False))},
        _training_worker,
    )


def start_evaluation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Start a background saved-model evaluation job.

    Raises JobError if another job is active, or if the request is invalid.
    """
    episodes = payload.get("episodes", 100)
    seed_base = payload.get("seed_base", 999_999)

    if not isinstance(episodes, int) or not isinstance(seed_base, int):
        raise JobError("episodes and seed_base must be integers")

    if episodes < 1 or episodes > 10_000:
        raise JobError("episodes must be between 1 and 10000")

    return _start_job(
        "evaluate",
        {
            "episodes": episodes,
            "seed_base": seed_base,
            "live": bool(payload.get("live", False)),
        },
        _evaluation_worker,
    )


def status() -> Dict[str, Any]:
    """Overall job status: busy flag plus a compact view of the current (or last) job."""
    with _state["lock"]:
        thread = _state["thread"]
        job = _state["job"]

        busy = thread is not None and thread.is_alive()

        result: Dict[str, Any] = {"busy": busy}

        if job is not None:
            result["job"] = {
                "job_id": job["job_id"],
                "job_type": job["job_type"],
                "state": job["state"],
                "message": job["message"],
                "error": job["error"],
                "started_at": job["started_at"],
                "finished_at": job["finished_at"],
                "log_len": len(job["log"]),
                "payload": job["payload"],
                "progress": job.get("progress", 0.0),
                "progress_label": job.get("progress_label", ""),
                "live_enabled": job.get("live_enabled", False),
                "steps_done": job.get("steps_done", 0),
                "total_steps": job.get("total_steps", 0),
            }
        else:
            result["job"] = None

        return result


def progress() -> Optional[Dict[str, Any]]:
    """Training progress: the evaluation rows collected so far for the current job."""
    with _state["lock"]:
        job = _state["job"]

        if job is None or job["job_type"] != "train":
            return None

        return {
            "job_id": job["job_id"],
            "state": job["state"],
            "message": job["message"],
            "rows": list(job["log"]),
        }


def stop() -> Dict[str, Any]:
    """Request a clean stop of the running job by setting its cancel event."""
    with _state["lock"]:
        cancel_event = _state["cancel_event"]

        if cancel_event is None:
            return {"requested": False, "message": "No active job to stop."}

        cancel_event.set()
        return {"requested": True, "message": "Stop requested."}


def last_run_config(model_type: str) -> Optional[Dict[str, Any]]:
    """Return the config JSON persisted next to a training log, if present."""
    if model_type not in ("mlp", "kan"):
        return None

    path = RESULTS_DIR / f"custom_dqn_{model_type}_config.json"
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def checkpoint_info() -> list:
    """List the files currently in the checkpoints directory."""
    files = []
    if CHECKPOINT_DIR.exists():
        for path in sorted(CHECKPOINT_DIR.iterdir()):
            if path.is_file():
                files.append({
                    "name": path.name,
                    "size": path.stat().st_size,
                    "modified": path.stat().st_mtime,
                })
    return files


def set_live_view(enabled: bool) -> Dict[str, Any]:
    """Enable or disable live frame streaming from the active job mid-run."""
    with _state["lock"]:
        job = _state["job"]
        thread = _state["thread"]
        running = thread is not None and thread.is_alive()

        if job is None or not running:
            return {
                "active": False,
                "live_enabled": False,
                "message": "No active job.",
            }

        job["live_enabled"] = bool(enabled)
        if not enabled:
            job["latest_frame"] = None
            job["frame_ts"] = 0.0

        return {
            "active": True,
            "job_id": job["job_id"],
            "job_type": job["job_type"],
            "live_enabled": job["live_enabled"],
        }


def live_view_status() -> Dict[str, Any]:
    """Whether the active job is streaming frames to the Live page right now."""
    with _state["lock"]:
        thread = _state["thread"]
        job = _state["job"]

        busy = thread is not None and thread.is_alive()
        active = busy and job is not None and job["job_type"] in ("train", "evaluate")

        return {
            "active": active,
            "busy": busy,
            "live_enabled": bool(job is not None and job["live_enabled"]),
            "job_id": job["job_id"] if job is not None else None,
            "job_type": job["job_type"] if job is not None else None,
        }


def peek_live_frame():
    """Return (latest_frame, frame_ts) from the active job, or (None, 0.0)."""
    with _state["lock"]:
        job = _state["job"]
        if job is None or not job.get("live_enabled") or job.get("latest_frame") is None:
            return None, 0.0
        return job["latest_frame"], job["frame_ts"]


def presets() -> Dict[str, Dict[str, Any]]:
    """Preset training runs offered to the user (merged over DEFAULT_TRAINING_CONFIG)."""
    return {
        "quick": {
            "total_steps": 20_000,
            "eval_every": 2_000,
            "eval_episodes": 10,
        },
        "standard": {},
        "thorough": {
            "total_steps": 300_000,
            "eval_every": 10_000,
            "eval_episodes": 30,
        },
    }