"""Export trained DQN checkpoints to ONNX for deployment in external simulators.

An ONNX file of the Q-network can be dropped into Unity (Sentis), Gazebo-side
tooling, or any ONNX runtime - that is the "train here, run it there" path of
the platform.  Works for **both** model types: the MLP is plain Linear/ReLU,
and the custom KAN layer is built from clamp/abs/relu/einsum ops that export
cleanly.

CLI::

    python -m rl.model_export --model mlp
    python -m rl.model_export --model kan --out my_kan.onnx

The exported graph takes input ``observations`` with shape ``(batch, obs_dim)``
(float32, the platform's normalized observation vector) and returns
``q_values`` with shape ``(batch, n_actions)``.
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from rl.model_factory import create_qnetwork_from_arch, load_arch


BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = BASE_DIR / "experiments" / "checkpoints"

ONNX_INPUT_NAME = "observations"
ONNX_OUTPUT_NAME = "q_values"


def resolve_checkpoint_path(
    model_type: str, checkpoint_dir: Optional[Path] = None
) -> Path:
    """Best checkpoint for ``model_type`` (falls back to the final one)."""
    checkpoint_dir = Path(checkpoint_dir or CHECKPOINT_DIR)
    best = checkpoint_dir / f"custom_dqn_{model_type}_best.pt"
    final = checkpoint_dir / f"custom_dqn_{model_type}.pt"
    if best.exists():
        return best
    if final.exists():
        return final
    raise FileNotFoundError(
        f"No checkpoint for model '{model_type}' in {checkpoint_dir} "
        f"(looked for {best.name} / {final.name})"
    )


def _dims_from_state_dict(model_type: str, state: Dict) -> Tuple[int, int]:
    """Recover (obs_dim, action_dim) from raw checkpoint tensors.

    Avoids instantiating the environment - important when the checkpoint was
    trained on an external simulator that is not running right now.
    """
    if model_type == "mlp":
        first = state.get("network.0.weight")
        last = state.get("network.4.weight")
        if first is None or last is None:
            raise ValueError(
                "Unrecognized MLP checkpoint layout (missing network.0 / "
                "network.4 weights)."
            )
        return int(first.shape[1]), int(last.shape[0])

    if model_type == "kan":
        input_coeffs = state.get("kan_input.coefficients")
        output_coeffs = state.get("kan_output.coefficients")
        if input_coeffs is None or output_coeffs is None:
            raise ValueError(
                "Unrecognized KAN checkpoint layout (missing kan_input / "
                "kan_output coefficients)."
            )
        # KANLayer coefficients: (in_features, out_features, grid_size)
        return int(input_coeffs.shape[0]), int(output_coeffs.shape[1])

    raise ValueError(f"model_type must be 'mlp' or 'kan', got {model_type!r}")



def export_checkpoint_to_onnx(
    model_type: str,
    checkpoint_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
    checkpoint_dir: Optional[Path] = None,
    opset_version: int = 13,
) -> Path:
    """Export the checkpoint's Q-network to ``out_path`` (or a default path)."""
    checkpoint_dir = Path(checkpoint_dir or CHECKPOINT_DIR)
    checkpoint_path = Path(
        checkpoint_path or resolve_checkpoint_path(model_type, checkpoint_dir)
    )
    out_path = Path(out_path or checkpoint_path.with_suffix(".onnx"))

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "policy" in state:
        state = state["policy"]
    obs_dim, action_dim = _dims_from_state_dict(model_type, state)

    arch = load_arch(checkpoint_dir, model_type)
    model = create_qnetwork_from_arch(model_type, obs_dim, action_dim, arch)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, obs_dim, dtype=torch.float32)

    def _export(**kwargs):
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            input_names=[ONNX_INPUT_NAME],
            output_names=[ONNX_OUTPUT_NAME],
            dynamic_axes={
                ONNX_INPUT_NAME: {0: "batch"},
                ONNX_OUTPUT_NAME: {0: "batch"},
            },
            **kwargs,
        )

    try:
        # The legacy TorchScript exporter gives the widest runtime
        # compatibility (Unity Sentis etc.) and supports older opsets; torch
        # >= 2.9 defaults to the dynamo exporter, so opt out explicitly.
        _export(dynamo=False, opset_version=opset_version)
    except TypeError:
        # Very old torch without the dynamo kwarg.
        _export(opset_version=opset_version)

    _verify_with_onnxruntime(out_path, model, obs_dim)
    return out_path


def _verify_with_onnxruntime(out_path: Path, model, obs_dim: int) -> None:
    """Compare ONNX runtime output against torch when onnxruntime is present."""
    try:
        import onnxruntime as ort
    except ImportError:
        print(
            "[model_export] note: onnxruntime not installed - skipping the "
            "numerical check (pip install onnxruntime to enable it)."
        )
        return

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    inputs = torch.randn(64, obs_dim, dtype=torch.float32)
    with torch.no_grad():
        expected = model(inputs).numpy()
    actual = session.run(None, {ONNX_INPUT_NAME: inputs.numpy()})[0]
    max_diff = float(abs(expected - actual).max())
    print(f"[model_export] onnxruntime check: max |Q_torch - Q_onnx| = {max_diff:.2e}")
    if max_diff > 1e-3:
        raise RuntimeError(
            f"ONNX output mismatch (max diff {max_diff}) - export is not faithful."
        )


def _main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rl.model_export",
        description="Export a trained DQN checkpoint to ONNX.",
    )
    parser.add_argument("--model", choices=["mlp", "kan"], required=True)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint .pt path")
    parser.add_argument("--out", default=None, help="Output .onnx path")
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()

    out = export_checkpoint_to_onnx(
        args.model,
        checkpoint_path=args.checkpoint,
        out_path=args.out,
        checkpoint_dir=args.checkpoint_dir,
    )
    print(f"ONNX model written to: {out.resolve()}")
    print(f"  input  : '{ONNX_INPUT_NAME}'  shape (batch, obs)  float32")
    print(f"  output : '{ONNX_OUTPUT_NAME}' shape (batch, n_actions) float32")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

