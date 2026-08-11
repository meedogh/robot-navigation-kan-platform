from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("experiments/results")


def load_log(model_type: str):
    path = RESULTS_DIR / f"custom_dqn_{model_type}_train_log.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing log file: {path}. "
            f"Train the {model_type} model first."
        )

    return pd.read_csv(path)


def main():
    mlp_log = load_log("mlp")
    kan_log = load_log("kan")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(mlp_log["training_step"], mlp_log["mean_reward"], label="MLP")
    plt.plot(kan_log["training_step"], kan_log["mean_reward"], label="KAN")
    plt.xlabel("Training Step")
    plt.ylabel("Mean Reward")
    plt.title("Reward Comparison")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(
        mlp_log["training_step"],
        mlp_log["success_rate"] * 100,
        label="MLP"
    )
    plt.plot(
        kan_log["training_step"],
        kan_log["success_rate"] * 100,
        label="KAN"
    )
    plt.xlabel("Training Step")
    plt.ylabel("Success Rate %")
    plt.title("Success Rate Comparison")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    output_path = RESULTS_DIR / "mlp_vs_kan_comparison.png"
    plt.savefig(output_path, dpi=200)

    print(f"Comparison plot saved to: {output_path}")

    print("\nFinal MLP metrics:")
    print(mlp_log.iloc[-1])

    print("\nFinal KAN metrics:")
    print(kan_log.iloc[-1])

    plt.show()


if __name__ == "__main__":
    main()