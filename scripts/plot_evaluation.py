import json
from matplotlib import pyplot as plt
import numpy as np

from config import METRICS_FILE, OUTPUT_DIRECTORY

AGENT_NAME: str = "Custom"

metrics = None
with open(f"{OUTPUT_DIRECTORY}/{METRICS_FILE}", "r") as file:
    metrics = json.load(file)
assert metrics is not None
agent_metrics = metrics["agent_metrics"]

def plot_episode_lengths(means: list[float], deviations: list[float]):
    assert len(means) == len(deviations)
    mean_array = np.array(means, dtype=np.float32)
    deviation_array = np.array(deviations, dtype=np.float32)
    index_array = np.arange(mean_array.shape[0], dtype=np.int32)

    plt.rcParams["font.size"] = 12
    figure, axes = plt.subplots(figsize=(6.4 * 1.5, 4.8 * 1.5))
    axes.plot(index_array, mean_array)
    axes.fill_between(
        index_array,
        mean_array - deviation_array,
        mean_array + deviation_array,
        alpha=0.1,
    )

    axes.set_title("Mean Episode Lengths")
    axes.set_xlabel("Episode Number")
    axes.set_ylabel("Length")
    axes.grid(visible=True, alpha=0.5)

    figure.tight_layout()
    plt.show()

def plot_agent_metrics(agent_metrics: dict[str, dict[str, list[float]]], metric: str, name: str):
    agents = list(agent_metrics.keys())
    agents.sort(key=lambda a: not a.startswith(AGENT_NAME))
    for agent in agents:
        assert f"{metric}_means" in agent_metrics[agent]
        assert f"{metric}_deviations" in agent_metrics[agent]

    plt.rcParams["font.size"] = 12
    figure, axes = plt.subplots(figsize=(6.4 * 1.5, 4.8 * 1.5))
    for agent in agents:
        means = np.array(agent_metrics[agent][f"{metric}_means"], dtype=np.float32)
        deviations = np.array(agent_metrics[agent][f"{metric}_deviations"], dtype=np.float32)
        indices = np.arange(means.shape[0], dtype=np.int32)
        axes.plot(indices, means, label=agent)
        axes.fill_between(
            indices,
            means - deviations,
            means + deviations,
            alpha=0.1,
            zorder=2.001 if agent == AGENT_NAME else 2.0,
        )

    axes.set_title(f"Mean {name}")
    axes.set_xlabel("Episode Number")
    axes.set_ylabel(name)
    axes.grid(visible=True, alpha=0.5)
    axes.legend(loc="lower right")

    figure.tight_layout()
    plt.show()

plot_episode_lengths(metrics["length_means"], metrics["length_deviations"])
plot_agent_metrics(agent_metrics, "reward", "Reward")
plot_agent_metrics(agent_metrics, "alive", "Alive Duration")