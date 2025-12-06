from collections import deque
import json
from matplotlib import pyplot as plt
from typing import Any

from train import TRAIN_PARAMS

ARENA_WINDOW: int = 10
REWARD_WINDOW: int = 10

data = None
with open(TRAIN_PARAMS.metrics_file, "r") as file:
    data = json.load(file)
assert data is not None

def calculate_reward(data: dict[str, Any]) -> tuple[list[int], list[float]]:
    episodes = []
    reward = []

    episode_cutoff = TRAIN_PARAMS.train_episodes - REWARD_WINDOW
    current_reward = 0.0
    total_reward = 0.0
    reward_window = deque()

    for episode in range(len(data["rewards"])):
        if data["episodes"][episode] <= episode_cutoff:
            continue
        current_reward += data["rewards"][episode]
        if data["episodes"][episode] < TRAIN_PARAMS.train_episodes:
            continue

        arena_reward = current_reward / REWARD_WINDOW
        current_reward = 0.0
        total_reward += arena_reward
        reward_window.append(arena_reward)
        if len(reward_window) > ARENA_WINDOW:
            total_reward -= reward_window.popleft()

        if len(reward_window) == ARENA_WINDOW:
            episodes.append(episode + 1)
            reward.append(total_reward / ARENA_WINDOW)

    return (episodes, reward)

def plot_reward(data: dict[str, Any]):
    episodes, reward = calculate_reward(data)

    plt.rcParams["font.size"] = 12
    figure, axes = plt.subplots(figsize=(6.4 * 1.5, 4.8 * 1.5))
    axes.plot(episodes, reward)
    axes.set_title("Training Reward")
    axes.set_xlabel("Episode Number")
    axes.set_ylabel("Smoothed Reward")
    axes.grid(visible=True, alpha=0.5)

    figure.tight_layout()
    plt.show()

def plot_losses(mean_losses: list[float]):
    episodes = list(range(1, len(mean_losses) + 1))

    plt.rcParams["font.size"] = 12
    figure, axes = plt.subplots(figsize=(6.4 * 1.5, 4.8 * 1.5))
    axes.plot(episodes, mean_losses)
    axes.set_title("Training Loss")
    axes.set_xlabel("Episode Number")
    axes.set_ylabel("Mean Loss")
    axes.grid(visible=True, alpha=0.5)

    figure.tight_layout()
    plt.show()

def plot_random_chances(random_chances: list[float]):
    episodes = list(range(1, len(random_chances) + 1))

    plt.rcParams["font.size"] = 12
    figure, axes = plt.subplots(figsize=(6.4 * 1.5, 4.8 * 1.5))
    axes.plot(episodes, random_chances)
    axes.set_title("Training Epsilon")
    axes.set_xlabel("Episode Number")
    axes.set_ylabel("Epsilon")
    axes.grid(visible=True, alpha=0.5)

    figure.tight_layout()
    plt.show()

plot_reward(data)
plot_losses(data["mean_losses"])
plot_random_chances(data["random_chances"])