import json
import statistics

from arena.entities import Trajectory
from arena.utilities.loader import Loader
from config import CONFIGS, METRICS_FILE, NUM_EPISODES, OUTPUT_DIRECTORY

episode_lengths = [[] for _ in range(NUM_EPISODES)]
agent_samples = {}

def add_trajectory(trajectory: Trajectory):
    episode = trajectory.episode
    episode_length = len(trajectory.alive_status)
    episode_lengths[episode].append(episode_length)

    for a in range(len(trajectory.agent_names)):
        agent = trajectory.agent_names[a]
        if agent not in agent_samples:
            agent_samples[agent] = {
                "total_rewards": [[] for _ in range(NUM_EPISODES)],
                "alive_durations": [[] for _ in range(NUM_EPISODES)],
            }
        samples = agent_samples[agent]

        total_reward = float(trajectory.cumulative_returns[-1][a])
        alive_duration = 0
        while alive_duration < episode_length and trajectory.alive_status[alive_duration][a]:
            alive_duration += 1
        samples["total_rewards"][episode].append(total_reward)
        samples["alive_durations"][episode].append(alive_duration)

for config in CONFIGS:
    loader = Loader(input_dir=config.output_directory)
    for session_directory in loader.list_sessions():
        print("Loading session", session_directory.name, "from", config.output_directory)
        data = loader.load_session(session_directory.name)
        for trajectory, _ in data:
            add_trajectory(trajectory)
print("Loaded all sessions")

def calculate_deviation(nums: list[float]) -> float:
    deviation = statistics.stdev(nums)
    return deviation / (len(nums) ** 0.5)

def calculate_statistics(nums: list[list[float]]) -> tuple[list[float], list[float]]:
    means = [statistics.mean(n) for n in nums]
    deviations = [calculate_deviation(n) for n in nums]
    return (means, deviations)

length_means, length_deviations = calculate_statistics(episode_lengths)
metrics = {}
for agent in agent_samples:
    samples = agent_samples[agent]
    reward_means, reward_deviations = calculate_statistics(samples["total_rewards"])
    alive_means, alive_deviations = calculate_statistics(samples["alive_durations"])
    metrics[agent] = {
        "reward_means": reward_means,
        "reward_deviations": reward_deviations,
        "alive_means": alive_means,
        "alive_deviations": alive_deviations,
    }

with open(f"{OUTPUT_DIRECTORY}/{METRICS_FILE}", "w") as file:
    json.dump({
        "length_means": length_means,
        "length_deviations": length_deviations,
        "agent_metrics": metrics,
    }, file, indent=4)
print("Saved episode metrics")