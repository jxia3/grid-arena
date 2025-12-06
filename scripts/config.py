from dataclasses import dataclass

@dataclass
class Config:
    num_episodes: int
    seed: int
    output_directory: str

NUM_EPISODES: int = 100
NUM_SEEDS: int = 100
SEED: int = 1337
NUM_WORKERS: int = 10

OUTPUT_DIRECTORY: str = "data/dqn_test"
OUTPUT_NAME: str = "trajectories"
METRICS_FILE: str = "metrics.json"

CONFIGS: list[Config] = [Config(
    num_episodes=NUM_EPISODES,
    seed=s,
    output_directory=f"{OUTPUT_DIRECTORY}/{OUTPUT_NAME}-{s}",
) for s in range(SEED, SEED + NUM_SEEDS)]