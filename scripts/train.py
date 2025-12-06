from arena import run
from arena.policies import custom
from arena.policies.custom import TrainParams
from config import SEED

TRAIN_PARAMS = TrainParams(
    arena_size=(15, 15),
    arena_steps=(100, 500),
    num_opponents=10,
    seed=SEED,

    learning_rate=0.001,
    discount_factor=0.99,
    replay_size=65_536,
    sample_count=128,
    random_range=(1.0, 0.05),
    random_decay=0.998,
    max_value=100.0,
    max_gradient=20.0,

    warmup_seeds=1,
    train_seeds=100,
    train_episodes=50,
    train_count=32,

    metrics_file="dqn/metrics.json",
    checkpoint_directory="dqn/checkpoints",
    save_interval=5,
)

if __name__ == "__main__":
    custom.train_dqn(TRAIN_PARAMS)