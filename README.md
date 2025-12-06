# Battle Arena Custom Agent Setup

## Installation

Install dependencies either using the `pyproject.toml` file or the `requirements.txt` file. The dependencies listed in both files are the same, except the `requirements.txt` file also lists `matplotlib`, which is used during development to visualize results. If an error indicates that the package `arena` is not installed. run `pip install -e .`.

## Running Custom Agents

Located in the `src/arena/policies/custom.py` folder are two custom agents:
1. A pretrained DQN agent implemented in the `DQNInferPolicy` class.
2. A heuristic-based agent implemented in the `HeuristicPolicy` class.

To select which implementation is used when loading the `Custom` class, simply change the class that `Custom` inherits from to either agent. The DQN agent loads weights from the file `dqn/checkpoints/model_100.pt` by default. This path can be changed to the location of any compatible weights file. The heuristic agent does not require weights to run.

## Development Scripts

The `scripts` directory contains a variety of scripts used for development. The `train.py` configures the hyperparameters for DQN training, and the `plot_training.py` script visualizes the training statistics. The `config.py` script configures the parameters for evaluating and saving an agent's performance over multiple seeds and episodes. The `evaluate.py` script runs this configuration. Then, the `analyze.py` and `plot_evaluation.py` scripts analyze and visualize these results.