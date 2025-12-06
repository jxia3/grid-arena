from __future__ import annotations

from abc import ABC, abstractmethod
import copy
import dataclasses
from dataclasses import dataclass
import json
import math
import numpy as np
from pathlib import Path
import random
from random import Random
import torch
from torch import nn, Tensor
from torch.optim import Adam, Optimizer
from typing import TYPE_CHECKING, Callable, Optional

from arena.constants import GRID_EMPTY, GRID_WALL, PLAYER_ID_OFFSET, Direction, Channel
from arena.schema import Action, Observation, Reward
from .policy import Policy

if TYPE_CHECKING:
    from arena.environment import Arena

VISION_RADIUS: int = 2
MAX_STEPS: int = 500
FEATURE_RADIUS: int = 3
DEBUG: bool = False

type Positions = dict[int, tuple[int, int]]

class RandomPolicy(Policy):
    @property
    def name(self) -> str:
        return "Custom (Random)"

    def get_action(self, observation: Observation) -> Action:
        return random.randint(0, len(Direction) - 1)

    def update(self, observation: Observation, action: Action, reward: Reward):
        pass

    def reset(self):
        pass

@dataclass
class BattleOutcomes:
    win: set[int]
    lose: set[int]

@dataclass
class Features:
    step: int
    actions: dict[Direction, np.ndarray]

class BeliefModel:
    identifier: int
    vision_radius: int
    feature_radius: int

    num_rows: int
    num_cols: int
    positions: dict[int, np.ndarray]
    strengths: dict[int, BattleOutcomes]
    step: int

    def __init__(self, identifier: int, vision_radius: int, feature_radius: int):
        self.identifier = identifier
        self.vision_radius = vision_radius
        self.feature_radius = feature_radius

        self.num_rows = -1
        self.num_cols = -1
        self.positions = {}
        self.strengths = {}
        self.strengths[identifier] = BattleOutcomes(win=set(), lose=set())
        self.step = -1

    def calculate_features(self, observation: Observation) -> Optional[Features]:
        if DEBUG:
            assert self.step >= 0

        grid, _, _ = self._split_observation(observation)
        if self.identifier not in grid:
            return None
        position = grid[self.identifier]
        actions = {}

        for direction in Direction:
            actions[direction] = np.zeros((3, self.feature_radius + 1), dtype=np.float32)
            next_position = self._get_next_position(position[0], position[1], direction)
            assert next_position is not None

            for r in range(self.feature_radius + 1):
                radius = self._get_radius(next_position[0], next_position[1], r)
                actions[direction][0][r] = self._calculate_win(radius)
                actions[direction][1][r] = self._calculate_lose(radius)
                actions[direction][2][r] = self._calculate_unknown(radius)

        return Features(step=self.step, actions=actions)

    def _calculate_win(self, positions: set[tuple[int, int]]) -> float:
        return self._calculate_probability(positions, lambda o, a: a in o.win)

    def _calculate_lose(self, positions: set[tuple[int, int]]) -> float:
        return self._calculate_probability(positions, lambda o, a: a in o.lose)

    def _calculate_unknown(self, positions: set[tuple[int, int]]) -> float:
        return self._calculate_probability(positions, lambda o, a: a not in o.win and a not in o.lose)

    def _calculate_probability(
        self,
        positions: set[tuple[int, int]],
        included: Callable[[BattleOutcomes, int], bool],
    ) -> float:
        if DEBUG:
            assert self.identifier in self.strengths
        outcomes = self.strengths[self.identifier]

        complement = 1.0
        for agent in self.positions:
            if not included(outcomes, agent):
                continue
            probability = 0.0
            for position in positions:
                probability += self.positions[agent][position[0]][position[1]]
            complement *= (1.0 - probability)

        return 1.0 - complement

    def update(self, observation: Observation):
        grid, before_combat, after_combat = self._split_observation(observation)
        self.step += 1

        visible_agents = set()
        for agent in grid:
            if agent != self.identifier:
                visible_agents.add(agent)
        for agent in after_combat:
            if agent != self.identifier:
                visible_agents.add(agent)

        self._remove_eliminated(grid, before_combat, after_combat)
        if self.identifier in grid:
            self._spread_positions(grid[self.identifier], visible_agents)
            self._set_positions(grid)
            self._set_positions(after_combat)
        self._update_strengths(before_combat, after_combat)

    def _split_observation(self, observation: Observation) -> tuple[Positions, Positions, Positions]:
        observation = observation.transpose(2, 0, 1)
        if self.num_rows == -1 or self.num_cols == -1:
            self.num_rows = observation.shape[1]
            self.num_cols = observation.shape[2]
        grid = self._get_positions(observation[Channel.MAP.value])
        before_combat = self._get_positions(observation[Channel.PRE_COMBAT.value])
        after_combat = self._get_positions(observation[Channel.POST_COMBAT.value])

        if DEBUG:
            assert len(observation.shape) == 3
            assert observation.shape[0] == len(Channel)
            assert observation.shape[1] == self.num_rows and observation.shape[2] == self.num_cols

        return (grid, before_combat, after_combat)

    def _get_positions(self, grid: np.ndarray) -> Positions:
        positions = {}
        rows, cols = np.nonzero((grid != GRID_EMPTY) & (grid != GRID_WALL))
        agents = grid[rows, cols]
        for a in range(len(agents)):
            positions[int(agents[a])] = (int(rows[a]), int(cols[a]))

        if DEBUG:
            reference_positions = self._get_reference_positions(grid)
            assert positions == reference_positions

        return positions

    def _get_reference_positions(self, grid: np.ndarray) -> Positions:
        if DEBUG:
            assert len(grid.shape) == 2
            assert grid.shape[0] == self.num_rows and grid.shape[1] == self.num_cols

        positions = {}
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                value = int(grid[r][c])
                if value != GRID_EMPTY and value != GRID_WALL:
                    positions[value] = (r, c)

        return positions

    def _remove_eliminated(self, grid: Positions, before_combat: Positions, after_combat: Positions):
        for agent in before_combat:
            if agent in after_combat or agent == self.identifier:
                continue
            if agent in grid:
                del grid[agent]
            if agent in self.positions:
                del self.positions[agent]

    def _spread_positions(self, position: tuple[int, int], visible_agents: set[int]):
        visible_positions = self._get_radius(position[0], position[1], self.vision_radius)
        for agent in self.positions:
            if DEBUG:
                assert agent != self.identifier
            if agent in visible_agents:
                continue
            self.positions[agent] = self._spread(self.positions[agent], visible_positions)

    def _spread(self, positions: np.ndarray, visible_positions: set[tuple[int, int]]) -> np.ndarray:
        if DEBUG:
            assert len(Direction) == 4
        next_positions = np.zeros_like(positions)
        probabilities = positions / len(Direction)

        next_positions[1:-1, :] += probabilities[2:, :] # Up
        next_positions[1, :] += probabilities[1, :]
        next_positions[1:-1, :] += probabilities[:-2, :] # Down
        next_positions[-2, :] += probabilities[-2, :]

        next_positions[:, 1:-1] += probabilities[:, 2:] # Left
        next_positions[:, 1] += probabilities[:, 1]
        next_positions[:, 1:-1] += probabilities[:, :-2] # Right
        next_positions[:, -2] += probabilities[:, -2]

        for position in visible_positions:
            next_positions[position[0]][position[1]] = 0.0
        remaining = np.sum(next_positions)
        if remaining > 0.0:
            next_positions /= remaining
        if DEBUG:
            reference_next = self._reference_spread(positions, visible_positions)
            assert np.allclose(next_positions, reference_next)

        return next_positions

    def _reference_spread(
        self,
        positions: np.ndarray,
        visible_positions: set[tuple[int, int]],
    ) -> np.ndarray:
        if DEBUG:
            assert len(positions.shape) == 2
            assert positions.shape[0] == self.num_rows and positions.shape[1] == self.num_cols
            assert positions.dtype == np.float32
        next_positions = np.zeros_like(positions)

        for r in range(positions.shape[0]):
            for c in range(positions.shape[1]):
                if positions[r][c] == 0.0:
                    continue

                moves = []
                for direction in Direction:
                    next_position = self._get_next_position(r, c, direction)
                    if next_position is not None:
                        moves.append(next_position)
                if DEBUG:
                    assert len(moves) > 0

                probability = positions[r][c] / len(moves)
                for move in moves:
                    next_positions[move[0]][move[1]] += probability

        if DEBUG:
            assert np.allclose(np.sum(next_positions), 1.0)
        for position in visible_positions:
            next_positions[position[0]][position[1]] = 0.0
        remaining = np.sum(next_positions)
        if remaining > 0.0:
            next_positions /= remaining

        return next_positions

    def _set_positions(self, positions: dict[int, tuple[int, int]]):
        for agent in positions:
            if agent == self.identifier:
                continue
            if agent not in self.positions:
                self.positions[agent] = np.zeros((self.num_rows, self.num_cols), dtype=np.float32)
            row, col = positions[agent]
            self.positions[agent].fill(0.0)
            self.positions[agent][row][col] = 1.0

    def _update_strengths(self, before_combat: Positions, after_combat: Positions):
        relations = self._calculate_relations(before_combat, after_combat)
        agents = list(relations.keys())

        for a in range(len(agents)):
            for b in range(a + 1, len(agents)):
                if len(relations[agents[a]]) > 1 or len(relations[agents[b]]) > 1:
                    continue
                related_a = next(iter(relations[agents[a]]))
                related_b = next(iter(relations[agents[b]]))
                if agents[a] != related_b or agents[b] != related_a:
                    continue

                if DEBUG:
                    assert agents[a] in after_combat or agents[b] in after_combat
                    assert agents[a] not in after_combat or agents[b] not in after_combat
                if agents[a] not in after_combat:
                    self._add_outcome(agents[b], agents[a])
                elif agents[b] not in after_combat:
                    self._add_outcome(agents[a], agents[b])

    def _calculate_relations(self, before_combat: Positions, after_combat: Positions) -> dict[int, set[int]]:
        for agent in after_combat:
            if agent not in before_combat:
                return {}
        agents = list(before_combat.keys())
        after_positions = {}

        for agent in agents:
            after_positions[agent] = set()
            if agent in after_combat:
                after_positions[agent].add(after_combat[agent])
                continue

            position = before_combat[agent]
            for direction in Direction:
                next_position = self._get_next_position(position[0], position[1], direction)
                if next_position is not None:
                    after_positions[agent].add(next_position)

        relations = {}
        for a in range(len(agents)):
            for b in range(a + 1, len(agents)):
                if not self._check_relation(before_combat, after_positions, agents[a], agents[b]):
                    continue
                if agents[a] not in relations:
                    relations[agents[a]] = set()
                if agents[b] not in relations:
                    relations[agents[b]]= set()
                relations[agents[a]].add(agents[b])
                relations[agents[b]].add(agents[a])

        return relations

    def _check_relation(
        self,
        before_combat: Positions,
        after_positions: dict[int, set[tuple[int, int]]],
        a: int,
        b: int,
    ) -> bool:
        if DEBUG:
            assert a in before_combat and a in after_positions
            assert b in before_combat and b in after_positions

        for after in after_positions[a]:
            if after in after_positions[b]:
                return True
        if before_combat[a] in after_positions[b] and before_combat[b] in after_positions[a]:
            return True

        return False

    def _add_outcome(self, winner: int, loser: int):
        if winner not in self.strengths:
            self.strengths[winner] = BattleOutcomes(win=set(), lose=set())
        if loser not in self.strengths:
            self.strengths[loser] = BattleOutcomes(win=set(), lose=set())
        # if DEBUG:
        #     assert winner not in self.strengths[loser].win
        #     assert loser not in self.strengths[winner].lose
        self.strengths[winner].win.add(loser)
        self.strengths[loser].lose.add(winner)

    def reset(self):
        self.positions.clear()
        self.step = -1

    def reset_strengths(self):
        self.strengths.clear()
        self.strengths[self.identifier] = BattleOutcomes(win=set(), lose=set())

    def _get_radius(self, row: int, col: int, radius: int) -> set[tuple[int, int]]:
        positions = set()
        for r in range(row - radius, row + radius + 1):
            for c in range(col - radius, col + radius + 1):
                if not self._check_valid_position(r, c) or self._check_wall_position(r, c):
                    continue
                if abs(r - row) + abs(c - col) > radius:
                    continue
                positions.add((r, c))
        return positions

    def _get_next_position(self, row: int, col: int, direction: Direction) -> Optional[tuple[int, int]]:
        next_row = row + int(direction.vector[0]) # type: ignore
        next_col = col + int(direction.vector[1]) # type: ignore
        if not self._check_valid_position(next_row, next_col):
            return None
        if not self._check_wall_position(next_row, next_col):
            return (next_row, next_col)
        return (row, col)

    def _check_valid_position(self, row: int, col: int):
        return 0 <= row and row < self.num_rows and 0 <= col and col < self.num_cols

    def _check_wall_position(self, row: int, col: int):
        return row == 0 or row == self.num_rows - 1 or col == 0 or col == self.num_cols - 1

class BeliefPolicy(Policy, ABC):
    belief: BeliefModel
    first_action: bool
    previous_features: Optional[Features]

    def __init__(self, identifier: int):
        super().__init__(identifier)
        self.belief = BeliefModel(identifier, VISION_RADIUS, FEATURE_RADIUS)
        self.first_action = True
        self.previous_features = None

    def get_action(self, observation: Observation) -> Action:
        if self.first_action:
            self.belief.update(observation)
            self.first_action = False
        features = self.belief.calculate_features(observation)
        self.previous_features = features
        if features is not None:
            return self._get_action(features)
        return random.randint(0, len(Direction) - 1)

    @abstractmethod
    def _get_action(self, features: Features) -> Action:
        raise NotImplementedError

    def update(self, observation: Observation, action: Action, reward: Reward):
        self.belief.update(observation)
        next_features = self.belief.calculate_features(observation)
        if self.previous_features is not None:
            self._update(self.previous_features, action, reward, next_features)
        self.previous_features = next_features

    @abstractmethod
    def _update(
        self,
        features: Features,
        action: Action,
        reward: Reward,
        next_features: Optional[Features],
    ):
        raise NotImplementedError

    def reset(self):
        self.belief.reset()
        self.first_action = True
        self.previous_features = None
        self._reset()

    def reset_strengths(self):
        self.belief.reset_strengths()

    @abstractmethod
    def _reset(self):
        raise NotImplementedError

class HeuristicPolicy(BeliefPolicy):
    WIN_VALUE: float = 1.0
    LOSE_VALUE: float = -2.0
    UNKNOWN_VALUE: float = 0.1
    VALUE_DISCOUNT: float = 0.75
    VALUE_THRESHOLD: float = 0.01

    @property
    def name(self) -> str:
        return "Custom (Heuristic)"

    def _get_action(self, features: Features) -> Action:
        actions = []
        max_value = -math.inf
        for direction in Direction:
            value = self._calculate_value(features.actions[direction])
            actions.append((direction, value))
            max_value = max(value, max_value)

        if DEBUG:
            assert max_value > -math.inf
        actions = list(filter(lambda a: a[1] >= max_value - self.VALUE_THRESHOLD, actions))
        if DEBUG:
            assert len(actions) > 0
        action = random.choice(actions)

        return action[0].value

    def _calculate_value(self, features: np.ndarray) -> float:
        if DEBUG:
            assert len(features.shape) == 2
            assert features.shape[0] == 3

        win_value = 0.0
        lose_value = 0.0
        unknown_value = 0.0
        discount = 1.0
        for r in range(features.shape[1]):
            win_value += features[0][r] * self.WIN_VALUE * discount
            lose_value += features[1][r] * self.LOSE_VALUE * discount
            unknown_value += features[2][r] * self.UNKNOWN_VALUE * discount
            discount *= self.VALUE_DISCOUNT

        return win_value + lose_value + unknown_value

    def _update(
        self,
        features: Features,
        action: Action,
        reward: Reward,
        next_features: Optional[Features],
    ):
        pass

    def _reset(self):
        pass

@dataclass
class TrainParams:
    arena_size: tuple[int, int]
    arena_steps: tuple[int, int]
    num_opponents: int
    seed: int

    learning_rate: float
    discount_factor: float
    replay_size: int
    sample_count: int
    random_range: tuple[float, float]
    random_decay: float
    max_value: float
    max_gradient: float

    warmup_seeds: int
    train_seeds: int
    train_episodes: int
    train_count: int

    metrics_file: str
    checkpoint_directory: str
    save_interval: int

@dataclass
class TrainSample:
    inputs: Tensor
    actions: Tensor
    rewards: Tensor
    next_inputs: Tensor
    terminated: Tensor

class ReplayBuffer:
    inputs: Tensor
    actions: Tensor
    rewards: Tensor
    next_inputs: Tensor
    terminated: Tensor

    index: int
    length: int
    max_length: int

    def __init__(self, input_size: int, max_length: int):
        self.inputs = torch.empty((max_length, input_size), dtype=torch.float32)
        self.actions = torch.empty(max_length, dtype=torch.int64)
        self.rewards = torch.empty(max_length, dtype=torch.float32)
        self.next_inputs = torch.empty((max_length, input_size), dtype=torch.float32)
        self.terminated = torch.empty(max_length, dtype=torch.bool)

        self.index = 0
        self.length = 0
        self.max_length = max_length

    def insert(
        self,
        input: Tensor,
        action: int,
        reward: float,
        next_input: Optional[Tensor],
    ):
        if DEBUG:
            assert self.index < self.max_length

        self.inputs[self.index] = input
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        if next_input is not None:
            self.next_inputs[self.index] = next_input
            self.terminated[self.index] = False
        else:
            self.next_inputs[self.index].zero_()
            self.terminated[self.index] = True

        self.index = (self.index + 1) % self.max_length
        if self.length < self.max_length:
            self.length += 1

    def sample(self, count: int) -> TrainSample:
        if DEBUG:
            assert count <= self.length
        indices = torch.randperm(self.length, dtype=torch.int64)[:count]
        return TrainSample(
            inputs=self.inputs[indices],
            actions=torch.unsqueeze(self.actions[indices], -1),
            rewards=torch.unsqueeze(self.rewards[indices], -1),
            next_inputs=self.next_inputs[indices],
            terminated=torch.unsqueeze(self.terminated[indices], -1),
        )

class DQNPolicy(BeliefPolicy):
    actions: list[Direction]
    input_size: int
    model: nn.Module

    def __init__(self, identifier: int):
        super().__init__(identifier)

        self.actions = list(Direction)
        if DEBUG:
            for a in range(len(self.actions)):
                assert self.actions[a].value == a
        self.input_size = 1 + 3 * (FEATURE_RADIUS + 1) * len(Direction)

        self.model = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.actions)),
        )

    def _create_input(self, features: Features) -> Tensor:
        input = [np.array(features.step / MAX_STEPS, dtype=np.float32)]
        for action in self.actions:
            input.append(features.actions[action].ravel())
        input = np.hstack(input, dtype=np.float32)
        return torch.from_numpy(input)

@dataclass
class UpdateMetrics:
    loss: float
    random_chance: float

class DQNTrainPolicy(DQNPolicy):
    params: TrainParams
    target_model: nn.Module
    replay_buffer: ReplayBuffer
    optimizer: Optimizer
    loss: nn.Module
    random_chance: float

    def __init__(self, identifier: int, params: TrainParams):
        super().__init__(identifier)
        self.params = params
        self.target_model = copy.deepcopy(self.model)
        self.model.train()
        self.target_model.eval()

        self.replay_buffer = ReplayBuffer(self.input_size, params.replay_size)
        self.optimizer = Adam(self.model.parameters(), lr=params.learning_rate)
        self.loss = nn.HuberLoss()
        self.random_chance = params.random_range[0]

    @torch.no_grad
    def _get_action(self, features: Features) -> Action:
        if random.random() < self.random_chance:
            return random.randint(0, len(Direction) - 1)
        input = self._create_input(features)
        action = torch.argmax(self.model(input))
        return self.actions[int(action)].value

    def _update(
        self,
        features: Features,
        action: Action,
        reward: Reward,
        next_features: Optional[Features],
    ):
        input = self._create_input(features)
        if next_features is not None:
            next_input = self._create_input(next_features)
            self.replay_buffer.insert(input, action, reward, next_input)
        else:
            self.replay_buffer.insert(input, action, reward, None)

    def train(self) -> UpdateMetrics:
        sample = self.replay_buffer.sample(self.params.sample_count)
        values = self.model(sample.inputs)
        action_values = torch.gather(values, 1, sample.actions)

        target_values = None
        with torch.no_grad():
            next_actions = torch.argmax(self.model(sample.next_inputs), dim=1, keepdim=True)
            next_values = self.target_model(sample.next_inputs)
            next_values = torch.gather(next_values, 1, next_actions)
            target_values = sample.rewards + self.params.discount_factor * next_values * (~sample.terminated)
            target_values = torch.clamp(target_values, max=self.params.max_value)
        if DEBUG:
            assert target_values is not None

        self.optimizer.zero_grad()
        loss = self.loss(action_values, target_values)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_gradient)
        self.optimizer.step()

        return UpdateMetrics(
            loss=float(loss.detach()),
            random_chance=self.random_chance,
        )

    def advance_episode(self):
        self.target_model.load_state_dict(self.model.state_dict())
        if self.random_chance > self.params.random_range[1]:
            self.random_chance *= self.params.random_decay
            self.random_chance = max(self.random_chance, self.params.random_range[1])

    def save_model(self, file_name: str):
        torch.save(self.model.state_dict(), file_name)

    def _reset(self):
        pass

@dataclass
class TrainMetrics:
    arenas: list[int]
    episodes: list[int]
    mean_losses: list[float]
    random_chances: list[float]
    rewards: list[float]

def create_arena(params: TrainParams, train_policy: DQNTrainPolicy) -> Arena:
    from arena.environment import Arena
    from arena.policies import REGISTRY

    def create_train_policy(identifier: int) -> DQNTrainPolicy:
        if DEBUG:
            assert identifier == train_policy.identifier
        return train_policy

    policies = list(filter(lambda p: p is not Custom, REGISTRY.values()))
    arena_policies = []
    for _ in range(params.num_opponents):
        arena_policies.append(random.choice(policies))
    arena_policies.append(create_train_policy)

    return Arena(
        height=params.arena_size[0],
        width=params.arena_size[1],
        policy_types=arena_policies,
        min_steps=params.arena_steps[0],
        max_steps=params.arena_steps[1],
        vision_radius=VISION_RADIUS,
    )

def train_arena(
    params: TrainParams,
    train_policy: DQNTrainPolicy,
    arena: Arena,
    arena_num: Optional[int],
    train_metrics: TrainMetrics,
    seed: int,
):
    for episode in range(1, params.train_episodes + 1):
        timestep = arena.reset(seed)
        seed += 1
        while not arena.is_done():
            timestep = arena.step()
        train_policy.reset()
        if arena_num is None:
            continue

        total_loss = 0.0
        random_chance = 0.0
        for _ in range(params.train_count):
            metrics = train_policy.train()
            total_loss += metrics.loss
            random_chance = metrics.random_chance
        train_policy.advance_episode()

        mean_loss = total_loss / params.train_count
        player_index = params.num_opponents
        if DEBUG:
            assert train_policy.identifier == arena.players[player_index].identifier
        reward = float(timestep.metadata.cumulative_returns[player_index])

        train_metrics.arenas.append(arena_num)
        train_metrics.episodes.append(episode)
        train_metrics.mean_losses.append(mean_loss)
        train_metrics.random_chances.append(random_chance)
        train_metrics.rewards.append(reward)

        print(f"[arena={arena_num}, episode={episode}]", end=" ")
        print(f"mean_loss={mean_loss}, random_chance={random_chance}, reward={reward}")

    train_policy.reset_strengths()
    if arena_num is None or arena_num % params.save_interval != 0:
        return
    with open(params.metrics_file, "w") as file:
        json.dump(dataclasses.asdict(train_metrics), file)
    train_policy.save_model(f"{params.checkpoint_directory}/model_{arena_num}.pt")

def train_dqn(params: TrainParams):
    from arena.utilities import logging

    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    logging.setup_logging("INFO")
    Path(params.metrics_file).parent.mkdir(parents=True, exist_ok=True)
    Path(params.checkpoint_directory).mkdir(parents=True, exist_ok=True)

    train_policy = DQNTrainPolicy(params.num_opponents + PLAYER_ID_OFFSET, params)
    seed_generator = Random(params.seed)
    arena_num = 0
    train_metrics = TrainMetrics(
        arenas=[],
        episodes=[],
        mean_losses=[],
        random_chances=[],
        rewards=[],
    )

    for _ in range(params.warmup_seeds):
        arena = create_arena(params, train_policy)
        seed = seed_generator.getrandbits(31)
        train_arena(params, train_policy, arena, None, train_metrics, seed)

    for _ in range(params.train_seeds):
        arena = create_arena(params, train_policy)
        arena_num += 1
        seed = seed_generator.getrandbits(31)
        train_arena(params, train_policy, arena, arena_num, train_metrics, seed)

class DQNInferPolicy(DQNPolicy):
    LOAD_FILE: str = "dqn/checkpoints/model_100.pt"

    def __init__(self, identifier: int):
        super().__init__(identifier)
        model_state = torch.load(self.LOAD_FILE)
        self.model.load_state_dict(model_state)
        self.model.eval()

    @property
    def name(self) -> str:
        return "Custom (DQN)"

    @torch.no_grad
    def _get_action(self, features: Features) -> Action:
        input = self._create_input(features)
        action = torch.argmax(self.model(input))
        return self.actions[int(action)].value

    def _update(
        self,
        features: Features,
        action: Action,
        reward: Reward,
        next_features: Optional[Features],
    ):
        pass

    def _reset(self):
        pass

class Custom(DQNInferPolicy):
    pass