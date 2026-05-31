from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm


RewardFn = Callable[[list[dict]], float]
DOMAIN_ORDER = ["정보·AI", "자연·공학", "진로·융합", "제2외국어·국제", "인문·사회", "예체능"]


@dataclass
class PolicyConfig:
    episodes: int = 300
    learning_rate: float = 0.003
    entropy_weight: float = 0.03
    value_weight: float = 0.5
    gamma: float = 0.95
    seed: int = 42
    hidden_dim: int = 64


class ActorCriticPolicy(nn.Module):
    def __init__(self, action_feature_dim: int, state_feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(action_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def action_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.actor(features).squeeze(-1)

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state).squeeze(-1)


def materialize_selection(selected: list[dict]) -> list[dict]:
    covered_pairs = set()
    out = []
    for item in selected:
        marginal = [x for x in item["covered"] if (x["school"], x["domain"]) not in covered_pairs]
        if not marginal:
            continue
        gain = sum(x["weight"] for x in marginal) - item.get("duplicate_ratio", 0) * 0.2
        enriched = {**item, "marginal": marginal, "gain": gain}
        out.append(enriched)
        covered_pairs |= {(x["school"], x["domain"]) for x in marginal}
    return out


def available_candidates(candidates: list[dict], selected: list[dict]) -> list[dict]:
    used = {(x["hub"], x["domain"]) for x in selected}
    covered = {
        (x["school"], x["domain"])
        for item in materialize_selection(selected)
        for x in item["marginal"]
    }
    out = []
    for cand in candidates:
        if (cand["hub"], cand["domain"]) in used:
            continue
        marginal = [x for x in cand["covered"] if (x["school"], x["domain"]) not in covered]
        if marginal:
            out.append({**cand, "marginal": marginal})
    return out


def candidate_features(cand: dict, weak_count: int, shortage_count: int, radius_km: float) -> np.ndarray:
    marginal = cand.get("marginal") or cand["covered"]
    weights = np.array([x["weight"] for x in marginal], dtype=float)
    distances = np.array([x["distance_km"] for x in marginal], dtype=float)
    sai_values = np.array([x.get("sai_before", 100.0) for x in marginal], dtype=float)
    schools = {x["school"] for x in marginal}
    domain = cand["domain"]
    base = [
        float(weights.sum() / max(shortage_count, 1)),
        float(len(marginal) / max(shortage_count, 1)),
        float(len(schools) / max(weak_count, 1)),
        float(max(0.0, 1.0 - distances.mean() / max(radius_km, 0.001))),
        float(1.0 - cand.get("duplicate_ratio", 0.0)),
        float(weights.mean() if len(weights) else 0.0),
        float((100.0 - sai_values.min()) / 100.0),
        float((100.0 - sai_values.mean()) / 100.0),
        float(np.quantile((100.0 - sai_values) / 100.0, 0.75)),
        float((sai_values <= np.quantile(sai_values, 0.25)).mean()),
        float(distances.max() / max(radius_km, 0.001)),
    ]
    domain_one_hot = [1.0 if domain == d else 0.0 for d in DOMAIN_ORDER]
    return np.array(base + domain_one_hot, dtype=np.float32)


def state_features(selected: list[dict], budget: int, shortage_count: int, radius_km: float, current_score: float) -> np.ndarray:
    materialized = materialize_selection(selected)
    covered = {(x["school"], x["domain"]) for item in materialized for x in item["marginal"]}
    distances = [x["distance_km"] for item in materialized for x in item["marginal"]]
    hubs = {item["hub"] for item in materialized}
    domain_counts = {domain: 0 for domain in DOMAIN_ORDER}
    for item in materialized:
        domain_counts[item["domain"]] += 1
    base = [
        len(materialized) / max(budget, 1),
        len(covered) / max(shortage_count, 1),
        current_score / 100.0,
        (np.mean(distances) / max(radius_km, 0.001)) if distances else 0.0,
        len(hubs) / max(len(materialized), 1),
    ]
    return np.array(base + [domain_counts[d] / max(budget, 1) for d in DOMAIN_ORDER], dtype=np.float32)


def action_feature_matrix(
    candidates: list[dict],
    state: np.ndarray,
    weak_count: int,
    shortage_count: int,
    radius_km: float,
) -> torch.Tensor:
    return torch.tensor(
        np.vstack([
            np.concatenate([candidate_features(cand, weak_count, shortage_count, radius_km), state])
            for cand in candidates
        ]),
        dtype=torch.float32,
    )


def sample_episode(
    candidates: list[dict],
    model: ActorCriticPolicy,
    reward_fn: RewardFn,
    budget: int,
    weak_count: int,
    shortage_count: int,
    radius_km: float,
):
    selected, log_probs, entropies, values, rewards = [], [], [], [], []
    current_score = 0.0
    for _ in range(budget):
        available = available_candidates(candidates, selected)
        if not available:
            break
        state = state_features(selected, budget, shortage_count, radius_km, current_score)
        features = action_feature_matrix(available, state, weak_count, shortage_count, radius_km)
        dist = Categorical(logits=model.action_logits(features))
        action = dist.sample()
        selected.append(available[int(action.item())])
        next_score = reward_fn(materialize_selection(selected))
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())
        values.append(model.value(torch.tensor(state, dtype=torch.float32)))
        rewards.append(next_score - current_score)
        current_score = next_score
    return materialize_selection(selected), log_probs, entropies, values, rewards


def deterministic_episode(
    candidates: list[dict],
    model: ActorCriticPolicy,
    reward_fn: RewardFn,
    budget: int,
    weak_count: int,
    shortage_count: int,
    radius_km: float,
) -> list[dict]:
    selected = []
    current_score = 0.0
    with torch.no_grad():
        for _ in range(budget):
            available = available_candidates(candidates, selected)
            if not available:
                break
            state = state_features(selected, budget, shortage_count, radius_km, current_score)
            features = action_feature_matrix(available, state, weak_count, shortage_count, radius_km)
            selected.append(available[int(torch.argmax(model.action_logits(features)).item())])
            current_score = reward_fn(materialize_selection(selected))
    return materialize_selection(selected)


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    out, total = [], 0.0
    for reward in reversed(rewards):
        total = reward + gamma * total
        out.append(total)
    return torch.tensor(list(reversed(out)), dtype=torch.float32)


def train_policy(candidates: list[dict], reward_fn: RewardFn, budget: int, weak_count: int, shortage_count: int, radius_km: float, config: PolicyConfig):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    candidate_dim = 11 + len(DOMAIN_ORDER)
    state_dim = 5 + len(DOMAIN_ORDER)
    model = ActorCriticPolicy(candidate_dim + state_dim, state_dim, config.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_reward, best_selected, logs = -np.inf, [], []
    iterator = tqdm(range(1, config.episodes + 1), desc="torch-actor-critic-assignment", unit="episode")
    for episode in iterator:
        selected, log_probs, entropies, values, rewards = sample_episode(
            candidates, model, reward_fn, budget, weak_count, shortage_count, radius_km
        )
        reward = reward_fn(selected)
        if reward > best_reward:
            best_reward, best_selected = reward, selected
        returns = discounted_returns(rewards, config.gamma) if rewards else torch.tensor([])
        value_tensor = torch.stack(values) if values else torch.tensor([])
        advantages = returns - value_tensor.detach() if len(rewards) else torch.tensor([])
        policy_loss = -(torch.stack(log_probs) * advantages).sum() if log_probs else torch.tensor(0.0)
        value_loss = ((value_tensor - returns) ** 2).mean() if len(rewards) else torch.tensor(0.0)
        entropy_bonus = torch.stack(entropies).sum() if entropies else torch.tensor(0.0)
        loss = policy_loss + config.value_weight * value_loss - config.entropy_weight * entropy_bonus
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logs.append({
            "episode": episode,
            "reward": reward,
            "step_reward_sum": sum(rewards),
            "value_loss": float(value_loss.detach().item()),
            "mean_advantage": float(advantages.mean().detach().item()) if len(rewards) else 0.0,
            "loss": float(loss.detach().item()),
            "selected_count": len(selected),
            "best_reward": best_reward,
        })
        if episode % 10 == 0:
            iterator.set_postfix(reward=round(reward, 3), best=round(best_reward, 3))
    deterministic = deterministic_episode(candidates, model, reward_fn, budget, weak_count, shortage_count, radius_km)
    deterministic_reward = reward_fn(deterministic)
    if deterministic_reward >= best_reward:
        best_selected, best_reward = deterministic, deterministic_reward
    logs.append({"episode": "deterministic_policy", "reward": deterministic_reward, "step_reward_sum": None, "value_loss": None, "mean_advantage": None, "loss": None, "selected_count": len(deterministic), "best_reward": best_reward})
    return best_selected, pd.DataFrame(logs), model
