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
    seed: int = 42
    hidden_dim: int = 64
    baseline_decay: float = 0.9


class AssignmentPolicy(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def materialize_selection(selected: list[dict]) -> list[dict]:
    covered_pairs = set()
    out = []
    for item in selected:
        marginal = [x for x in item["covered"] if (x["school"], x.get("subject", x["domain"])) not in covered_pairs]
        if not marginal:
            continue
        gain = sum(x["weight"] for x in marginal) - item.get("duplicate_ratio", 0) * 0.2
        enriched = {**item, "marginal": marginal, "gain": gain}
        out.append(enriched)
        covered_pairs |= {(x["school"], x.get("subject", x["domain"])) for x in marginal}
    return out


def available_candidates(candidates: list[dict], selected: list[dict]) -> list[dict]:
    used = {(x["hub"], x.get("subject", x["domain"])) for x in selected}
    covered = {
        (x["school"], x.get("subject", x["domain"]))
        for item in materialize_selection(selected)
        for x in item["marginal"]
    }
    out = []
    for cand in candidates:
        if (cand["hub"], cand.get("subject", cand["domain"])) in used:
            continue
        marginal = [x for x in cand["covered"] if (x["school"], x.get("subject", x["domain"])) not in covered]
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


def feature_matrix(candidates: list[dict], weak_count: int, shortage_count: int, radius_km: float) -> torch.Tensor:
    return torch.tensor(
        np.vstack([candidate_features(c, weak_count, shortage_count, radius_km) for c in candidates]),
        dtype=torch.float32,
    )


def sample_episode(
    candidates: list[dict],
    model: AssignmentPolicy,
    budget: int,
    weak_count: int,
    shortage_count: int,
    radius_km: float,
) -> tuple[list[dict], list[torch.Tensor], list[torch.Tensor]]:
    selected = []
    log_probs = []
    entropies = []
    for _ in range(budget):
        available = available_candidates(candidates, selected)
        if not available:
            break
        features = feature_matrix(available, weak_count, shortage_count, radius_km)
        dist = Categorical(logits=model(features))
        action = dist.sample()
        selected.append(available[int(action.item())])
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())
    return materialize_selection(selected), log_probs, entropies


def deterministic_episode(
    candidates: list[dict],
    model: AssignmentPolicy,
    budget: int,
    weak_count: int,
    shortage_count: int,
    radius_km: float,
) -> list[dict]:
    selected = []
    with torch.no_grad():
        for _ in range(budget):
            available = available_candidates(candidates, selected)
            if not available:
                break
            features = feature_matrix(available, weak_count, shortage_count, radius_km)
            action = int(torch.argmax(model(features)).item())
            selected.append(available[action])
    return materialize_selection(selected)


def train_policy(
    candidates: list[dict],
    reward_fn: RewardFn,
    budget: int,
    weak_count: int,
    shortage_count: int,
    radius_km: float,
    config: PolicyConfig,
) -> tuple[list[dict], pd.DataFrame, AssignmentPolicy]:
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    feature_dim = 11 + len(DOMAIN_ORDER)
    model = AssignmentPolicy(feature_dim, config.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    baseline = 0.0
    best_reward = -np.inf
    best_selected = []
    logs = []

    iterator = tqdm(range(1, config.episodes + 1), desc="torch-rl-assignment", unit="episode")
    for episode in iterator:
        selected, log_probs, entropies = sample_episode(
            candidates, model, budget, weak_count, shortage_count, radius_km
        )
        reward = reward_fn(selected)
        if reward > best_reward:
            best_reward = reward
            best_selected = selected
        baseline = reward if episode == 1 else config.baseline_decay * baseline + (1 - config.baseline_decay) * reward
        advantage = reward - baseline

        policy_loss = -torch.stack(log_probs).sum() * float(advantage) if log_probs else torch.tensor(0.0)
        entropy_bonus = torch.stack(entropies).sum() if entropies else torch.tensor(0.0)
        loss = policy_loss - config.entropy_weight * entropy_bonus
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logs.append({
            "episode": episode,
            "reward": reward,
            "baseline": baseline,
            "advantage": advantage,
            "loss": float(loss.detach().item()),
            "selected_count": len(selected),
            "best_reward": best_reward,
        })
        if episode % 10 == 0:
            iterator.set_postfix(reward=round(reward, 3), best=round(best_reward, 3))

    deterministic = deterministic_episode(candidates, model, budget, weak_count, shortage_count, radius_km)
    deterministic_reward = reward_fn(deterministic)
    if deterministic_reward >= best_reward:
        best_selected = deterministic
        best_reward = deterministic_reward
    logs.append({
        "episode": "deterministic_policy",
        "reward": deterministic_reward,
        "baseline": baseline,
        "advantage": deterministic_reward - baseline,
        "loss": None,
        "selected_count": len(deterministic),
        "best_reward": best_reward,
    })
    return best_selected, pd.DataFrame(logs), model
