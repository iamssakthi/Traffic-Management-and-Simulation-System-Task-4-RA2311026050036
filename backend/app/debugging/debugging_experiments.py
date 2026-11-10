"""
Controlled Debugging Experiment Runner (Task 4).
Executes reproducible comparative trials between Faulty and Corrected RL agents
under strictly identical seeds, scenarios, and step budgets.
Calculates statistical improvements and stores reproducible telemetry.
"""
from typing import Dict, Any, List, Optional
import time
import numpy as np

from app.simulation.environment import TrafficEnvironment
from app.training.baseline import FixedTimeController
from app.debugging.faulty_agent import FaultyQLearningAgent
from app.debugging.corrected_agent import CorrectedQLearningAgent
from app.debugging.faulty_dqn import FaultyDQNAgent
from app.debugging.corrected_dqn import CorrectedDQNAgent

class DebuggingExperimentRunner:
    def __init__(self):
        self.cached_results: Dict[str, Any] = {}

    def run_tabular_comparison(
        self,
        scenario_id: str = "heavy_north_south",
        episodes: int = 15,
        episode_duration: float = 120.0,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Runs Faulty vs Corrected Tabular Q-Learning under identical seeds and traffic.
        """
        # 1. Faulty Q-Learning Run
        faulty_env = TrafficEnvironment(scenario_id=scenario_id, episode_duration=episode_duration, seed=seed)
        faulty_agent = FaultyQLearningAgent(
            seed=seed,
            bug_terminal_bootstrap=True,
            bug_premature_decay=True,
            bug_inverted_reward=False,
        )

        faulty_history = []
        t0 = time.time()
        for ep in range(episodes):
            disc_s, _ = faulty_env.reset(seed=seed + ep)
            ep_reward = 0.0
            done = False
            while not done:
                action = faulty_agent.select_action(disc_s)
                next_disc_s, _, r, done, info = faulty_env.step(action)
                faulty_agent.update(disc_s, action, r, next_disc_s, done)
                disc_s = next_disc_s
                ep_reward += r
            faulty_agent.decay_epsilon()
            metrics = faulty_env.get_summary_metrics()
            faulty_history.append({
                "episode": ep + 1,
                "reward": round(ep_reward, 2),
                "waiting_time": metrics["mean_waiting_time"],
                "queue_length": metrics["mean_queue_length"],
                "throughput": metrics["total_departures"],
                "epsilon": round(float(faulty_agent.epsilon), 3),
                "mean_td_error": round(float(np.mean(faulty_agent.td_error_history[-50:] or [0.0])), 3),
            })
        faulty_duration = time.time() - t0

        # 2. Corrected Q-Learning Run
        corr_env = TrafficEnvironment(scenario_id=scenario_id, episode_duration=episode_duration, seed=seed)
        corr_agent = CorrectedQLearningAgent(seed=seed)

        corr_history = []
        t1 = time.time()
        for ep in range(episodes):
            disc_s, _ = corr_env.reset(seed=seed + ep)
            ep_reward = 0.0
            done = False
            while not done:
                action = corr_agent.select_action(disc_s)
                next_disc_s, _, r, done, info = corr_env.step(action)
                corr_agent.update(disc_s, action, r, next_disc_s, done)
                disc_s = next_disc_s
                ep_reward += r
            corr_agent.decay_epsilon()
            metrics = corr_env.get_summary_metrics()
            corr_history.append({
                "episode": ep + 1,
                "reward": round(ep_reward, 2),
                "waiting_time": metrics["mean_waiting_time"],
                "queue_length": metrics["mean_queue_length"],
                "throughput": metrics["total_departures"],
                "epsilon": round(float(corr_agent.epsilon), 3),
                "mean_td_error": round(float(np.mean(corr_agent.td_error_history[-50:] or [0.0])), 3),
            })
        corr_duration = time.time() - t1

        # Summary averages
        f_wait = float(np.mean([h["waiting_time"] for h in faulty_history]))
        c_wait = float(np.mean([h["waiting_time"] for h in corr_history]))
        f_queue = float(np.mean([h["queue_length"] for h in faulty_history]))
        c_queue = float(np.mean([h["queue_length"] for h in corr_history]))
        f_thru = float(np.mean([h["throughput"] for h in faulty_history]))
        c_thru = float(np.mean([h["throughput"] for h in corr_history]))
        f_rew = float(np.mean([h["reward"] for h in faulty_history]))
        c_rew = float(np.mean([h["reward"] for h in corr_history]))

        # Improvements: lower-is-better: (F - C)/F * 100, higher-is-better: (C - F)/|F| * 100
        wait_imp = ((f_wait - c_wait) / max(0.01, f_wait)) * 100.0
        queue_imp = ((f_queue - c_queue) / max(0.01, f_queue)) * 100.0
        thru_imp = ((c_thru - f_thru) / max(0.01, f_thru)) * 100.0
        rew_imp = ((c_rew - f_rew) / max(0.01, abs(f_rew))) * 100.0

        result = {
            "algorithm": "Tabular Q-Learning",
            "scenario": scenario_id,
            "episodes": episodes,
            "seed": seed,
            "faulty": {
                "name": "Faulty Q-Learning (Terminal Bootstrap + Epsilon Collapse)",
                "mean_waiting_time": round(f_wait, 2),
                "mean_queue_length": round(f_queue, 2),
                "mean_throughput": round(f_thru, 2),
                "mean_reward": round(f_rew, 2),
                "training_duration_s": round(faulty_duration, 2),
                "final_epsilon": round(float(faulty_agent.epsilon), 3),
                "history": faulty_history,
                "bugs_active": ["BUG-ALG-01", "BUG-ALG-03"],
            },
            "corrected": {
                "name": "Corrected Q-Learning (Verified TD Update & Calibrated Decay)",
                "mean_waiting_time": round(c_wait, 2),
                "mean_queue_length": round(c_queue, 2),
                "mean_throughput": round(c_thru, 2),
                "mean_reward": round(c_rew, 2),
                "training_duration_s": round(corr_duration, 2),
                "final_epsilon": round(float(corr_agent.epsilon), 3),
                "history": corr_history,
                "bugs_resolved": ["BUG-ALG-01", "BUG-ALG-03"],
            },
            "improvements": {
                "waiting_time_percent": round(wait_imp, 2),
                "queue_length_percent": round(queue_imp, 2),
                "throughput_percent": round(thru_imp, 2),
                "reward_percent": round(rew_imp, 2),
            },
            "verdict": (
                f"Corrected Q-Learning achieved a {wait_imp:.1f}% reduction in average vehicular delay and "
                f"a {queue_imp:.1f}% reduction in queue accumulation over the faulty implementation."
            )
        }

        self.cached_results["q_learning"] = result
        return result

    def run_dqn_comparison(
        self,
        scenario_id: str = "balanced",
        episodes: int = 8,
        episode_duration: float = 100.0,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Runs Faulty vs Corrected Deep Q-Network (DQN) under identical seeds.
        """
        # 1. Faulty DQN Run
        faulty_env = TrafficEnvironment(scenario_id=scenario_id, episode_duration=episode_duration, seed=seed)
        faulty_agent = FaultyDQNAgent(
            seed=seed,
            batch_size=32,
            bug_leak_gradients=True,
            bug_missing_zero_grad=True,
            bug_terminal_bootstrap=True,
        )

        faulty_history = []
        t0 = time.time()
        for ep in range(episodes):
            _, cont_s = faulty_env.reset(seed=seed + ep)
            ep_reward = 0.0
            done = False
            losses = []
            while not done:
                action = faulty_agent.select_action(cont_s)
                _, next_cont_s, r, done, _ = faulty_env.step(action)
                loss = faulty_agent.update(cont_s, action, r, next_cont_s, done)
                if loss is not None:
                    losses.append(loss)
                cont_s = next_cont_s
                ep_reward += r
            faulty_agent.decay_epsilon()
            metrics = faulty_env.get_summary_metrics()
            faulty_history.append({
                "episode": ep + 1,
                "reward": round(ep_reward, 2),
                "waiting_time": metrics["mean_waiting_time"],
                "queue_length": metrics["mean_queue_length"],
                "throughput": metrics["total_departures"],
                "loss": round(float(np.mean(losses or [0.0])), 4),
                "gradient_norm": round(float(np.mean(faulty_agent.gradient_norm_history[-20:] or [0.0])), 3),
            })
        faulty_duration = time.time() - t0

        # 2. Corrected DQN Run
        corr_env = TrafficEnvironment(scenario_id=scenario_id, episode_duration=episode_duration, seed=seed)
        corr_agent = CorrectedDQNAgent(seed=seed, batch_size=32)

        corr_history = []
        t1 = time.time()
        for ep in range(episodes):
            _, cont_s = corr_env.reset(seed=seed + ep)
            ep_reward = 0.0
            done = False
            losses = []
            while not done:
                action = corr_agent.select_action(cont_s)
                _, next_cont_s, r, done, _ = corr_env.step(action)
                loss = corr_agent.update(cont_s, action, r, next_cont_s, done)
                if loss is not None:
                    losses.append(loss)
                cont_s = next_cont_s
                ep_reward += r
            corr_agent.decay_epsilon()
            metrics = corr_env.get_summary_metrics()
            corr_history.append({
                "episode": ep + 1,
                "reward": round(ep_reward, 2),
                "waiting_time": metrics["mean_waiting_time"],
                "queue_length": metrics["mean_queue_length"],
                "throughput": metrics["total_departures"],
                "loss": round(float(np.mean(losses or [0.0])), 4),
                "gradient_norm": round(float(np.mean(corr_agent.gradient_norm_history[-20:] or [0.0])), 3),
            })
        corr_duration = time.time() - t1

        f_wait = float(np.mean([h["waiting_time"] for h in faulty_history]))
        c_wait = float(np.mean([h["waiting_time"] for h in corr_history]))
        f_queue = float(np.mean([h["queue_length"] for h in faulty_history]))
        c_queue = float(np.mean([h["queue_length"] for h in corr_history]))
        f_thru = float(np.mean([h["throughput"] for h in faulty_history]))
        c_thru = float(np.mean([h["throughput"] for h in corr_history]))
        f_rew = float(np.mean([h["reward"] for h in faulty_history]))
        c_rew = float(np.mean([h["reward"] for h in corr_history]))

        wait_imp = ((f_wait - c_wait) / max(0.01, f_wait)) * 100.0
        queue_imp = ((f_queue - c_queue) / max(0.01, f_queue)) * 100.0
        thru_imp = ((c_thru - f_thru) / max(0.01, f_thru)) * 100.0
        rew_imp = ((c_rew - f_rew) / max(0.01, abs(f_rew))) * 100.0

        result = {
            "algorithm": "Deep Q-Network (DQN)",
            "scenario": scenario_id,
            "episodes": episodes,
            "seed": seed,
            "faulty": {
                "name": "Faulty DQN (Target Leak + Accumulating Gradients + MSE)",
                "mean_waiting_time": round(f_wait, 2),
                "mean_queue_length": round(f_queue, 2),
                "mean_throughput": round(f_thru, 2),
                "mean_reward": round(f_rew, 2),
                "training_duration_s": round(faulty_duration, 2),
                "history": faulty_history,
                "bugs_active": ["BUG-IMP-01", "BUG-IMP-02", "BUG-ALG-01"],
            },
            "corrected": {
                "name": "Corrected DQN (Detached Targets + Huber Loss + Norm Clipping)",
                "mean_waiting_time": round(c_wait, 2),
                "mean_queue_length": round(c_queue, 2),
                "mean_throughput": round(c_thru, 2),
                "mean_reward": round(c_rew, 2),
                "training_duration_s": round(corr_duration, 2),
                "history": corr_history,
                "bugs_resolved": ["BUG-IMP-01", "BUG-IMP-02", "BUG-ALG-01"],
            },
            "improvements": {
                "waiting_time_percent": round(wait_imp, 2),
                "queue_length_percent": round(queue_imp, 2),
                "throughput_percent": round(thru_imp, 2),
                "reward_percent": round(rew_imp, 2),
            },
            "verdict": (
                f"Corrected DQN achieved a {wait_imp:.1f}% reduction in average vehicular delay and "
                f"a {queue_imp:.1f}% reduction in queues, resolving gradient explosion issues."
            )
        }

        self.cached_results["dqn"] = result
        return result

