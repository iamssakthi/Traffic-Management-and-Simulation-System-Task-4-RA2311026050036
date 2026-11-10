"""
Formal Bug Catalog & Taxonomy for RL Debugging Challenge (Task 4).
Classifies algorithmic, implementation, environment/state, and training defects
with exact code diffs, root-cause analyses, and mathematical impacts.
"""
from typing import Dict, Any, List

BUG_CATALOG: Dict[str, Dict[str, Any]] = {
    "BUG-ALG-01": {
        "id": "BUG-ALG-01",
        "name": "Terminal State Bootstrap Violation",
        "category": "Algorithmic Error",
        "subcategory": "Bellman Target Formulation",
        "affected_algorithms": ["Tabular Q-Learning", "Deep Q-Network (DQN)"],
        "severity": "CRITICAL",
        "file_name": "faulty_agent.py / faulty_dqn.py",
        "function_name": "update()",
        "line_number": 65,
        "description": "Failing to mask the future discounted reward when the transition is terminal (done = True).",
        "root_cause": "The temporal difference target calculation omits the (1.0 - done) binary mask: target = reward + gamma * max_a' Q(s', a').",
        "mathematical_impact": (
            "Violates the fundamental Bellman boundary condition V(s_terminal) = 0. "
            "Injects phantom future value into terminal states, causing severe overestimation of state-action utilities "
            "and corrupting the return calculation G_t."
        ),
        "expected_behavior": "When done == True, TD Target must strictly equal the immediate reward r: target = reward.",
        "actual_behavior": "When done == True, TD Target bootstraps next-episode state values: target = reward + gamma * max Q(s', a').",
        "faulty_code": (
            "# FAULTY IMPLEMENTATION:\n"
            "# Missing terminal state condition\n"
            "target = reward + self.gamma * np.max(self.q_table[next_state])\n"
            "td_error = target - self.q_table[state, action]\n"
            "self.q_table[state, action] += self.alpha * td_error"
        ),
        "corrected_code": (
            "# CORRECTED IMPLEMENTATION:\n"
            "# Strictly zero out bootstrap when transition terminates episode\n"
            "if done:\n"
            "    target = reward\n"
            "else:\n"
            "    target = reward + self.gamma * np.max(self.q_table[next_state])\n"
            "td_error = target - self.q_table[state, action]\n"
            "self.q_table[state, action] += self.alpha * td_error"
        ),
        "validation_metric": "TD Error stability & Q-value convergence at episode boundaries."
    },
    "BUG-ALG-02": {
        "id": "BUG-ALG-02",
        "name": "Inverted Congestion Reward Sign",
        "category": "Algorithmic Error",
        "subcategory": "Reward Function Shaping",
        "affected_algorithms": ["Tabular Q-Learning", "Deep Q-Network (DQN)"],
        "severity": "CRITICAL",
        "file_name": "faulty_trainer.py / environment.py",
        "function_name": "calculate_reward()",
        "line_number": 272,
        "description": "The sign of vehicle queue and waiting delay penalties is accidentally positive instead of negative.",
        "root_cause": "Reward formula implements R = + [wq * Q + ww * dW] instead of R = - [wq * Q + ww * dW].",
        "mathematical_impact": (
            "Inverts the objective function: the agent seeks to maximize queue lengths and prolong waiting times, "
            "causing catastrophic gridlock, vehicle starvation, and throughput collapse."
        ),
        "expected_behavior": "Queue lengths and accumulated waiting times must decrease reward (negative penalty).",
        "actual_behavior": "Accumulated queues and delays increase scalar reward, incentivizing intersection deadlock.",
        "faulty_code": (
            "# FAULTY IMPLEMENTATION:\n"
            "# Inverted penalty signs incentivize queues\n"
            "reward = (\n"
            "    self.queue_weight * (total_queue / 10.0) +\n"
            "    self.waiting_weight * (wait_increment / 10.0)\n"
            ") + self.throughput_weight * departures"
        ),
        "corrected_code": (
            "# CORRECTED IMPLEMENTATION:\n"
            "# Congestion must be penalized (negative cost)\n"
            "reward = -(\n"
            "    self.queue_weight * (total_queue / 10.0) +\n"
            "    self.waiting_weight * (wait_increment / 10.0) +\n"
            "    self.switch_weight * switch_penalty\n"
            ") + self.throughput_weight * departures"
        ),
        "validation_metric": "Mean Queue Length (< 5 veh) and Waiting Time (< 20s)."
    },
    "BUG-ALG-03": {
        "id": "BUG-ALG-03",
        "name": "Premature Epsilon Collapse",
        "category": "Training Error",
        "subcategory": "Exploration-Exploitation Scheduling",
        "affected_algorithms": ["Tabular Q-Learning", "Deep Q-Network (DQN)"],
        "severity": "HIGH",
        "file_name": "faulty_agent.py",
        "function_name": "decay_epsilon()",
        "line_number": 75,
        "description": "Exploration rate epsilon decays excessively fast with an epsilon_min of 0.0.",
        "root_cause": "decay_rate set to 0.70 instead of 0.995, and epsilon_min set to 0.0, causing exploration to vanish in under 10 episodes.",
        "mathematical_impact": (
            "The agent stops exploring unvisited state-action pairs almost immediately. It becomes trapped in "
            "a myopic local optimum (such as always choosing KEEP and never switching phases), leading to infinite queues on opposing approaches."
        ),
        "expected_behavior": "Gradual exponential decay over hundreds of episodes with a non-zero floor (epsilon_min = 0.05).",
        "actual_behavior": "Epsilon drops from 1.0 to 0.02 by Episode 8, locking the policy permanently.",
        "faulty_code": (
            "# FAULTY IMPLEMENTATION:\n"
            "# Aggressive decay and zero floor\n"
            "self.epsilon = self.epsilon * 0.70  # Rapid collapse\n"
            "if self.epsilon < 0.001:\n"
            "    self.epsilon = 0.0"
        ),
        "corrected_code": (
            "# CORRECTED IMPLEMENTATION:\n"
            "# Controlled decay with non-zero minimum floor\n"
            "self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)\n"
            "# default: epsilon_decay = 0.995, epsilon_min = 0.05"
        ),
        "validation_metric": "Exploration trajectory across 50 episodes and state-space coverage."
    },
    "BUG-IMP-01": {
        "id": "BUG-IMP-01",
        "name": "Target Network Gradient Leakage",
        "category": "Implementation Error",
        "subcategory": "Autograd & Computational Graph",
        "affected_algorithms": ["Deep Q-Network (DQN)"],
        "severity": "CRITICAL",
        "file_name": "faulty_dqn.py",
        "function_name": "update()",
        "line_number": 112,
        "description": "Failing to detach target network outputs or omitting torch.no_grad() when calculating Q-target.",
        "root_cause": "The target network output tensor remains attached to PyTorch autograd graph: q_next = self.target_net(b_next_states).max(1)[0].",
        "mathematical_impact": (
            "Gradients flow back into the target network during loss.backward(). This turns DQN's semi-gradient temporal difference "
            "method into an invalid dual-network optimization, causing loss divergence and numerical instability."
        ),
        "expected_behavior": "Target values are treated as fixed regression labels with zero autograd gradients.",
        "actual_behavior": "Gradients update both online and target network parameters simultaneously.",
        "faulty_code": (
            "# FAULTY IMPLEMENTATION:\n"
            "# Target tensor remains attached to autograd graph\n"
            "q_next = self.target_net(b_next_states).max(1)[0].unsqueeze(1)\n"
            "q_target = b_rewards + (1.0 - b_dones) * self.gamma * q_next\n"
            "loss = self.criterion(q_eval, q_target)  # Target has requires_grad=True!"
        ),
        "corrected_code": (
            "# CORRECTED IMPLEMENTATION:\n"
            "# Explicit torch.no_grad() context and detached tensor\n"
            "with torch.no_grad():\n"
            "    q_next = self.target_net(b_next_states).max(1)[0].unsqueeze(1)\n"
            "    q_target = b_rewards + (1.0 - b_dones) * self.gamma * q_next\n"
            "loss = self.criterion(q_eval, q_target)"
        ),
        "validation_metric": "Target network requires_grad flag and loss curve convergence."
    },
    "BUG-IMP-02": {
        "id": "BUG-IMP-02",
        "name": "Missing Optimizer Gradient Zeroing",
        "category": "Implementation Error",
        "subcategory": "PyTorch Optimization Pipeline",
        "affected_algorithms": ["Deep Q-Network (DQN)"],
        "severity": "CRITICAL",
        "file_name": "faulty_dqn.py",
        "function_name": "update()",
        "line_number": 120,
        "description": "Omitting optimizer.zero_grad() before loss.backward() in the training loop.",
        "root_cause": "PyTorch accumulates gradients by default. Without zeroing, gradients from all previous batches sum up indefinitely.",
        "mathematical_impact": (
            "Gradient magnitudes grow monotonically with step count. Effective parameter updates explode, "
            "causing weights to blow up to NaN/Inf and destroying neural representations."
        ),
        "expected_behavior": "Gradients are calculated strictly for the current mini-batch samples.",
        "actual_behavior": "Gradients continuously accumulate across every past iteration.",
        "faulty_code": (
            "# FAULTY IMPLEMENTATION:\n"
            "# Missing optimizer.zero_grad() causes unbounded gradient accumulation\n"
            "loss.backward()\n"
            "self.optimizer.step()  # Step size grows uncontrollably!"
        ),
        "corrected_code": (
            "# CORRECTED IMPLEMENTATION:\n"
            "# Always clear prior gradients before backward propagation\n"
            "self.optimizer.zero_grad()\n"
            "loss.backward()\n"
            "torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)\n"
            "self.optimizer.step()"
        ),
        "validation_metric": "Policy network gradient norms and absence of NaN weights."
    },
    "BUG-IMP-03": {
        "id": "BUG-IMP-03",
        "name": "Replay Buffer Off-By-One State Misalignment",
        "category": "Implementation Error",
        "subcategory": "Memory Buffer Sampling",
        "affected_algorithms": ["Deep Q-Network (DQN)"],
        "severity": "HIGH",
        "file_name": "faulty_replay_buffer.py",
        "function_name": "sample()",
        "line_number": 32,
        "description": "Mini-batch sampling pairs state S_t with the wrong subsequent state S_{t+2} due to index shift.",
        "root_cause": "Indices for next_states are sampled with an off-by-one slice: next_states[i+1] instead of next_states[i].",
        "mathematical_impact": (
            "Destroys the Markov decision process transition relation (s, a, r, s'). The agent attempts to associate "
            "an action with an unrelated downstream state, preventing value function convergence."
        ),
        "expected_behavior": "Each tuple (s, a, r, s', done) strictly pairs the observation with its immediate consequence.",
        "actual_behavior": "s is paired with s' from an offset future timestep or different vehicle condition.",
        "faulty_code": (
            "# FAULTY IMPLEMENTATION:\n"
            "# Off-by-one misalignment corrupts transition tuples\n"
            "states = [b[0] for b in batch]\n"
            "next_states = [self.buffer[(idx + 1) % len(self.buffer)][3] for idx in sampled_indices]"
        ),
        "corrected_code": (
            "# CORRECTED IMPLEMENTATION:\n"
            "# Atomic transition tuple unpacking guarantees perfect alignment\n"
            "batch = self.rng.sample(self.buffer, batch_size)\n"
            "states, actions, rewards, next_states, dones = zip(*batch)"
        ),
        "validation_metric": "Transition tuple consistency tests and TD error magnitude."
    },
    "BUG-ALG-04": {
        "id": "BUG-ALG-04",
        "name": "Frozen Target Network Synchronization",
        "category": "Algorithmic Error",
        "subcategory": "Target Network Stability",
        "affected_algorithms": ["Deep Q-Network (DQN)"],
        "severity": "HIGH",
        "file_name": "faulty_dqn.py",
        "function_name": "update()",
        "line_number": 130,
        "description": "Target network weights are never synchronized with the policy network.",
        "root_cause": "The update condition target_update_freq is never met or the sync statement is commented out.",
        "mathematical_impact": (
            "The target network remains frozen at its random initialization. The online network trains against "
            "meaningless static random targets, causing policy divergence and complete failure to learn traffic dynamics."
        ),
        "expected_behavior": "Target network periodically copies policy network weights every C=100 steps.",
        "actual_behavior": "Target network retains random weight initialization forever.",
        "faulty_code": (
            "# FAULTY IMPLEMENTATION:\n"
            "# Synchronization statement omitted\n"
            "if self.train_step_count % self.target_update_freq == 0:\n"
            "    pass  # Forgotten target network copy!"
        ),
        "corrected_code": (
            "# CORRECTED IMPLEMENTATION:\n"
            "# Periodic full parameter transfer\n"
            "if self.train_step_count % self.target_update_freq == 0:\n"
            "    self.target_net.load_state_dict(self.policy_net.state_dict())"
        ),
        "validation_metric": "Target network parameter distance ||theta_target - theta_policy||."
    },
    "BUG-ENV-01": {
        "id": "BUG-ENV-01",
        "name": "Unnormalized State Feature Disparity",
        "category": "Environment / State Error",
        "subcategory": "State Representation",
        "affected_algorithms": ["Deep Q-Network (DQN)"],
        "severity": "MEDIUM",
        "file_name": "faulty_trainer.py",
        "function_name": "get_continuous_state()",
        "line_number": 185,
        "description": "State vector mixes raw unbounded waiting times (0-600s) with binary phase values (0-1).",
        "root_cause": "Waiting times are passed without division by 60.0, and queue counts without division by 15.0.",
        "mathematical_impact": (
            "Large-magnitude features (waiting seconds) overpower gradients by orders of magnitude compared to "
            "phase indicators (0.0 or 1.0). The neural network ignores the signal phase and focuses solely on noisy delays."
        ),
        "expected_behavior": "All 10 features normalized to approximately [0.0, 2.0].",
        "actual_behavior": "Features range from 0.0 to 450.0+, creating severe gradient scaling imbalances.",
        "faulty_code": (
            "# FAULTY IMPLEMENTATION:\n"
            "# Raw unnormalized features saturate network inputs\n"
            "state = [\n"
            "    queues['N'], queues['S'], queues['E'], queues['W'],\n"
            "    waits['N'], waits['S'], waits['E'], waits['W'],  # Raw seconds (0-500s)\n"
            "    phase_id, elapsed_time\n"
            "]"
        ),
        "corrected_code": (
            "# CORRECTED IMPLEMENTATION:\n"
            "# Normalized features with bounding safeguards\n"
            "state = [\n"
            "    min(2.0, queues['N'] / 15.0), min(2.0, queues['S'] / 15.0),\n"
            "    min(2.0, queues['E'] / 15.0), min(2.0, queues['W'] / 15.0),\n"
            "    min(3.0, waits['N'] / 60.0), min(3.0, waits['S'] / 60.0),\n"
            "    min(3.0, waits['E'] / 60.0), min(3.0, waits['W'] / 60.0),\n"
            "    float(phase_id), min(2.0, elapsed_time / 60.0)\n"
            "]"
        ),
        "validation_metric": "Neural layer activation distributions and weight gradient condition numbers."
    }
}

