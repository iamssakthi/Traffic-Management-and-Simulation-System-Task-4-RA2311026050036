"""
Automated testing suite for Task 4: Advanced RL Debugging Challenge.
Validates bug detection, proves failure of faulty implementations,
and verifies mathematical correctness of corrected implementations.
"""
import pytest
import torch
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.debugging.bug_catalog import BUG_CATALOG
from app.debugging.faulty_agent import FaultyQLearningAgent
from app.debugging.corrected_agent import CorrectedQLearningAgent
from app.debugging.faulty_dqn import FaultyDQNAgent
from app.debugging.corrected_dqn import CorrectedDQNAgent
from app.debugging.faulty_replay_buffer import FaultyReplayBuffer
from app.debugging.corrected_replay_buffer import CorrectedReplayBuffer

client = TestClient(app)

def test_bug_catalog_integrity():
    """Verify bug taxonomy has all required categories and properties."""
    assert len(BUG_CATALOG) >= 6
    categories = {b["category"] for b in BUG_CATALOG.values()}
    assert "Algorithmic Error" in categories
    assert "Implementation Error" in categories
    assert "Training Error" in categories

    for bid, bug in BUG_CATALOG.items():
        assert "root_cause" in bug
        assert "mathematical_impact" in bug
        assert "faulty_code" in bug
        assert "corrected_code" in bug

def test_bug_alg_01_terminal_state_bootstrap():
    """
    BUG-ALG-01 Verification:
    Faulty agent bootstraps phantom future value when done=True.
    Corrected agent strictly enforces target = reward.
    """
    reward = -3.5
    next_s = 10

    # Faulty agent with non-zero next-state Q-values
    faulty = FaultyQLearningAgent(bug_terminal_bootstrap=True)
    faulty.q_table[next_s, 0] = 8.0
    faulty.q_table[next_s, 1] = 6.0

    # Terminal update (done = True)
    # Target should be -3.5, but faulty adds 0.95 * 8.0 = 7.6 -> Target = 4.1!
    faulty.update(state=0, action=0, reward=reward, next_state=next_s, done=True)
    # Check that Q-value was updated towards corrupted target
    assert faulty.bug_triggers["BUG-ALG-01"] >= 1
    assert faulty.q_table[0, 0] > reward  # Corrupted by phantom future value!

    # Corrected agent:
    corr = CorrectedQLearningAgent()
    corr.q_table[next_s, 0] = 8.0
    corr.q_table[next_s, 1] = 6.0
    corr.update(state=0, action=0, reward=reward, next_state=next_s, done=True)
    # Q(0,0) = 0 + 0.10 * (-3.5 - 0) = -0.35 (strictly negative!)
    assert corr.q_table[0, 0] == pytest.approx(0.10 * reward)
    assert corr.q_table[0, 0] < 0.0

def test_bug_alg_03_premature_epsilon_collapse():
    """
    BUG-ALG-03 Verification:
    Faulty agent decays epsilon rapidly to 0.0 within 8 episodes.
    Corrected agent preserves minimum exploration floor (epsilon_min = 0.05).
    """
    faulty = FaultyQLearningAgent(bug_premature_decay=True)
    corr = CorrectedQLearningAgent(epsilon_min=0.05, epsilon_decay=0.995)

    for _ in range(12):
        faulty.decay_epsilon()
        corr.decay_epsilon()

    # Faulty epsilon collapsed to 0
    assert faulty.epsilon == 0.0
    # Corrected epsilon preserves healthy exploration
    assert corr.epsilon >= 0.05

def test_bug_imp_01_target_gradient_leakage():
    """
    BUG-IMP-01 Verification:
    Faulty DQN leaks gradients into target network parameters.
    Corrected DQN target outputs have requires_grad=False.
    """
    # 1. Faulty DQN:
    faulty = FaultyDQNAgent(bug_leak_gradients=True, batch_size=4)
    # Seed buffer
    for i in range(10):
        s = np.zeros(10, dtype=np.float32)
        faulty.replay_buffer.push(s, 0, 1.0, s, False)

    b_states, b_actions, b_rewards, b_next_states, b_dones = faulty.replay_buffer.sample(4)
    q_next_faulty = faulty.target_net(b_next_states).max(1)[0].unsqueeze(1)
    # Faulty target tensor has active autograd graph!
    assert q_next_faulty.requires_grad is True

    # 2. Corrected DQN:
    corr = CorrectedDQNAgent(batch_size=4)
    for i in range(10):
        s = np.zeros(10, dtype=np.float32)
        corr.replay_buffer.push(s, 0, 1.0, s, False)

    b_states, b_actions, b_rewards, b_next_states, b_dones = corr.replay_buffer.sample(4)
    with torch.no_grad():
        q_next_corr = corr.target_net(b_next_states).max(1)[0].unsqueeze(1)
    # Corrected target tensor has zero autograd gradient
    assert q_next_corr.requires_grad is False

def test_bug_imp_03_replay_buffer_misalignment():
    """
    BUG-IMP-03 Verification:
    Faulty replay buffer shifts next_states by 1, corrupting transition alignment.
    Corrected replay buffer guarantees strict atomic alignment.
    """
    faulty_buf = FaultyReplayBuffer(capacity=100, seed=42, enable_misalignment_bug=True)
    corr_buf = CorrectedReplayBuffer(capacity=100, seed=42)

    # Insert sequential transitions with identifiable IDs
    for i in range(10):
        s = np.array([float(i)] * 10, dtype=np.float32)
        s_prime = np.array([float(i + 0.5)] * 10, dtype=np.float32)
        faulty_buf.push(s, 0, float(i), s_prime, False)
        corr_buf.push(s, 0, float(i), s_prime, False)

    # Sample batch
    f_states, _, _, f_next, _ = faulty_buf.sample(5)
    c_states, _, _, c_next, _ = corr_buf.sample(5)

    # In corrected buffer, each next_state[k][0] == state[k][0] + 0.5
    for k in range(5):
        assert c_next[k][0].item() == pytest.approx(c_states[k][0].item() + 0.5)

    # In faulty buffer with misalignment bug, at least one pair is shifted
    mismatches = 0
    for k in range(5):
        if abs(f_next[k][0].item() - (f_states[k][0].item() + 0.5)) > 0.01:
            mismatches += 1
    assert mismatches > 0  # Confirms that misalignment bug was triggered!

def test_debugging_api_endpoints():
    """Integration test for FastAPI Task 4 debugging endpoints."""
    # 1. Overview
    res = client.get("/api/debugging/overview")
    assert res.status_code == 200
    data = res.json()
    assert data["total_bugs"] >= 6
    assert len(data["debugging_methodology"]) == 6

    # 2. Bug list
    bugs_res = client.get("/api/debugging/bugs")
    assert bugs_res.status_code == 200
    assert bugs_res.json()["total"] >= 6

    # 3. Bug detail
    detail_res = client.get("/api/debugging/bugs/BUG-ALG-01")
    assert detail_res.status_code == 200
    assert detail_res.json()["id"] == "BUG-ALG-01"

    # 4. Code diff
    diff_res = client.get("/api/debugging/code-diff/BUG-ALG-01")
    assert diff_res.status_code == 200
    assert "faulty_code" in diff_res.json()
    assert "corrected_code" in diff_res.json()

    # 5. Automated test endpoint
    test_res = client.post("/api/debugging/test?bug_id=BUG-ALG-01")
    assert test_res.status_code == 200
    t_data = test_res.json()
    assert t_data["faulty_passed"] is False
    assert t_data["corrected_passed"] is True
    assert t_data["status"] == "VERIFIED_CORRECTED"

