import {
  SimulationState,
  Scenario,
  AlgorithmInfo,
  TrainingStatus,
  TrainingMetric,
  Experiment,
  EvaluationComparison,
  QTableSummary,
  AppConfig,
  MdpDefinition,
  MdpState,
  MdpActions,
  MdpPolicy,
  MdpStateValueItem,
  BellmanResult,
  RewardResult,
  DecisionRecord,
  TransitionSummary,
  BugItem,
  BugOverview,
  DebuggingComparisonResult,
  DebuggingMetricComparisonItem,
  BugTestResult,
} from '../types';

const BASE_URL = '';

export async function fetchHealth(): Promise<{ status: string; backend: string }> {
  const res = await fetch(`${BASE_URL}/api/health`);
  return res.json();
}

export async function fetchAlgorithms(): Promise<AlgorithmInfo[]> {
  const res = await fetch(`${BASE_URL}/api/algorithms`);
  const data = await res.json();
  return data.algorithms;
}

export async function fetchScenarios(): Promise<Scenario[]> {
  const res = await fetch(`${BASE_URL}/api/scenarios`);
  const data = await res.json();
  return data.scenarios;
}

export async function fetchConfig(): Promise<AppConfig> {
  const res = await fetch(`${BASE_URL}/api/config`);
  return res.json();
}

export async function updateConfig(cfg: Partial<AppConfig['timing'] & AppConfig['rewards']>): Promise<any> {
  const res = await fetch(`${BASE_URL}/api/config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(cfg),
  });
  return res.json();
}

// Training APIs
export async function startTraining(params: {
  algorithm: string;
  episodes: number;
  episode_duration: number;
  scenario_id: string;
  alpha: number;
  gamma: number;
  epsilon: number;
  epsilon_min: number;
  epsilon_decay: number;
  seed: number;
}): Promise<any> {
  const res = await fetch(`${BASE_URL}/api/train/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return res.json();
}

export async function fetchTrainingStatus(): Promise<TrainingStatus> {
  const res = await fetch(`${BASE_URL}/api/train/status`);
  return res.json();
}

export async function fetchTrainingHistory(): Promise<TrainingMetric[]> {
  const res = await fetch(`${BASE_URL}/api/train/history`);
  const data = await res.json();
  return data.history;
}

export async function pauseTraining(): Promise<any> {
  const res = await fetch(`${BASE_URL}/api/train/pause`, { method: 'POST' });
  return res.json();
}

export async function resumeTraining(): Promise<any> {
  const res = await fetch(`${BASE_URL}/api/train/resume`, { method: 'POST' });
  return res.json();
}

export async function stopTraining(): Promise<any> {
  const res = await fetch(`${BASE_URL}/api/train/stop`, { method: 'POST' });
  return res.json();
}

// Model & Lab APIs
export async function fetchQTable(): Promise<QTableSummary> {
  const res = await fetch(`${BASE_URL}/api/models/q_table`);
  return res.json();
}

export async function fetchModels(): Promise<any[]> {
  const res = await fetch(`${BASE_URL}/api/models`);
  const data = await res.json();
  return data.models;
}

// Evaluation APIs
export async function runEvaluation(params: {
  algorithm: string;
  scenario_id: string;
  fixed_green: number;
  num_episodes: number;
}): Promise<EvaluationComparison> {
  const res = await fetch(`${BASE_URL}/api/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return res.json();
}

// Experiment APIs
export async function fetchExperiments(): Promise<Experiment[]> {
  const res = await fetch(`${BASE_URL}/api/experiments`);
  const data = await res.json();
  return data.experiments;
}

// Simulation Control APIs
export async function fetchSimulationState(): Promise<SimulationState> {
  const res = await fetch(`${BASE_URL}/api/simulation/state`);
  return res.json();
}

export async function startSimulation(): Promise<any> {
  const res = await fetch(`${BASE_URL}/api/simulation/start`, { method: 'POST' });
  return res.json();
}

export async function pauseSimulation(): Promise<any> {
  const res = await fetch(`${BASE_URL}/api/simulation/pause`, { method: 'POST' });
  return res.json();
}

export async function stepSimulation(action: number = 0, controller_type: string = 'manual'): Promise<SimulationState> {
  const res = await fetch(`${BASE_URL}/api/simulation/step`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, controller_type }),
  });
  const data = await res.json();
  return data.state;
}

export async function resetSimulation(scenario_id: string = 'balanced', seed?: number): Promise<SimulationState> {
  const res = await fetch(`${BASE_URL}/api/simulation/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scenario_id, seed }),
  });
  const data = await res.json();
  return data.state;
}

export async function setSimulationController(controller_type: string): Promise<any> {
  const res = await fetch(`${BASE_URL}/api/simulation/controller?controller_type=${controller_type}`, {
    method: 'POST',
  });
  return res.json();
}

export async function setSimulationSpeed(speed: number): Promise<any> {
  const res = await fetch(`${BASE_URL}/api/simulation/speed?speed=${speed}`, {
    method: 'POST',
  });
  return res.json();
}

// -------------------------------------------------------------
// Task 3: MDP & Bellman Service Endpoints
// -------------------------------------------------------------
export async function fetchMdpDefinition(): Promise<MdpDefinition> {
  const res = await fetch(`${BASE_URL}/api/mdp/definition`);
  return res.json();
}

export async function fetchMdpState(): Promise<MdpState> {
  const res = await fetch(`${BASE_URL}/api/mdp/state`);
  return res.json();
}

export async function fetchMdpActions(): Promise<MdpActions> {
  const res = await fetch(`${BASE_URL}/api/mdp/actions`);
  return res.json();
}

export async function fetchMdpPolicy(state?: number, epsilon: number = 0.0): Promise<MdpPolicy> {
  const url = state !== undefined
    ? `${BASE_URL}/api/mdp/policy?state=${state}&epsilon=${epsilon}`
    : `${BASE_URL}/api/mdp/policy?epsilon=${epsilon}`;
  const res = await fetch(url);
  return res.json();
}

export async function fetchMdpValues(phase?: number, limit: number = 100): Promise<{ total_states: number; filtered_count: number; states: MdpStateValueItem[] }> {
  const url = phase !== undefined
    ? `${BASE_URL}/api/mdp/value?phase=${phase}&limit=${limit}`
    : `${BASE_URL}/api/mdp/value?limit=${limit}`;
  const res = await fetch(url);
  return res.json();
}

export async function calculateBellman(params: {
  q_current: number;
  alpha: number;
  reward: number;
  gamma: number;
  next_max_q: number;
}): Promise<BellmanResult> {
  const res = await fetch(`${BASE_URL}/api/mdp/bellman/calculate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return res.json();
}

export async function calculateReward(params: {
  total_queue: number;
  incremental_wait: number;
  switch_triggered: boolean;
  departures: number;
  queue_weight?: number;
  waiting_weight?: number;
  switch_weight?: number;
  throughput_weight?: number;
}): Promise<RewardResult> {
  const res = await fetch(`${BASE_URL}/api/mdp/reward/calculate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return res.json();
}

export async function fetchMdpTransitions(limit: number = 30): Promise<TransitionSummary> {
  const res = await fetch(`${BASE_URL}/api/mdp/transitions?limit=${limit}`);
  return res.json();
}

export async function fetchDecisionHistory(limit: number = 50): Promise<{ history: DecisionRecord[] }> {
  const res = await fetch(`${BASE_URL}/api/mdp/decision-history?limit=${limit}`);
  return res.json();
}

// -------------------------------------------------------------
// Task 4: Advanced RL Debugging APIs
// -------------------------------------------------------------
export async function fetchDebuggingOverview(): Promise<BugOverview> {
  const res = await fetch(`${BASE_URL}/api/debugging/overview`);
  return res.json();
}

export async function fetchDebuggingBugs(category?: string, algorithm?: string): Promise<{ total: number; bugs: BugItem[] }> {
  let url = `${BASE_URL}/api/debugging/bugs`;
  const params: string[] = [];
  if (category) params.push(`category=${encodeURIComponent(category)}`);
  if (algorithm) params.push(`algorithm=${encodeURIComponent(algorithm)}`);
  if (params.length > 0) url += `?${params.join('&')}`;
  const res = await fetch(url);
  return res.json();
}

export async function fetchDebuggingBugDetail(bug_id: string): Promise<BugItem> {
  const res = await fetch(`${BASE_URL}/api/debugging/bugs/${bug_id}`);
  return res.json();
}

export async function fetchDebuggingCodeDiff(bug_id: string): Promise<any> {
  const res = await fetch(`${BASE_URL}/api/debugging/code-diff/${bug_id}`);
  return res.json();
}

export async function runDebuggingFaulty(algorithm: string, scenario_id: string, episodes: number = 10): Promise<any> {
  const res = await fetch(
    `${BASE_URL}/api/debugging/run-faulty?algorithm=${algorithm}&scenario_id=${scenario_id}&episodes=${episodes}`,
    { method: 'POST' }
  );
  return res.json();
}

export async function runDebuggingCorrected(algorithm: string, scenario_id: string, episodes: number = 10): Promise<any> {
  const res = await fetch(
    `${BASE_URL}/api/debugging/run-corrected?algorithm=${algorithm}&scenario_id=${scenario_id}&episodes=${episodes}`,
    { method: 'POST' }
  );
  return res.json();
}

export async function fetchDebuggingComparison(algorithm: string = 'q_learning'): Promise<DebuggingComparisonResult> {
  const res = await fetch(`${BASE_URL}/api/debugging/comparison?algorithm=${algorithm}`);
  return res.json();
}

export async function fetchDebuggingMetrics(): Promise<{ metrics: DebuggingMetricComparisonItem[]; summary: string }> {
  const res = await fetch(`${BASE_URL}/api/debugging/metrics`);
  return res.json();
}

export async function testDebuggingBug(bug_id: string): Promise<BugTestResult> {
  const res = await fetch(`${BASE_URL}/api/debugging/test?bug_id=${bug_id}`, { method: 'POST' });
  return res.json();
}

export async function resetDebugging(): Promise<any> {
  const res = await fetch(`${BASE_URL}/api/debugging/reset`, { method: 'POST' });
  return res.json();
}


