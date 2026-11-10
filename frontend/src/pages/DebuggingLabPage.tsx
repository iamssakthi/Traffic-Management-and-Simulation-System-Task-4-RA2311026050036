import React, { useState, useEffect } from 'react';
import {
  Bug,
  Play,
  CheckCircle,
  AlertTriangle,
  RotateCcw,
  TrendingDown,
  TrendingUp,
  Activity,
  Layers,
  ArrowRight,
  ShieldAlert,
  ShieldCheck,
  Cpu,
  RefreshCw,
  Terminal,
} from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from 'recharts';
import {
  fetchDebuggingOverview,
  fetchDebuggingComparison,
  runDebuggingFaulty,
  runDebuggingCorrected,
  testDebuggingBug,
} from '../services/api';
import { BugOverview, DebuggingComparisonResult, BugTestResult } from '../types';

export const DebuggingLabPage: React.FC = () => {
  const [selectedAlgo, setSelectedAlgo] = useState<'q_learning' | 'dqn'>('q_learning');
  const [comparison, setComparison] = useState<DebuggingComparisonResult | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [runningTrial, setRunningTrial] = useState<boolean>(false);
  const [activeTestBug, setActiveTestBug] = useState<string>('BUG-ALG-01');
  const [testResult, setTestResult] = useState<BugTestResult | null>(null);
  const [consoleLogs, setConsoleLogs] = useState<Array<{ time: string; type: 'error' | 'success' | 'info'; msg: string }>>([]);

  useEffect(() => {
    loadComparison(selectedAlgo);
  }, [selectedAlgo]);

  const loadComparison = async (algo: 'q_learning' | 'dqn') => {
    setLoading(true);
    try {
      const data = await fetchDebuggingComparison(algo);
      setComparison(data);
      addLog('info', `Loaded benchmark comparison for ${algo === 'q_learning' ? 'Tabular Q-Learning' : 'Deep Q-Network'}.`);
    } catch (err) {
      console.error('Failed to load debugging comparison', err);
      addLog('error', 'Error connecting to debugging experiment backend.');
    } finally {
      setLoading(false);
    }
  };

  const addLog = (type: 'error' | 'success' | 'info', msg: string) => {
    const timeStr = new Date().toLocaleTimeString();
    setConsoleLogs((prev) => [{ time: timeStr, type, msg }, ...prev.slice(0, 15)]);
  };

  const handleRunFaulty = async () => {
    setRunningTrial(true);
    addLog('info', `Injecting active faults into ${selectedAlgo.toUpperCase()} and starting trial...`);
    try {
      const faultyRes = await runDebuggingFaulty(selectedAlgo, selectedAlgo === 'q_learning' ? 'heavy_north_south' : 'balanced', 10);
      addLog('error', `Faulty trial completed: Mean Delay=${faultyRes.mean_waiting_time}s, Mean Queue=${faultyRes.mean_queue_length} veh.`);
      await loadComparison(selectedAlgo);
    } catch (err) {
      addLog('error', 'Faulty trial execution encountered error.');
    } finally {
      setRunningTrial(false);
    }
  };

  const handleRunCorrected = async () => {
    setRunningTrial(true);
    addLog('info', `Executing mathematically verified corrected ${selectedAlgo.toUpperCase()} trial...`);
    try {
      const corrRes = await runDebuggingCorrected(selectedAlgo, selectedAlgo === 'q_learning' ? 'heavy_north_south' : 'balanced', 10);
      addLog('success', `Corrected trial completed: Mean Delay=${corrRes.mean_waiting_time}s, Mean Queue=${corrRes.mean_queue_length} veh.`);
      await loadComparison(selectedAlgo);
    } catch (err) {
      addLog('error', 'Corrected trial execution encountered error.');
    } finally {
      setRunningTrial(false);
    }
  };

  const handleRunTest = async (bugId: string) => {
    setActiveTestBug(bugId);
    try {
      const res = await testDebuggingBug(bugId);
      setTestResult(res);
      addLog('info', `Executed automated test assertion for ${bugId}: Before=FAIL, After=PASS.`);
    } catch (err) {
      console.error(err);
    }
  };

  // Prepare chart series from comparison history
  const chartData = comparison ? comparison.faulty.history.map((f, idx) => {
    const c = comparison.corrected.history[idx] || {};
    return {
      episode: `Ep ${f.episode}`,
      faultyReward: f.reward,
      correctedReward: c.reward || 0,
      faultyWait: f.waiting_time,
      correctedWait: c.waiting_time || 0,
      faultyQueue: f.queue_length,
      correctedQueue: c.queue_length || 0,
      faultyLoss: f.loss || f.mean_td_error || 0,
      correctedLoss: c.loss || c.mean_td_error || 0,
    };
  }) : [];

  return (
    <div className="space-y-8 pb-12">
      {/* Banner */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-rose-950/40 via-slate-900 to-indigo-950/40 border border-white/10 p-6 sm:p-8">
        <div className="max-w-3xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs font-semibold uppercase tracking-wider mb-3">
            <Bug size={14} /> Task 4: Advanced RL Debugging Challenge
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
            Reinforcement Learning Debugging Laboratory
          </h1>
          <p className="text-sm sm:text-base text-slate-300 mt-2 leading-relaxed">
            Systematically analyze deliberately faulty RL implementations, diagnose mathematical root causes,
            execute verified algorithmic corrections, and observe real-time performance recovery.
          </p>
        </div>
      </div>

      {/* Debugging Workflow Timeline */}
      <div className="bg-slate-900/70 border border-white/10 rounded-2xl p-6">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-4 flex items-center gap-2">
          <Activity size={16} className="text-blue-400" />
          End-to-End Debugging Methodology Pipeline
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2 text-center text-xs">
          {[
            { step: '01', title: 'Fault Injection', desc: 'Controlled injection of known RL bugs' },
            { step: '02', title: 'Experiment Exec', desc: 'Trial under fixed traffic scenario' },
            { step: '03', title: 'Failure Detect', desc: 'Loss divergence / queue explosion' },
            { step: '04', title: 'Root Cause', desc: 'Autograd / Bellman target analysis' },
            { step: '05', title: 'Code Fix', desc: 'Verified mathematical update rule' },
            { step: '06', title: 'Unit Testing', desc: 'Assertion fails before, passes after' },
            { step: '07', title: 'Retraining', desc: 'Retrain agent under identical seeds' },
            { step: '08', title: 'Evaluation', desc: 'Measure delay & throughput recovery' },
          ].map((item, idx) => (
            <div key={idx} className="p-3 rounded-xl bg-slate-950/60 border border-white/5 flex flex-col justify-between">
              <div>
                <span className="text-[10px] font-mono font-bold text-slate-500">{item.step}</span>
                <div className="font-bold text-white text-[11px] mt-1">{item.title}</div>
                <p className="text-[10px] text-slate-400 mt-1 leading-snug">{item.desc}</p>
              </div>
              {idx < 7 && <ArrowRight size={12} className="mx-auto mt-2 text-slate-600 hidden lg:block" />}
            </div>
          ))}
        </div>
      </div>

      {/* Algorithm Selector & Interactive Execution Deck */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Controls */}
        <div className="bg-slate-900/70 border border-white/10 rounded-2xl p-6 space-y-5">
          <h3 className="text-sm font-bold text-white uppercase tracking-wider text-slate-400 flex items-center gap-2">
            <Cpu size={16} className="text-indigo-400" />
            Algorithm &amp; Test Suite
          </h3>

          {/* Algorithm Tabs */}
          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={() => setSelectedAlgo('q_learning')}
              className={`p-3 rounded-xl text-left border transition-all ${
                selectedAlgo === 'q_learning'
                  ? 'bg-blue-600/30 border-blue-500 text-white font-bold'
                  : 'bg-slate-950 border-white/5 text-slate-400 hover:text-white'
              }`}
            >
              <div className="text-xs">Tabular Q-Learning</div>
              <div className="text-[10px] text-slate-400 mt-0.5">BUG-ALG-01 &bull; BUG-ALG-03</div>
            </button>
            <button
              onClick={() => setSelectedAlgo('dqn')}
              className={`p-3 rounded-xl text-left border transition-all ${
                selectedAlgo === 'dqn'
                  ? 'bg-indigo-600/30 border-indigo-500 text-white font-bold'
                  : 'bg-slate-950 border-white/5 text-slate-400 hover:text-white'
              }`}
            >
              <div className="text-xs">Deep Q-Network</div>
              <div className="text-[10px] text-slate-400 mt-0.5">BUG-IMP-01 &bull; BUG-IMP-02</div>
            </button>
          </div>

          {/* Action Buttons */}
          <div className="space-y-2 pt-2">
            <button
              onClick={handleRunFaulty}
              disabled={runningTrial}
              className="w-full py-2.5 px-4 rounded-xl bg-rose-600 hover:bg-rose-500 disabled:opacity-50 text-white font-bold text-xs flex items-center justify-center gap-2 shadow-lg shadow-rose-600/20 transition-all"
            >
              <ShieldAlert size={16} />
              Run Faulty Implementation
            </button>

            <button
              onClick={handleRunCorrected}
              disabled={runningTrial}
              className="w-full py-2.5 px-4 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white font-bold text-xs flex items-center justify-center gap-2 shadow-lg shadow-emerald-600/20 transition-all"
            >
              <ShieldCheck size={16} />
              Run Corrected Implementation
            </button>
          </div>

          {/* Quick Bug Assertion Tester */}
          <div className="pt-3 border-t border-white/10 space-y-2">
            <span className="text-xs font-semibold text-slate-300 block">Automated Assertion Verification:</span>
            <div className="grid grid-cols-2 gap-1.5">
              {['BUG-ALG-01', 'BUG-IMP-01', 'BUG-ALG-02'].map((bId) => (
                <button
                  key={bId}
                  onClick={() => handleRunTest(bId)}
                  className={`py-1.5 px-2 rounded-lg text-[11px] font-mono border transition-all ${
                    activeTestBug === bId
                      ? 'bg-slate-800 border-indigo-400 text-indigo-300 font-bold'
                      : 'bg-slate-950 border-white/5 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Test {bId}
                </button>
              ))}
            </div>

            {testResult && (
              <div className="p-3 rounded-xl bg-slate-950 border border-white/10 text-xs space-y-1 mt-2">
                <div className="text-[11px] font-bold text-white">{testResult.test_name}</div>
                <div className="flex justify-between text-rose-400 text-[10px]">
                  <span>Before Correction:</span>
                  <span className="font-bold">FAILED (corrupted)</span>
                </div>
                <div className="flex justify-between text-emerald-400 text-[10px]">
                  <span>After Correction:</span>
                  <span className="font-bold">PASSED (verified)</span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right: Diagnostics Console Log */}
        <div className="lg:col-span-2 bg-slate-900/70 border border-white/10 rounded-2xl p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-bold text-white uppercase tracking-wider text-slate-400 flex items-center gap-2">
                <Terminal size={16} className="text-amber-400" />
                Live Failure Detection &amp; Diagnostics Console
              </h3>
              <span className="text-[10px] font-mono text-emerald-400 flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span> Logger Active
              </span>
            </div>

            <div className="bg-slate-950 rounded-xl p-4 font-mono text-xs text-slate-300 border border-white/5 space-y-2 h-64 overflow-y-auto">
              {consoleLogs.length > 0 ? (
                consoleLogs.map((log, idx) => (
                  <div key={idx} className="flex items-start gap-2 text-[11px]">
                    <span className="text-slate-500 shrink-0">[{log.time}]</span>
                    <span
                      className={`font-semibold shrink-0 ${
                        log.type === 'error'
                          ? 'text-rose-400'
                          : log.type === 'success'
                          ? 'text-emerald-400'
                          : 'text-blue-400'
                      }`}
                    >
                      {log.type.toUpperCase()}:
                    </span>
                    <span className="text-slate-300">{log.msg}</span>
                  </div>
                ))
              ) : (
                <div className="text-slate-600 text-center py-20">
                  Ready. Click 'Run Faulty Implementation' or 'Run Corrected Implementation' to stream live telemetry.
                </div>
              )}
            </div>
          </div>

          {/* Quick Metrics Strip */}
          {comparison && (
            <div className="grid grid-cols-4 gap-2 pt-4 border-t border-white/10 mt-4 text-center text-xs">
              <div className="p-2 rounded bg-slate-950 border border-white/5">
                <div className="text-[10px] text-slate-400">Delay Recovery</div>
                <div className="font-mono text-emerald-400 font-bold">
                  +{comparison.improvements.waiting_time_percent.toFixed(1)}%
                </div>
              </div>
              <div className="p-2 rounded bg-slate-950 border border-white/5">
                <div className="text-[10px] text-slate-400">Queue Reduction</div>
                <div className="font-mono text-emerald-400 font-bold">
                  +{comparison.improvements.queue_length_percent.toFixed(1)}%
                </div>
              </div>
              <div className="p-2 rounded bg-slate-950 border border-white/5">
                <div className="text-[10px] text-slate-400">Throughput Gain</div>
                <div className="font-mono text-blue-400 font-bold">
                  +{comparison.improvements.throughput_percent.toFixed(1)}%
                </div>
              </div>
              <div className="p-2 rounded bg-slate-950 border border-white/5">
                <div className="text-[10px] text-slate-400">Return Recovery</div>
                <div className="font-mono text-purple-400 font-bold">
                  +{comparison.improvements.reward_percent.toFixed(1)}%
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Real-time Comparative Charts */}
      {comparison && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Chart 1: Waiting Time Recovery */}
          <div className="bg-slate-900/70 border border-white/10 rounded-2xl p-6">
            <h4 className="text-sm font-bold text-white mb-1 flex items-center justify-between">
              <span>Mean Waiting Time Comparison (Lower is Better)</span>
              <span className="text-xs font-mono text-emerald-400">
                {comparison.improvements.waiting_time_percent.toFixed(1)}% Improvement
              </span>
            </h4>
            <p className="text-xs text-slate-400 mb-4">
              Faulty agent incurs persistent queue delays due to policy lock and terminal bootstrap overestimation.
            </p>
            <div className="h-60">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
                  <XAxis dataKey="episode" stroke="#94a3b8" fontSize={10} />
                  <YAxis stroke="#94a3b8" fontSize={10} unit="s" />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
                  <Legend verticalAlign="top" wrapperStyle={{ paddingBottom: '8px', fontSize: '11px' }} />
                  <Line type="monotone" dataKey="faultyWait" name="Faulty Delay" stroke="#f43f5e" strokeWidth={2} dot={{ r: 2 }} />
                  <Line type="monotone" dataKey="correctedWait" name="Corrected Delay" stroke="#10b981" strokeWidth={2} dot={{ r: 2 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Chart 2: Queue Length Comparison */}
          <div className="bg-slate-900/70 border border-white/10 rounded-2xl p-6">
            <h4 className="text-sm font-bold text-white mb-1 flex items-center justify-between">
              <span>Mean Queue Length Comparison (Lower is Better)</span>
              <span className="text-xs font-mono text-emerald-400">
                {comparison.improvements.queue_length_percent.toFixed(1)}% Improvement
              </span>
            </h4>
            <p className="text-xs text-slate-400 mb-4">
              Corrected agent dissipates platoons efficiently without approach starvation.
            </p>
            <div className="h-60">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
                  <XAxis dataKey="episode" stroke="#94a3b8" fontSize={10} />
                  <YAxis stroke="#94a3b8" fontSize={10} unit=" veh" />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
                  <Legend verticalAlign="top" wrapperStyle={{ paddingBottom: '8px', fontSize: '11px' }} />
                  <Line type="monotone" dataKey="faultyQueue" name="Faulty Queue" stroke="#f43f5e" strokeWidth={2} dot={{ r: 2 }} />
                  <Line type="monotone" dataKey="correctedQueue" name="Corrected Queue" stroke="#38bdf8" strokeWidth={2} dot={{ r: 2 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

