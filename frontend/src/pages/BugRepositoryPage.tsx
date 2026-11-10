import React, { useState, useEffect } from 'react';
import {
  fetchDebuggingBugs,
  testDebuggingBug,
} from '../services/api';
import { BugItem, BugTestResult } from '../types';
import {
  Bug,
  Filter,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Play,
  Code2,
  FileCode,
  Sparkles,
  ArrowRight,
  ShieldAlert,
} from 'lucide-react';

interface BugRepositoryProps {
  onNavigateToCode?: (bugId: string) => void;
}

export const BugRepositoryPage: React.FC<BugRepositoryProps> = ({ onNavigateToCode }) => {
  const [bugs, setBugs] = useState<BugItem[]>([]);
  const [filteredBugs, setFilteredBugs] = useState<BugItem[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string>('ALL');
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>('ALL');
  const [selectedSeverity, setSelectedSeverity] = useState<string>('ALL');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedBug, setSelectedBug] = useState<BugItem | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [testingBugId, setTestingBugId] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<BugTestResult | null>(null);

  useEffect(() => {
    loadBugs();
  }, []);

  const loadBugs = async () => {
    try {
      setLoading(true);
      const res = await fetchDebuggingBugs();
      setBugs(res.bugs || []);
      setFilteredBugs(res.bugs || []);
      if (res.bugs && res.bugs.length > 0) {
        setSelectedBug(res.bugs[0]);
      }
    } catch (err) {
      console.error('Failed to load bugs:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let result = [...bugs];
    if (selectedCategory !== 'ALL') {
      result = result.filter(b => b.category.toLowerCase().includes(selectedCategory.toLowerCase()));
    }
    if (selectedAlgorithm !== 'ALL') {
      result = result.filter(b =>
        b.affected_algorithms.some(algo => algo.toLowerCase().includes(selectedAlgorithm.toLowerCase()))
      );
    }
    if (selectedSeverity !== 'ALL') {
      result = result.filter(b => b.severity === selectedSeverity);
    }
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      result = result.filter(
        b =>
          b.id.toLowerCase().includes(q) ||
          b.subcategory.toLowerCase().includes(q) ||
          b.description.toLowerCase().includes(q) ||
          b.root_cause.toLowerCase().includes(q)
      );
    }
    setFilteredBugs(result);
    if (result.length > 0 && (!selectedBug || !result.some(b => b.id === selectedBug.id))) {
      setSelectedBug(result[0]);
    }
  }, [bugs, selectedCategory, selectedAlgorithm, selectedSeverity, searchQuery]);

  const handleRunTest = async (bugId: string) => {
    try {
      setTestingBugId(bugId);
      const res = await testDebuggingBug(bugId);
      setTestResult(res);
    } catch (err) {
      console.error('Failed to run bug assertion test:', err);
    } finally {
      setTestingBugId(null);
    }
  };

  const getSeverityBadge = (severity: string) => {
    switch (severity) {
      case 'CRITICAL':
        return <span className="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-rose-900/60 text-rose-300 border border-rose-700/50">CRITICAL</span>;
      case 'HIGH':
        return <span className="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-amber-900/60 text-amber-300 border border-amber-700/50">HIGH</span>;
      case 'MEDIUM':
        return <span className="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-blue-900/60 text-blue-300 border border-blue-700/50">MEDIUM</span>;
      default:
        return <span className="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-gray-800 text-gray-300">LOW</span>;
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-3 py-1 text-xs font-bold tracking-wider uppercase bg-red-500/20 text-red-400 border border-red-500/30 rounded-md">
              Task 4 Bug Taxonomy
            </span>
            <span className="text-xs text-slate-400">8 Injected Faults Cataloged & Audited</span>
          </div>
          <h1 className="text-2xl font-black text-white mt-2 flex items-center gap-2">
            <Bug className="w-7 h-7 text-red-400" />
            RL Defect Repository & Root Cause Catalog
          </h1>
          <p className="text-sm text-slate-400 mt-1 max-w-3xl">
            Exhaustive classification of algorithmic, computational, and architectural defects deliberately introduced into
            Tabular Q-Learning and Deep Q-Networks for empirical debugging and comparative validation.
          </p>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2 text-slate-400 text-sm font-medium">
          <Filter className="w-4 h-4 text-emerald-400" />
          <span>Filters:</span>
        </div>

        {/* Category filter */}
        <div className="flex items-center gap-1.5">
          <label className="text-xs text-slate-400">Category:</label>
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="bg-slate-800 border border-slate-700 rounded-lg text-xs px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-emerald-500"
          >
            <option value="ALL">All Categories</option>
            <option value="Algorithmic">Algorithmic</option>
            <option value="Implementation">Implementation / PyTorch</option>
            <option value="Environment">Environment / Reward</option>
          </select>
        </div>

        {/* Algorithm filter */}
        <div className="flex items-center gap-1.5">
          <label className="text-xs text-slate-400">Algorithm:</label>
          <select
            value={selectedAlgorithm}
            onChange={(e) => setSelectedAlgorithm(e.target.value)}
            className="bg-slate-800 border border-slate-700 rounded-lg text-xs px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-emerald-500"
          >
            <option value="ALL">All Algorithms</option>
            <option value="Q-Learning">Tabular Q-Learning</option>
            <option value="DQN">Deep Q-Network</option>
          </select>
        </div>

        {/* Severity filter */}
        <div className="flex items-center gap-1.5">
          <label className="text-xs text-slate-400">Severity:</label>
          <select
            value={selectedSeverity}
            onChange={(e) => setSelectedSeverity(e.target.value)}
            className="bg-slate-800 border border-slate-700 rounded-lg text-xs px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-emerald-500"
          >
            <option value="ALL">All Severities</option>
            <option value="CRITICAL">Critical</option>
            <option value="HIGH">High</option>
            <option value="MEDIUM">Medium</option>
          </select>
        </div>

        {/* Search input */}
        <div className="flex-1 min-w-[200px]">
          <input
            type="text"
            placeholder="Search bug ID, title, or root cause..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-slate-800/80 border border-slate-700 rounded-lg text-xs px-3 py-1.5 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500"
          />
        </div>

        <div className="text-xs text-slate-400">
          Showing <span className="font-semibold text-emerald-400">{filteredBugs.length}</span> of {bugs.length} defects
        </div>
      </div>

      {/* Main Split Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: List of Bugs */}
        <div className="lg:col-span-5 space-y-3">
          {loading ? (
            <div className="p-8 text-center text-slate-400">Loading catalog...</div>
          ) : filteredBugs.length === 0 ? (
            <div className="p-8 text-center text-slate-500 bg-slate-900 border border-slate-800 rounded-xl">
              No matching bugs found for selected filters.
            </div>
          ) : (
            filteredBugs.map((bug) => {
              const isSelected = selectedBug?.id === bug.id;
              return (
                <div
                  key={bug.id}
                  onClick={() => {
                    setSelectedBug(bug);
                    setTestResult(null);
                  }}
                  className={`p-4 rounded-xl border cursor-pointer transition-all ${
                    isSelected
                      ? 'bg-slate-900 border-emerald-500/70 shadow-lg shadow-emerald-950/30'
                      : 'bg-slate-900/60 border-slate-800 hover:bg-slate-900/90 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xs font-bold text-emerald-400 bg-emerald-950/60 border border-emerald-800/60 px-2 py-0.5 rounded">
                          {bug.id}
                        </span>
                        {getSeverityBadge(bug.severity)}
                      </div>
                      <h3 className="font-bold text-white text-sm mt-1">{bug.subcategory}</h3>
                    </div>
                  </div>

                  <p className="text-xs text-slate-400 line-clamp-2 mt-2">{bug.description}</p>

                  <div className="mt-3 pt-2.5 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
                    <span className="flex items-center gap-1 font-mono text-[11px] text-slate-300">
                      <FileCode className="w-3.5 h-3.5 text-slate-400" />
                      {bug.file_name}:{bug.line_number}
                    </span>
                    <span className="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-medium">
                      {bug.affected_algorithms.join(', ')}
                    </span>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Right Column: Detailed Bug Specification & Verification Panel */}
        <div className="lg:col-span-7">
          {selectedBug ? (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-6 sticky top-6">
              {/* Header */}
              <div className="flex items-start justify-between pb-4 border-b border-slate-800">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-sm font-bold text-emerald-400 bg-emerald-950/60 border border-emerald-800/60 px-2.5 py-0.5 rounded">
                      {selectedBug.id}
                    </span>
                    <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold">
                      {selectedBug.category}
                    </span>
                    {getSeverityBadge(selectedBug.severity)}
                  </div>
                  <h2 className="text-xl font-bold text-white mt-2">{selectedBug.subcategory}</h2>
                  <div className="flex items-center gap-2 mt-1.5 text-xs text-slate-400 font-mono">
                    <FileCode className="w-4 h-4 text-slate-400" />
                    <span>{selectedBug.file_name}</span>
                    <span>•</span>
                    <span className="text-emerald-400">def {selectedBug.function_name}()</span>
                    <span>•</span>
                    <span>Line {selectedBug.line_number}</span>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  {onNavigateToCode && (
                    <button
                      onClick={() => onNavigateToCode(selectedBug.id)}
                      className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600/80 hover:bg-blue-600 text-white text-xs font-semibold rounded-lg transition"
                    >
                      <Code2 className="w-3.5 h-3.5" />
                      Diff View
                    </button>
                  )}
                  <button
                    onClick={() => handleRunTest(selectedBug.id)}
                    disabled={testingBugId === selectedBug.id}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-xs font-semibold rounded-lg transition"
                  >
                    <Play className="w-3.5 h-3.5" />
                    {testingBugId === selectedBug.id ? 'Testing...' : 'Run Assertion Test'}
                  </button>
                </div>
              </div>

              {/* Bug Description */}
              <div className="space-y-1.5">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">Symptom & Description</h4>
                <p className="text-sm text-slate-200 leading-relaxed bg-slate-950/50 p-3 rounded-lg border border-slate-800">
                  {selectedBug.description}
                </p>
              </div>

              {/* Root Cause & Mathematical Impact */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-rose-950/20 border border-rose-900/40 rounded-lg p-3.5 space-y-1">
                  <div className="flex items-center gap-1.5 text-xs font-bold text-rose-400 uppercase tracking-wider">
                    <AlertTriangle className="w-3.5 h-3.5" />
                    Root Cause
                  </div>
                  <p className="text-xs text-rose-200/90 leading-relaxed">{selectedBug.root_cause}</p>
                </div>

                <div className="bg-amber-950/20 border border-amber-900/40 rounded-lg p-3.5 space-y-1">
                  <div className="flex items-center gap-1.5 text-xs font-bold text-amber-400 uppercase tracking-wider">
                    <ShieldAlert className="w-3.5 h-3.5" />
                    Mathematical Impact
                  </div>
                  <p className="text-xs text-amber-200/90 leading-relaxed">{selectedBug.mathematical_impact}</p>
                </div>
              </div>

              {/* Expected vs Actual Behavior */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-slate-950/70 border border-slate-800 rounded-lg p-3 space-y-1">
                  <div className="flex items-center gap-1.5 text-xs font-bold text-emerald-400">
                    <CheckCircle className="w-3.5 h-3.5" />
                    Expected Mathematical Behavior
                  </div>
                  <p className="text-xs text-slate-300">{selectedBug.expected_behavior}</p>
                </div>

                <div className="bg-slate-950/70 border border-slate-800 rounded-lg p-3 space-y-1">
                  <div className="flex items-center gap-1.5 text-xs font-bold text-rose-400">
                    <XCircle className="w-3.5 h-3.5" />
                    Faulty Observed Behavior
                  </div>
                  <p className="text-xs text-slate-300">{selectedBug.actual_behavior}</p>
                </div>
              </div>

              {/* Verification Metric */}
              <div className="bg-slate-950/60 border border-slate-800/80 rounded-lg p-3 flex items-center justify-between">
                <div className="text-xs text-slate-400">
                  <span className="font-semibold text-slate-300">Target Validation Metric: </span>
                  <span className="font-mono text-emerald-400">{selectedBug.validation_metric}</span>
                </div>
                <span className="text-[11px] text-slate-500">Automated PyTest verified</span>
              </div>

              {/* Real-Time Assertion Test Output Box */}
              {testResult && testResult.bug_id === selectedBug.id && (
                <div className="bg-slate-950 border border-emerald-500/40 rounded-xl p-4 space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <div className="flex items-center gap-2">
                      <Sparkles className="w-4 h-4 text-emerald-400" />
                      <span className="text-xs font-bold uppercase tracking-wider text-emerald-300">
                        Empirical Assertion Result: {testResult.test_name}
                      </span>
                    </div>
                    <span className="px-2 py-0.5 text-xs font-bold rounded bg-emerald-900/60 text-emerald-300 border border-emerald-700">
                      STATUS: {testResult.status}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div className="p-2.5 rounded-lg bg-rose-950/30 border border-rose-900/40">
                      <div className="text-rose-400 font-bold mb-1">Faulty Implementation:</div>
                      <div className="text-slate-300">
                        Passed Assertion: <span className="text-rose-400 font-bold">{testResult.faulty_passed ? 'YES' : 'FAIL (Defect Caught)'}</span>
                      </div>
                      {testResult.faulty_observed !== undefined && (
                        <div className="text-slate-400 mt-1 font-mono text-[11px] truncate">
                          Observed: {JSON.stringify(testResult.faulty_observed)}
                        </div>
                      )}
                    </div>

                    <div className="p-2.5 rounded-lg bg-emerald-950/30 border border-emerald-900/40">
                      <div className="text-emerald-400 font-bold mb-1">Corrected Implementation:</div>
                      <div className="text-slate-300">
                        Passed Assertion: <span className="text-emerald-400 font-bold">{testResult.corrected_passed ? 'PASS (Strict)' : 'FAIL'}</span>
                      </div>
                      {testResult.corrected_observed !== undefined && (
                        <div className="text-slate-400 mt-1 font-mono text-[11px] truncate">
                          Observed: {JSON.stringify(testResult.corrected_observed)}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="h-full flex items-center justify-center p-12 bg-slate-900/40 border border-slate-800 rounded-xl text-slate-500">
              Select a defect from the catalog to view details and execute tests.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

