import React, { useState, useEffect } from 'react';
import { fetchDebuggingBugs, fetchDebuggingCodeDiff } from '../services/api';
import { BugItem } from '../types';
import {
  Code2,
  CheckCircle2,
  XCircle,
  FileCode,
  Copy,
  Check,
  AlertTriangle,
  Lightbulb,
  ArrowRight,
  ShieldCheck,
} from 'lucide-react';

interface CodeComparisonProps {
  initialBugId?: string;
  onNavigateToRepository?: () => void;
}

export const CodeComparisonPage: React.FC<CodeComparisonProps> = ({
  initialBugId,
  onNavigateToRepository,
}) => {
  const [bugs, setBugs] = useState<BugItem[]>([]);
  const [selectedBugId, setSelectedBugId] = useState<string>(initialBugId || 'BUG-ALG-01');
  const [diffData, setDiffData] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [copiedSide, setCopiedSide] = useState<'faulty' | 'corrected' | null>(null);

  useEffect(() => {
    loadBugs();
  }, []);

  useEffect(() => {
    if (selectedBugId) {
      loadDiff(selectedBugId);
    }
  }, [selectedBugId]);

  const loadBugs = async () => {
    try {
      const res = await fetchDebuggingBugs();
      setBugs(res.bugs || []);
      if (res.bugs && res.bugs.length > 0 && !initialBugId) {
        setSelectedBugId(res.bugs[0].id);
      }
    } catch (err) {
      console.error('Failed to load bugs for code diff:', err);
    }
  };

  const loadDiff = async (bugId: string) => {
    try {
      setLoading(true);
      const res = await fetchDebuggingCodeDiff(bugId);
      setDiffData(res);
    } catch (err) {
      console.error('Failed to load code diff:', err);
    } finally {
      setLoading(false);
    }
  };

  const currentBug = bugs.find((b) => b.id === selectedBugId) || (diffData?.bug as BugItem);

  const handleCopy = (text: string, side: 'faulty' | 'corrected') => {
    navigator.clipboard.writeText(text);
    setCopiedSide(side);
    setTimeout(() => setCopiedSide(null), 2000);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-3 py-1 text-xs font-bold tracking-wider uppercase bg-blue-500/20 text-blue-400 border border-blue-500/30 rounded-md">
              Task 4 Code Diff Inspection
            </span>
            <span className="text-xs text-slate-400">Side-by-Side Algorithmic & Implementation Comparison</span>
          </div>
          <h1 className="text-2xl font-black text-white mt-2 flex items-center gap-2">
            <Code2 className="w-7 h-7 text-blue-400" />
            Faulty vs Corrected Implementation Code Diff
          </h1>
          <p className="text-sm text-slate-400 mt-1 max-w-3xl">
            Inspect source code side-by-side to pinpoint erroneous Bellman bootstrapping, backprop graph leakage,
            off-by-one buffer index shifts, and premature hyperparameter decays.
          </p>
        </div>

        {onNavigateToRepository && (
          <button
            onClick={onNavigateToRepository}
            className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold rounded-lg border border-slate-700 transition"
          >
            <span>Bug Repository</span>
            <ArrowRight className="w-4 h-4 text-emerald-400" />
          </button>
        )}
      </div>

      {/* Bug Selector Bar */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">Select Defect:</label>
          <select
            value={selectedBugId}
            onChange={(e) => setSelectedBugId(e.target.value)}
            className="bg-slate-800 border border-slate-700 text-sm font-semibold rounded-lg px-3 py-1.5 text-white focus:outline-none focus:border-blue-500"
          >
            {bugs.map((b) => (
              <option key={b.id} value={b.id}>
                {b.id}: {b.subcategory} ({b.affected_algorithms.join(', ')})
              </option>
            ))}
          </select>
        </div>

        {currentBug && (
          <div className="flex items-center gap-3 text-xs">
            <span className="font-mono text-slate-400 flex items-center gap-1.5">
              <FileCode className="w-4 h-4 text-slate-400" />
              {currentBug.file_name} : line {currentBug.line_number}
            </span>
            <span className="px-2.5 py-0.5 rounded-full font-semibold bg-blue-900/40 text-blue-300 border border-blue-800">
              {currentBug.category}
            </span>
          </div>
        )}
      </div>

      {/* Side-by-Side Diff Panels */}
      {loading ? (
        <div className="p-16 text-center text-slate-400 bg-slate-900/50 rounded-xl border border-slate-800">
          Loading code comparison diff...
        </div>
      ) : currentBug ? (
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left: Faulty Implementation */}
            <div className="bg-slate-900/90 border border-rose-900/50 rounded-xl overflow-hidden shadow-lg shadow-rose-950/20">
              <div className="bg-rose-950/40 border-b border-rose-900/60 px-4 py-3 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <XCircle className="w-5 h-5 text-rose-400" />
                  <span className="text-sm font-bold text-rose-300 uppercase tracking-wider">
                    Faulty Implementation
                  </span>
                  <span className="text-xs text-rose-400/80 font-mono">(Contains Defect)</span>
                </div>
                <button
                  onClick={() => handleCopy(currentBug.faulty_code, 'faulty')}
                  className="flex items-center gap-1 text-xs text-slate-400 hover:text-white px-2 py-1 rounded bg-slate-800/80 hover:bg-slate-700 transition"
                  title="Copy code"
                >
                  {copiedSide === 'faulty' ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-emerald-400" />
                      <span className="text-emerald-400">Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5" />
                      <span>Copy</span>
                    </>
                  )}
                </button>
              </div>

              <div className="p-4 bg-slate-950 font-mono text-xs overflow-x-auto text-rose-200/90 leading-relaxed">
                <pre className="whitespace-pre">{currentBug.faulty_code}</pre>
              </div>

              <div className="p-3 bg-rose-950/20 border-t border-rose-900/40 text-xs text-rose-300 flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold">Erroneous Behavior: </span>
                  {currentBug.actual_behavior}
                </div>
              </div>
            </div>

            {/* Right: Corrected Implementation */}
            <div className="bg-slate-900/90 border border-emerald-900/50 rounded-xl overflow-hidden shadow-lg shadow-emerald-950/20">
              <div className="bg-emerald-950/40 border-b border-emerald-900/60 px-4 py-3 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                  <span className="text-sm font-bold text-emerald-300 uppercase tracking-wider">
                    Corrected Implementation
                  </span>
                  <span className="text-xs text-emerald-400/80 font-mono">(Rigorous / Validated)</span>
                </div>
                <button
                  onClick={() => handleCopy(currentBug.corrected_code, 'corrected')}
                  className="flex items-center gap-1 text-xs text-slate-400 hover:text-white px-2 py-1 rounded bg-slate-800/80 hover:bg-slate-700 transition"
                  title="Copy code"
                >
                  {copiedSide === 'corrected' ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-emerald-400" />
                      <span className="text-emerald-400">Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5" />
                      <span>Copy</span>
                    </>
                  )}
                </button>
              </div>

              <div className="p-4 bg-slate-950 font-mono text-xs overflow-x-auto text-emerald-200/90 leading-relaxed">
                <pre className="whitespace-pre">{currentBug.corrected_code}</pre>
              </div>

              <div className="p-3 bg-emerald-950/20 border-t border-emerald-900/40 text-xs text-emerald-300 flex items-start gap-2">
                <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold">Corrected Contract: </span>
                  {currentBug.expected_behavior}
                </div>
              </div>
            </div>
          </div>

          {/* Theoretical & Mathematical Justification Box */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
            <div className="flex items-center gap-2 text-sm font-bold text-amber-400 uppercase tracking-wider">
              <Lightbulb className="w-4 h-4 text-amber-400" />
              Mathematical Justification & Rigorous Correction Analysis
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-300 leading-relaxed">
              <div className="bg-slate-950/60 p-4 rounded-lg border border-slate-800">
                <span className="font-semibold text-rose-300 block mb-1">Root Cause Mechanism:</span>
                {currentBug.root_cause}
              </div>
              <div className="bg-slate-950/60 p-4 rounded-lg border border-slate-800">
                <span className="font-semibold text-emerald-300 block mb-1">Empirical & Convergence Impact:</span>
                {currentBug.mathematical_impact}
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
};

