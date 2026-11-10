import React from 'react';
import {
  Activity,
  Cpu,
  BarChart3,
  Sliders,
  History,
  BookOpen,
  LayoutDashboard,
  Zap,
  Network,
  Calculator,
  Compass,
  Bug,
  Code2,
  GitCompare,
  TrendingUp,
} from 'lucide-react';

export type NavTab =
  | 'overview'
  | 'simulation'
  | 'training'
  | 'laboratory'
  | 'mdp'
  | 'bellman'
  | 'policy-value'
  | 'comparison'
  | 'debugging-lab'
  | 'bug-repository'
  | 'code-comparison'
  | 'performance-eval'
  | 'experiments'
  | 'settings';

interface NavbarProps {
  activeTab: NavTab;
  onTabChange: (tab: NavTab) => void;
  isTraining?: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, onTabChange, isTraining }) => {
  const navItems: Array<{ id: NavTab; label: string; icon: React.ReactNode; badge?: string }> = [
    { id: 'overview', label: 'Overview', icon: <LayoutDashboard size={16} /> },
    { id: 'simulation', label: 'Simulation', icon: <Activity size={16} /> },
    { id: 'training', label: 'Training', icon: <Cpu size={16} /> },
    { id: 'laboratory', label: 'Algorithm Lab', icon: <BookOpen size={16} /> },
    { id: 'mdp', label: 'MDP Explorer', icon: <Network size={16} /> },
    { id: 'bellman', label: 'Bellman', icon: <Calculator size={16} /> },
    { id: 'debugging-lab', label: 'RL Debugging', icon: <Bug size={16} />, badge: 'Task 4' },
    { id: 'bug-repository', label: 'Bug Catalog', icon: <Code2 size={16} /> },
    { id: 'code-comparison', label: 'Code Diff', icon: <GitCompare size={16} /> },
    { id: 'performance-eval', label: 'Evaluation', icon: <TrendingUp size={16} /> },
    { id: 'experiments', label: 'Experiments', icon: <History size={16} /> },
    { id: 'settings', label: 'Settings', icon: <Sliders size={16} /> },
  ];

  return (
    <header className="sticky top-0 z-50 bg-[#0B0F19]/95 backdrop-blur-md border-b border-white/10 px-4 sm:px-6 py-2.5">
      <div className="max-w-[1600px] mx-auto flex items-center justify-between gap-4">
        {/* Brand / Logo */}
        <div className="flex items-center gap-2.5 cursor-pointer shrink-0" onClick={() => onTabChange('overview')}>
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-500 to-emerald-400 p-0.5 shadow-md shadow-indigo-500/20 flex items-center justify-center">
            <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
              <Zap className="text-emerald-400" size={18} />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <span className="font-extrabold text-sm tracking-tight text-white">TrafficRL</span>
              <span className="text-[9px] uppercase font-bold tracking-wider px-1.5 py-0.2 rounded bg-rose-500/20 text-rose-300 border border-rose-500/30">
                Task 4 Debugger
              </span>
            </div>
            <p className="text-[10px] text-slate-400 hidden xl:block">Intelligent Traffic Signal Optimization</p>
          </div>
        </div>

        {/* Navigation Tabs - Horizontal Scrolling on small/medium screens */}
        <nav className="flex items-center gap-1 bg-slate-900/70 p-1 rounded-xl border border-white/5 overflow-x-auto scrollbar-none max-w-full">
          {navItems.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onTabChange(item.id)}
                className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-all ${
                  isActive
                    ? 'bg-blue-600 text-white shadow-md shadow-blue-600/30 font-semibold'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
                }`}
              >
                {item.icon}
                <span>{item.label}</span>
                {item.badge && (
                  <span className="text-[8px] uppercase tracking-wider font-bold px-1 py-0.2 rounded bg-rose-400/20 text-rose-300 border border-rose-400/30">
                    {item.badge}
                  </span>
                )}
                {item.id === 'training' && isTraining && (
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-ping"></span>
                )}
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
};

