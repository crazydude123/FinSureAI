'use client';

import { Progress } from './ui/progress';
import type { JobLifecycleState } from '@/lib/types';

interface Props {
  value: number;
  state: JobLifecycleState;
  label?: string;
}

export default function ProgressBar({ value, state, label }: Props) {
  const stateColor = state === 'COMPLETED' ? 'text-emerald-400' : state === 'FAILED' ? 'text-red-400' : 'text-slate-400';

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs">
        <span className={stateColor}>{label ?? state}</span>
        <span>{Math.round(value)}%</span>
      </div>
      <Progress value={value} />
    </div>
  );
}
