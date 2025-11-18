'use client';

import { Radio } from 'lucide-react';

const MODELS = [
  {
    label: 'Qwen/Qwen3-4B-Instruct-2507',
    value: 'Qwen/Qwen3-4B-Instruct-2507',
    latency: 'Best for latency + cost'
  },
  {
    label: 'meta-llama/Llama-3.1-70B-Instruct',
    value: 'meta-llama/Llama-3.1-70B-Instruct',
    latency: 'Maximum accuracy'
  }
];

interface ModelSelectProps {
  value: string;
  onChange: (model: string) => void;
}

export default function ModelSelect({ value, onChange }: ModelSelectProps) {
  return (
    <div className="space-y-3">
      {MODELS.map((model) => (
        <button
          key={model.value}
          className={`flex w-full items-center justify-between rounded-2xl border px-4 py-3 text-left transition ${
            value === model.value ? 'border-emerald-400 bg-emerald-500/10' : 'border-slate-800 bg-slate-900/60'
          }`}
          onClick={() => onChange(model.value)}
          type="button"
        >
          <div>
            <p className="text-sm font-medium text-slate-50">{model.label}</p>
            <p className="text-xs text-slate-400">{model.latency}</p>
          </div>
          <Radio className={`h-5 w-5 ${value === model.value ? 'text-emerald-400' : 'text-slate-500'}`} />
        </button>
      ))}
    </div>
  );
}
