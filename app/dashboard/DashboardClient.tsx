'use client';

import { useEffect, useMemo, useState } from 'react';
import FileUpload from '@/app/components/FileUpload';
import ModelSelect from '@/app/components/ModelSelect';
import TrainButton from '@/app/components/TrainButton';
import ProgressBar from '@/app/components/ProgressBar';
import ChatBox from '@/app/components/ChatBox';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/app/components/ui/card';
import type { JobStatusPayload } from '@/lib/types';

interface Props {
  userId: string;
}

export default function DashboardClient({ userId }: Props) {
  const [dataset, setDataset] = useState<{ fileName: string; publicUrl: string; size: number } | null>(null);
  const [model, setModel] = useState('Qwen/Qwen3-4B-Instruct-2507');
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<JobStatusPayload>({
    state: 'IDLE',
    progress: 0,
    logs: [],
    endpointUrl: null,
    datasetUrl: null,
    datasetFileName: null,
    modelName: null
  });
  const [error, setError] = useState<string | null>(null);
  const [launching, setLaunching] = useState(false);

  const canStart = Boolean(dataset && !launching && status.state !== 'RUNNING');

  useEffect(() => {
    if (!jobId) {
      return;
    }

    setStatus((prev) => ({ ...prev, state: 'RUNNING' }));
    const interval = setInterval(async () => {
      const response = await fetch(`/api/status?job_id=${jobId}`);
      if (!response.ok) {
        setError('Unable to fetch job status');
        return;
      }
      const payload = await response.json();
      setStatus((prev) => ({
        ...prev,
        ...payload,
        state: payload.state,
        progress: payload.progress ?? 0,
        message: payload.message,
        logs: payload.logs ?? [],
        endpointUrl: payload.endpointUrl ?? prev.endpointUrl
      }));

      if (payload.state === 'COMPLETED' || payload.state === 'FAILED') {
        clearInterval(interval);
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [jobId]);

  const statusLabel = useMemo(() => {
    if (status.state === 'COMPLETED') return 'Model ready';
    if (status.state === 'FAILED') return 'Job failed';
    if (status.state === 'RUNNING') return status.message ?? 'Fine-tuning in progress';
    return 'Idle';
  }, [status]);

  async function startTraining() {
    if (!dataset) {
      setError('Upload a dataset first.');
      return;
    }
    setLaunching(true);
    setError(null);

    const res = await fetch('/api/finetune', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_url: dataset.publicUrl, model_name: model, user_id: userId })
    });

    setLaunching(false);

    if (!res.ok) {
      setError('Unable to start finetuning.');
      return;
    }

    const data = await res.json();
    setJobId(data.job_id);
    setStatus((prev) => ({
      ...prev,
      state: 'RUNNING',
      progress: 5,
      message: 'Queued on Modal',
      datasetFileName: dataset.fileName,
      datasetUrl: dataset.publicUrl,
      modelName: model
    }));
  }

  const activeDatasetName = status.datasetFileName ?? dataset?.fileName ?? 'Not selected';
  const activeModel = status.modelName ?? model;

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Dataset upload</CardTitle>
            <CardDescription>Store private datasets in Supabase Storage.</CardDescription>
          </CardHeader>
          <CardContent>
            <FileUpload onUploadComplete={setDataset} />
            {dataset ? (
              <p className="mt-4 text-xs text-emerald-400">Uploaded {dataset.fileName}</p>
            ) : null}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Model selection</CardTitle>
            <CardDescription>Pick a base model for LoRA + GRPO.</CardDescription>
          </CardHeader>
          <CardContent>
            <ModelSelect value={model} onChange={setModel} />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Training controls</CardTitle>
            <CardDescription>Launch OpenPipe job and monitor via W&B.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <TrainButton disabled={!canStart} loading={launching} onClick={startTraining} />
            <ProgressBar value={status.progress} state={status.state as any} label={statusLabel} />
            {error ? <p className="text-xs text-red-400">{error}</p> : null}
            <div className="rounded-xl border border-slate-900/60 bg-slate-950/40 p-3 text-xs text-slate-400">
              <p className="font-semibold uppercase tracking-wide text-slate-500">Job details</p>
              <ul className="mt-2 space-y-1">
                <li>
                  <span className="text-slate-500">Model:</span> {activeModel}
                </li>
                <li>
                  <span className="text-slate-500">Dataset:</span> {activeDatasetName}
                </li>
                <li>
                  <span className="text-slate-500">State:</span> {status.state}
                </li>
                <li>
                  <span className="text-slate-500">Message:</span> {status.message ?? '—'}
                </li>
                {jobId ? (
                  <li>
                    <span className="text-slate-500">Job ID:</span> {jobId}
                  </li>
                ) : null}
                {(status as any).metrics?.reward !== undefined ? (
                  <li>
                    <span className="text-slate-500">Reward:</span> {(status as any).metrics.reward.toFixed(3)}
                  </li>
                ) : null}
                {(status as any).metrics?.loss !== undefined ? (
                  <li>
                    <span className="text-slate-500">Loss:</span> {(status as any).metrics.loss.toFixed(4)}
                  </li>
                ) : null}
                {(status as any).wandbUrl ? (
                  <li>
                    <a
                      href={(status as any).wandbUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-400 hover:text-blue-300 underline"
                    >
                      View in W&B Dashboard →
                    </a>
                  </li>
                ) : null}
              </ul>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Training logs</CardTitle>
            <CardDescription>Live tail from OpenPipe/W&B.</CardDescription>
          </CardHeader>
          <CardContent>
            {status.logs && status.logs.length > 0 ? (
              <ul className="space-y-2 text-xs text-slate-400">
                {status.logs.slice(-5).map((log, index) => (
                  <li key={`${log}-${index}`} className="rounded-lg bg-slate-950/60 px-3 py-2">
                    {log}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-xs text-slate-500">Waiting for logs…</p>
            )}
          </CardContent>
        </Card>
      </div>
      <div className="flex flex-col gap-6">
        <Card className="flex-1">
          <CardHeader>
            <CardTitle>Chat with FinSureAI</CardTitle>
            <CardDescription>Test the fine-tuned model directly.</CardDescription>
          </CardHeader>
          <CardContent className="h-[520px]">
            <ChatBox endpointUrl={status.endpointUrl} jobId={jobId} />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
