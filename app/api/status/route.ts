import { NextResponse } from 'next/server';
import type { SupabaseClient } from '@supabase/supabase-js';
import { getJobStatus } from '@/lib/modalClient';
import { createSupabaseServiceClient } from '@/lib/supabaseAdmin';
import type { JobStatusPayload } from '@/lib/types';

function extractLogs(status: Record<string, any>): string[] {
  if (Array.isArray(status.logs)) {
    return status.logs.map((entry) => String(entry));
  }
  if (Array.isArray(status.task_history)) {
    return status.task_history
      .map((task) => task.message ?? task.status ?? task.description)
      .filter(Boolean)
      .map(String);
  }
  if (typeof status.log === 'string') {
    return status.log.split('\n').filter(Boolean);
  }
  return [];
}

function deriveDatasetFilename(url?: string | null) {
  if (!url) return null;
  try {
    const decoded = decodeURIComponent(url);
    const parts = decoded.split('/');
    return parts[parts.length - 1] ?? null;
  } catch {
    return null;
  }
}

async function fetchJobStatus(jobId: string, supabase?: SupabaseClient): Promise<JobStatusPayload> {
  const client = supabase ?? createSupabaseServiceClient();
  const { data: jobRecord } = await client
    .from('jobs')
    .select('dataset_url, model_name, endpoint_url, status')
    .eq('job_id', jobId)
    .maybeSingle();
  const status = await getJobStatus(jobId);
  const payload: JobStatusPayload = {
    state: (status.state ?? status.status ?? 'RUNNING').toUpperCase(),
    progress: status.progress ?? 10,
    message: status.message ?? status.status_text,
    logs: extractLogs(status),
    endpointUrl: status.result?.endpoint_url ?? jobRecord?.endpoint_url ?? null,
    datasetUrl: jobRecord?.dataset_url ?? null,
    datasetFileName: deriveDatasetFilename(jobRecord?.dataset_url),
    modelName: jobRecord?.model_name ?? null
  } as JobStatusPayload;

  if (payload.state === 'COMPLETED' && payload.endpointUrl && jobRecord?.endpoint_url !== payload.endpointUrl) {
    await client.from('jobs').update({ status: 'COMPLETED', endpoint_url: payload.endpointUrl }).eq('job_id', jobId);
  }

  if (payload.state === 'FAILED' && jobRecord?.status !== 'FAILED') {
    await client.from('jobs').update({ status: 'FAILED' }).eq('job_id', jobId);
  }

  return payload;
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const jobId = searchParams.get('job_id');

  if (!jobId) {
    return NextResponse.json({ error: 'job_id missing' }, { status: 400 });
  }

  try {
    const payload = await fetchJobStatus(jobId);
    return NextResponse.json(payload);
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Unable to load job status' }, { status: 500 });
  }
}
