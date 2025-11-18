import { NextResponse } from 'next/server';
import type { SupabaseClient } from '@supabase/supabase-js';
import { triggerFinetuneJob } from '@/lib/modalClient';
import { createSupabaseServiceClient } from '@/lib/supabaseAdmin';
import type { TrainRequestPayload } from '@/lib/types';

export async function createFinetuneRecord(payload: TrainRequestPayload, supabase?: SupabaseClient) {
  const { dataset_url: datasetUrl, model_name: modelName, user_id: userId } = payload;
  const { job_id } = await triggerFinetuneJob({ datasetUrl, modelName, userId });
  const client = supabase ?? createSupabaseServiceClient();
  await client.from('jobs').insert({ job_id, user_id: userId, dataset_url: datasetUrl, model_name: modelName, status: 'RUNNING' });
  return { job_id };
}

export async function POST(request: Request) {
  const payload = (await request.json()) as TrainRequestPayload;

  if (!payload.dataset_url || !payload.model_name || !payload.user_id) {
    return NextResponse.json({ error: 'Missing payload' }, { status: 400 });
  }

  try {
    new URL(payload.dataset_url);
  } catch {
    return NextResponse.json({ error: 'dataset_url must be a valid URL' }, { status: 400 });
  }

  try {
    const job = await createFinetuneRecord(payload);
    return NextResponse.json(job);
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Unable to start Modal job' }, { status: 500 });
  }
}
