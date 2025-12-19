import { NextResponse } from 'next/server';
import type { SupabaseClient } from '@supabase/supabase-js';
import { triggerFinetuneJob } from '@/lib/openpipeClient';  // Changed from modalClient
import { createSupabaseServiceClient } from '@/lib/supabaseAdmin';
import type { TrainRequestPayload } from '@/lib/types';

async function createFinetuneRecord(payload: TrainRequestPayload, supabase?: SupabaseClient) {
  const { dataset_url: datasetUrl, model_name: modelName, user_id: userId } = payload;

  console.log('[API] Triggering OpenPipe fine-tuning job');
  console.log('[API] Dataset:', datasetUrl);
  console.log('[API] Model:', modelName);
  console.log('[API] User:', userId);

  const { job_id } = await triggerFinetuneJob({ datasetUrl, modelName, userId });

  console.log('[API] Job created:', job_id);

  const client = supabase ?? createSupabaseServiceClient();
  await client.from('jobs').insert({
    job_id,
    user_id: userId,
    dataset_url: datasetUrl,
    model_name: modelName,
    status: 'RUNNING',
    created_at: new Date().toISOString(),
  });

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
    console.error('[API] Error starting OpenPipe job:', error);
    return NextResponse.json(
      {
        error: 'Unable to start training',
        details: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}
