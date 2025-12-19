import { NextResponse } from 'next/server';
import type { SupabaseClient } from '@supabase/supabase-js';
import { runInference } from '@/lib/openpipeClient';  // Changed from modalClient
import { createSupabaseServiceClient } from '@/lib/supabaseAdmin';
import type { InferenceRequestPayload } from '@/lib/types';

async function resolveEndpoint(jobId: string, supabase?: SupabaseClient) {
  const client = supabase ?? createSupabaseServiceClient();
  const { data } = await client.from('jobs').select('endpoint_url').eq('job_id', jobId).single();
  return data?.endpoint_url ?? null;
}

async function callInference(payload: InferenceRequestPayload, supabase?: SupabaseClient) {
  const endpointUrl = payload.endpoint_url ?? (payload.job_id ? await resolveEndpoint(payload.job_id, supabase) : null);
  if (!endpointUrl) {
    throw new Error('Endpoint not ready yet');
  }
  const response = await runInference(endpointUrl, payload.message);
  return response.output ?? response.message ?? response;
}

function createTextStream(message: string) {
  const encoder = new TextEncoder();
  const words = message.split(/(\s+)/);
  let index = 0;

  return new ReadableStream({
    pull(controller) {
      if (index >= words.length) {
        controller.close();
        return;
      }
      const chunk = words[index] ?? '';
      controller.enqueue(encoder.encode(chunk));
      index += 1;
    }
  });
}

export async function POST(request: Request) {
  const payload = (await request.json()) as InferenceRequestPayload;
  const shouldStream = Boolean(payload.stream);

  if (!payload.message) {
    return NextResponse.json({ error: 'message is required' }, { status: 400 });
  }

  try {
    const message = await callInference(payload);
    if (shouldStream) {
      return new Response(createTextStream(String(message)), {
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'Cache-Control': 'no-store'
        }
      });
    }
    return NextResponse.json({ message });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Inference failed';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
