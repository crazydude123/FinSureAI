import { NextResponse } from 'next/server';
import crypto from 'crypto';
import type { SupabaseClient } from '@supabase/supabase-js';
import { createSupabaseServiceClient } from '@/lib/supabaseAdmin';
import { getServerEnv } from '@/lib/env';

const env = getServerEnv();
const ALLOWED_EXTENSIONS = ['json', 'jsonl', 'zip'];
const MAX_FILE_BYTES = 200 * 1024 * 1024; // 200MB

function ensureValidDataset(file: File) {
  const extension = file.name.split('.').pop()?.toLowerCase();
  if (!extension || !ALLOWED_EXTENSIONS.includes(extension)) {
    throw new Error('Only .json, .jsonl, or .zip datasets are supported.');
  }
  if (file.size > MAX_FILE_BYTES) {
    throw new Error('Dataset exceeds the 200MB upload limit.');
  }
}

function buildObjectPath(filename: string) {
  const normalized = filename.toLowerCase().replace(/[^a-z0-9._-]/g, '-');
  const extension = normalized.split('.').pop();
  return `uploads/${crypto.randomUUID()}-${normalized}${extension ? '' : '.jsonl'}`;
}

async function uploadBufferToSupabase(buffer: Buffer, filename: string, contentType: string, supabase?: SupabaseClient) {
  const objectPath = buildObjectPath(filename);
  const client = supabase ?? createSupabaseServiceClient();
  const { data, error } = await client.storage.from(env.SUPABASE_BUCKET_NAME).upload(objectPath, buffer, {
    upsert: false,
    contentType: contentType || 'application/octet-stream'
  });

  if (error) {
    throw new Error(error.message);
  }

  const { data: publicUrlData } = client.storage.from(env.SUPABASE_BUCKET_NAME).getPublicUrl(data.path);
  return { publicUrl: publicUrlData.publicUrl, path: data.path };
}

export async function POST(request: Request) {
  const formData = await request.formData();
  const file = formData.get('file') as File | null;

  if (!file) {
    return NextResponse.json({ error: 'File missing' }, { status: 400 });
  }

  try {
    ensureValidDataset(file);
    const buffer = Buffer.from(await file.arrayBuffer());
    const payload = await uploadBufferToSupabase(buffer, file.name, file.type);
    return NextResponse.json(payload);
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : 'Upload failed';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
