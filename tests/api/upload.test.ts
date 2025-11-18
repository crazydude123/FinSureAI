import { describe, expect, it, vi } from 'vitest';
import { uploadBufferToSupabase } from '@/app/api/upload/route';

const mockClient = {
  storage: {
    from: () => ({
      upload: vi.fn().mockResolvedValue({ data: { path: 'uploads/example.jsonl' }, error: null }),
      getPublicUrl: () => ({ data: { publicUrl: 'https://cdn.supabase.co/uploads/example.jsonl' } })
    })
  }
} as any;

describe('dataset upload', () => {
  it('stores a .jsonl file and returns the public url', async () => {
    const payload = await uploadBufferToSupabase(Buffer.from('{"prompt":"hi"}\n'), 'mini.jsonl', 'application/json', mockClient);
    expect(payload.path).toContain('uploads/');
    expect(payload.publicUrl).toContain('https://');
  });
});
