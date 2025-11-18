import { describe, expect, it, vi } from 'vitest';
import { createFinetuneRecord } from '@/app/api/finetune/route';

vi.mock('@/lib/modalClient', () => ({
  triggerFinetuneJob: vi.fn().mockResolvedValue({ job_id: 'job_test_1' })
}));

describe('finetune job trigger', () => {
  it('stores job metadata and returns the job id', async () => {
    const insert = vi.fn().mockResolvedValue({});
    const mockClient = { from: () => ({ insert }) } as any;
    const payload = await createFinetuneRecord(
      {
        dataset_url: 'https://supabase.storage/datasets/demo.jsonl',
        model_name: 'Qwen/Qwen3-4B-Instruct-2507',
        user_id: 'user_1'
      },
      mockClient
    );

    expect(payload.job_id).toBe('job_test_1');
    expect(insert).toHaveBeenCalled();
  });
});
