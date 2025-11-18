import { describe, expect, it, vi } from 'vitest';
import { callInference } from '@/app/api/inference/route';

vi.mock('@/lib/modalClient', () => ({
  runInference: vi.fn().mockResolvedValue({ output: 'The stock outlook is positive.' })
}));

describe('inference proxy', () => {
  it('forwards prompt to modal endpoint and returns completion', async () => {
    const message = await callInference({
      endpoint_url: 'https://modal.run/finsureai',
      message: 'Summarize Apple earnings.'
    });

    expect(message).toContain('positive');
  });
});
