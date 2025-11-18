import { defineConfig } from 'vitest/config';
import path from 'path';

process.env.SUPABASE_URL ??= 'https://test.supabase.co';
process.env.SUPABASE_ANON_KEY ??= 'public-anon-key';
process.env.SUPABASE_SERVICE_ROLE_KEY ??= 'service-role-key';
process.env.NEXT_PUBLIC_SUPABASE_URL ??= process.env.SUPABASE_URL;
process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ??= process.env.SUPABASE_ANON_KEY;
process.env.MODAL_TOKEN_ID ??= 'test-modal-id';
process.env.MODAL_TOKEN_SECRET ??= 'test-modal-secret';
process.env.SUPABASE_BUCKET_NAME ??= 'datasets';
process.env.JWT_SECRET ??= 'test-jwt-secret-value-123';

export default defineConfig({
  test: {
    environment: 'node',
    globals: true,
    include: ['tests/**/*.test.ts']
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, '.')
    }
  }
});
