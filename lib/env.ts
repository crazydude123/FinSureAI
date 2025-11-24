export function getServerEnv() {
    return {
        SUPABASE_URL: process.env.NEXT_PUBLIC_SUPABASE_URL || '',
        SUPABASE_ANON_KEY: process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || '',
        SUPABASE_BUCKET_NAME: process.env.SUPABASE_BUCKET || 'datasets',
    };
}
