export function getServerEnv() {
    return {
        SUPABASE_BUCKET_NAME: process.env.SUPABASE_BUCKET || 'datasets',
    };
}
