import { createClient } from '@supabase/supabase-js'
import { createBrowserClient } from '@supabase/ssr'

function getSupabaseUrl() {
    const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
    if (!url || url.trim() === '' || url === 'your_supabase_project_url') {
        throw new Error(
            'Missing or invalid NEXT_PUBLIC_SUPABASE_URL environment variable. ' +
            'Please check your .env.local file and ensure it contains a valid Supabase project URL.'
        );
    }
    return url.trim();
}

function getSupabaseAnonKey() {
    const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
    if (!key || key.trim() === '' || key === 'your_supabase_anon_key') {
        throw new Error(
            'Missing or invalid NEXT_PUBLIC_SUPABASE_ANON_KEY environment variable. ' +
            'Please check your .env.local file and ensure it contains a valid Supabase anon key.'
        );
    }
    return key.trim();
}

let supabaseInstance: ReturnType<typeof createClient> | null = null;

export function getSupabaseClient() {
    if (!supabaseInstance) {
        const url = getSupabaseUrl();
        const key = getSupabaseAnonKey();
        supabaseInstance = createClient(url, key);
    }
    return supabaseInstance;
}

export const supabase = getSupabaseClient();

export function createSupabaseBrowserClient() {
    const url = getSupabaseUrl();
    const key = getSupabaseAnonKey();
    
    if (typeof window === 'undefined') {
        throw new Error('createSupabaseBrowserClient can only be called in the browser');
    }
    
    // Use createBrowserClient from @supabase/ssr for proper cookie handling
    // This is the recommended approach for Next.js App Router
    return createBrowserClient(url, key);
}
