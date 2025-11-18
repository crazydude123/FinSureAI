import Link from 'next/link';
import { Button } from './ui/button';
import { createSupabaseServerClient } from '@/lib/supabaseClient';
import SignOutButton from './SignOutButton';

export default async function Navbar() {
  let email: string | null = null;

  try {
    const supabase = await createSupabaseServerClient();
    const { data } = await supabase.auth.getUser();
    email = data.user?.email ?? null;
  } catch (error) {
    console.error('Unable to load auth state', error);
  }

  return (
    <header className="border-b border-slate-900/70 bg-slate-950/70 backdrop-blur-md">
      <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-4 py-4">
        <Link href="/" className="text-lg font-semibold text-slate-100">
          FinSureAI
        </Link>
        <div className="flex items-center gap-3 text-sm text-slate-400">
          {email ? <span className="hidden text-xs text-slate-500 sm:inline">{email}</span> : null}
          {email ? <SignOutButton /> : (
            <Button asChild variant="outline">
              <Link href="/">Sign in</Link>
            </Button>
          )}
        </div>
      </div>
    </header>
  );
}
