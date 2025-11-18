'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { createSupabaseBrowserClient } from '@/lib/supabaseClient';
import { Button } from './ui/button';
import { Input } from './ui/input';

export default function AuthForm() {
  const supabase = createSupabaseBrowserClient();
  const router = useRouter();
  const [isSignUp, setIsSignUp] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const email = String(formData.get('email'));
    const password = String(formData.get('password'));
    setLoading(true);
    setError(null);

    const fn = isSignUp ? supabase.auth.signUp : supabase.auth.signInWithPassword;
    const { error: authError } = await fn({ email, password });

    setLoading(false);
    if (authError) {
      setError(authError.message);
      return;
    }

    router.refresh();
  }

  return (
    <div className="w-full max-w-md rounded-3xl border border-slate-900/60 bg-slate-900/60 p-8 shadow-xl shadow-black/40">
      <form className="space-y-4" onSubmit={handleSubmit}>
        <div>
          <label className="text-sm text-slate-400" htmlFor="email">
            Email
          </label>
          <Input id="email" name="email" type="email" required disabled={loading} placeholder="alex@company.com" />
        </div>
        <div>
          <label className="text-sm text-slate-400" htmlFor="password">
            Password
          </label>
          <Input id="password" name="password" type="password" minLength={8} required disabled={loading} placeholder="••••••••" />
        </div>
        {error ? <p className="text-sm text-red-400">{error}</p> : null}
        <Button className="w-full" disabled={loading} type="submit">
          {loading ? 'Please wait…' : isSignUp ? 'Create account' : 'Sign in'}
        </Button>
      </form>
      <p className="mt-4 text-center text-sm text-slate-500">
        {isSignUp ? 'Already have an account?' : "Need an account?"}{' '}
        <button className="text-brand" onClick={() => setIsSignUp((prev) => !prev)}>
          {isSignUp ? 'Sign in' : 'Sign up'}
        </button>
      </p>
    </div>
  );
}
