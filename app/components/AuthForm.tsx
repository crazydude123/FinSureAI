'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { createSupabaseBrowserClient } from '@/lib/supabaseClient';
import { Button } from './ui/button';
import { Input } from './ui/input';

export default function AuthForm() {
  const router = useRouter();
  const [isSignUp, setIsSignUp] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    
    const formData = new FormData(event.currentTarget);
    const email = String(formData.get('email'));
    const password = String(formData.get('password'));
    setLoading(true);
    setError(null);
    setSuccessMessage(null);

    try {
      // Create client fresh for each request to ensure proper initialization
      const supabase = createSupabaseBrowserClient();
      
      if (!supabase || !supabase.auth) {
        setLoading(false);
        setError('Supabase client is not configured. Please check your environment variables.');
        return;
      }

      let authError = null;
      let session = null;
      let user = null;
      
      if (isSignUp) {
        const result = await supabase.auth.signUp({ email, password });
        authError = result.error;
        session = result.data.session;
        user = result.data.user;
        
        console.log('Sign up result:', { error: authError, hasSession: !!session, hasUser: !!user });
        
        // If sign up succeeds but no session (email confirmation required)
        if (!authError && !session && user) {
          setLoading(false);
          setError(null);
          setSuccessMessage('Account created! Please check your email to confirm your account, then sign in.');
          // Switch to sign in mode
          setIsSignUp(false);
          return;
        }
      } else {
        const result = await supabase.auth.signInWithPassword({ email, password });
        authError = result.error;
        session = result.data.session;
        console.log('Sign in result:', { error: authError, hasSession: !!session });
      }

      if (authError) {
        setLoading(false);
        setError(authError.message);
        return;
      }

      // Only redirect if we have a session
      if (session) {
        router.push('/dashboard');
        router.refresh();
      } else {
        setLoading(false);
        setError('Authentication succeeded but no session was created. Please try signing in.');
      }
    } catch (err) {
      setLoading(false);
      const errorMessage = err instanceof Error ? err.message : 'An unexpected error occurred';
      setError(errorMessage);
      console.error('Auth error:', err);
    }
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
        {successMessage ? <p className="text-sm text-green-400">{successMessage}</p> : null}
        <Button className="w-full" disabled={loading} type="submit">
          {loading ? 'Please wait…' : isSignUp ? 'Create account' : 'Sign in'}
        </Button>
      </form>
      <p className="mt-4 text-center text-sm text-slate-500">
        {isSignUp ? 'Already have an account?' : "Need an account?"}{' '}
        <button 
          className="text-brand" 
          onClick={() => {
            setIsSignUp((prev) => !prev);
            setError(null);
            setSuccessMessage(null);
          }}
        >
          {isSignUp ? 'Sign in' : 'Sign up'}
        </button>
      </p>
    </div>
  );
}
