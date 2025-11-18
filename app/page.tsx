import { redirect } from 'next/navigation';
import AuthForm from './components/AuthForm';
import { createSupabaseServerClient } from '@/lib/supabaseClient';

export default async function LandingPage() {
  const supabase = await createSupabaseServerClient();
  const {
    data: { session }
  } = await supabase.auth.getSession();

  if (session) {
    redirect('/dashboard');
  }

  return (
    <section className="flex flex-col items-center gap-8 text-center">
      <div className="max-w-3xl space-y-4">
        <p className="inline-flex items-center rounded-full border border-emerald-500/50 bg-emerald-500/10 px-4 py-1 text-xs uppercase tracking-widest text-emerald-300">
          FinSureAI Builder
        </p>
        <h1 className="text-4xl font-semibold leading-tight text-slate-50 sm:text-5xl">
          Fine-tune Qwen or Llama with your financial data in minutes.
        </h1>
        <p className="text-lg text-slate-300">
          Upload curated datasets, launch Modal GPU jobs, monitor progress, and chat with the customized model directly in the dashboard.
        </p>
      </div>
      <AuthForm />
    </section>
  );
}
