import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabaseServer';
import DashboardClient from './DashboardClient';

export default async function DashboardPage() {
  const supabase = await createSupabaseServerClient();
  const {
    data: { session }
  } = await supabase.auth.getSession();

  if (!session) {
    redirect('/');
  }

  return (
    <section className="space-y-6">
      <div>
        <p className="text-xs uppercase tracking-widest text-emerald-300">Control room</p>
        <h1 className="text-3xl font-semibold text-slate-50">Manage datasets, training jobs, and inference.</h1>
      </div>
      <DashboardClient userId={session.user.id} />
    </section>
  );
}
