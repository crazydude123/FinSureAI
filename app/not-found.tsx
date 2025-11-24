import Link from 'next/link';
import { Button } from './components/ui/button';

export const dynamic = 'force-dynamic';

export default function NotFound() {
    return (
        <section className="flex flex-col items-center gap-6 text-center py-20">
            <div className="space-y-4">
                <h1 className="text-6xl font-bold text-slate-50">404</h1>
                <h2 className="text-2xl font-semibold text-slate-300">Page Not Found</h2>
                <p className="text-slate-400">
                    The page you're looking for doesn't exist or has been moved.
                </p>
            </div>
            <Button asChild>
                <Link href="/">Go Home</Link>
            </Button>
        </section>
    );
}
