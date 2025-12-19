import { NextResponse } from 'next/server';
import { getJobStatus } from '@/lib/openpipeClient';
import { createSupabaseServiceClient } from '@/lib/supabaseAdmin';

export async function GET(
    request: Request,
    { params }: { params: { id: string } }
) {
    const jobId = params.id;

    if (!jobId) {
        return NextResponse.json({ error: 'Job ID required' }, { status: 400 });
    }

    try {
        console.log(`[API] Fetching status for job: ${jobId}`);

        // Get status from OpenPipe/W&B
        const status = await getJobStatus(jobId);

        console.log(`[API] Job ${jobId} status:`, status.state, `(${status.progress}%)`);

        // Update Supabase with latest status
        const supabase = createSupabaseServiceClient();
        const { error: updateError } = await supabase
            .from('jobs')
            .update({
                status: status.state,
                progress: status.progress,
                updated_at: new Date().toISOString(),
                // Store result if completed
                ...(status.result && {
                    result: status.result,
                }),
            })
            .eq('job_id', jobId);

        if (updateError) {
            console.error('[API] Failed to update job status in database:', updateError);
        }

        return NextResponse.json(status);
    } catch (error) {
        console.error(`[API] Error fetching job status for ${jobId}:`, error);
        return NextResponse.json(
            {
                error: 'Failed to fetch job status',
                details: error instanceof Error ? error.message : 'Unknown error',
            },
            { status: 500 }
        );
    }
}
