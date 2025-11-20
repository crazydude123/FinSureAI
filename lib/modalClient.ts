interface FinetuneJobParams {
    datasetUrl: string;
    modelName: string;
    userId: string;
}

export async function triggerFinetuneJob(params: FinetuneJobParams) {
    // Mock implementation
    console.log('Triggering mock finetune job with params:', params);
    return { job_id: `mock-job-${Date.now()}` };
}

export async function runInference(endpointUrl: string, message: string) {
    // Mock implementation
    console.log('Running mock inference on', endpointUrl, 'with message:', message);
    return { message: `Mock response to: ${message}`, output: undefined };
}

export async function getJobStatus(jobId: string) {
    // Mock implementation
    console.log('Getting mock job status for:', jobId);
    return {
        state: 'COMPLETED',
        status: 'COMPLETED',
        progress: 100,
        message: 'Job completed successfully',
        status_text: 'Job completed successfully',
        logs: ['Job started', 'Training...', 'Completed'],
        result: { endpoint_url: 'https://mock-endpoint.modal.run' }
    };
}
