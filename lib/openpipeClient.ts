/**
 * OpenPipe Client Library
 * 
 * This client wraps Docker Compose commands to trigger OpenPipe ART training
 * and provides functions for inference and job status monitoring via W&B API.
 */

import { spawn, ChildProcess } from 'child_process';
import path from 'path';
import { OpenAI } from 'openai';

// ============================================================================
// Types
// ============================================================================

export interface FinetuneJobParams {
    datasetUrl: string;
    modelName: string;
    userId: string;
    numSteps?: number;
    rolloutsPerStep?: number;
}

export interface JobStatus {
    state: 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED';
    progress: number;
    logs: string[];
    metrics?: {
        reward?: number;
        loss?: number;
        step?: number;
    };
    result?: {
        model_name?: string;
        inference_url?: string;
        wandb_url?: string;
    };
}

export interface InferenceResult {
    output: string;
    usage?: {
        prompt_tokens: number;
        completion_tokens: number;
        total_tokens: number;
    };
}

// ============================================================================
// Configuration
// ============================================================================

const DOCKER_COMPOSE_FILE = path.join(process.cwd(), 'openpipe/docker-compose.yml');
const WANDB_API_BASE = 'https://api.wandb.ai';

// Store active training processes
const activeProcesses = new Map<string, ChildProcess>();

// ============================================================================
// Main Functions
// ============================================================================

/**
 * Trigger a fine-tuning job using OpenPipe ART
 * 
 * @param params - Fine-tuning parameters
 * @returns Job ID for tracking
 */
export async function triggerFinetuneJob(params: FinetuneJobParams): Promise<{ job_id: string }> {
    const job_id = `openpipe-${Date.now()}-${params.userId}`;

    console.log(`[OpenPipe] Starting fine-tuning job: ${job_id}`);
    console.log(`[OpenPipe] Dataset: ${params.datasetUrl}`);
    console.log(`[OpenPipe] Model: ${params.modelName}`);

    // Build Docker Compose command
    const args = [
        'compose',
        '-f', DOCKER_COMPOSE_FILE,
        'run',
        '--rm',
        '--name', job_id,
        'openpipe-training',
        'python3', 'finetune_job.py',
        params.datasetUrl,
        params.modelName,
        params.userId,
    ];

    // Add optional parameters
    if (params.numSteps) {
        args.push(params.numSteps.toString());
    }
    if (params.rolloutsPerStep) {
        args.push(params.rolloutsPerStep.toString());
    }

    // Spawn Docker process
    const dockerProcess = spawn('docker', args, {
        cwd: process.cwd(),
        env: {
            ...process.env,
            WANDB_API_KEY: process.env.WANDB_API_KEY,
        },
    });

    // Store process reference
    activeProcesses.set(job_id, dockerProcess);

    // Handle stdout
    dockerProcess.stdout.on('data', (data) => {
        const output = data.toString();
        console.log(`[OpenPipe][${job_id}] ${output}`);
    });

    // Handle stderr
    dockerProcess.stderr.on('data', (data) => {
        const error = data.toString();
        console.error(`[OpenPipe][${job_id}] ERROR: ${error}`);
    });

    // Handle process exit
    dockerProcess.on('close', (code) => {
        console.log(`[OpenPipe][${job_id}] Process exited with code ${code}`);
        activeProcesses.delete(job_id);
    });

    // Handle process errors
    dockerProcess.on('error', (error) => {
        console.error(`[OpenPipe][${job_id}] Failed to start: ${error.message}`);
        activeProcesses.delete(job_id);
    });

    return { job_id };
}

/**
 * Get the status of a fine-tuning job from W&B API
 * 
 * @param jobId - Job ID returned from triggerFinetuneJob
 * @returns Current job status
 */
export async function getJobStatus(jobId: string): Promise<JobStatus> {
    try {
        // Check if process is still running locally
        const process = activeProcesses.get(jobId);
        const isRunningLocally = process && !process.killed;

        // Extract user ID from job ID (format: openpipe-timestamp-userId)
        const parts = jobId.split('-');
        const userId = parts.length >= 3 ? parts.slice(2).join('-') : 'default';

        // Query W&B API for run status
        // Note: W&B run name format is "finsureai-{userId}"
        const wandbProject = 'finsureai';
        const wandbRunName = `finsureai-${userId}`;

        if (!process.env.WANDB_API_KEY) {
            console.warn('[OpenPipe] WANDB_API_KEY not set, returning mock status');
            return {
                state: isRunningLocally ? 'RUNNING' : 'PENDING',
                progress: 0,
                logs: ['Training in progress...'],
            };
        }

        // Fetch run data from W&B
        const response = await fetch(
            `${WANDB_API_BASE}/api/v1/runs/${wandbProject}/${wandbRunName}`,
            {
                headers: {
                    'Authorization': `Bearer ${process.env.WANDB_API_KEY}`,
                },
            }
        );

        if (!response.ok) {
            // Run not found or API error - check local process
            if (isRunningLocally) {
                return {
                    state: 'RUNNING',
                    progress: 0,
                    logs: ['Training started, waiting for W&B sync...'],
                };
            }

            return {
                state: 'PENDING',
                progress: 0,
                logs: ['Job queued, not yet started'],
            };
        }

        const data = await response.json();

        // Parse W&B run data
        const state = parseWandbState(data.state);
        const currentStep = data.summary?.step || 0;
        const totalSteps = data.config?.num_steps || 20;
        const progress = Math.min(100, Math.round((currentStep / totalSteps) * 100));

        return {
            state,
            progress,
            logs: data.events?.slice(-10).map((e: any) => e.message) || [],
            metrics: {
                reward: data.summary?.['train/reward'],
                loss: data.summary?.['train/loss'],
                step: currentStep,
            },
            result: state === 'COMPLETED' ? {
                model_name: data.summary?.model_name,
                inference_url: data.summary?.inference_url,
                wandb_url: `https://wandb.ai/${wandbProject}/runs/${data.name}`,
            } : undefined,
        };
    } catch (error) {
        console.error(`[OpenPipe] Error fetching job status: ${error}`);

        // Fallback: check local process
        const process = activeProcesses.get(jobId);
        if (process && !process.killed) {
            return {
                state: 'RUNNING',
                progress: 0,
                logs: ['Training in progress...'],
            };
        }

        return {
            state: 'FAILED',
            progress: 0,
            logs: [`Error: ${error instanceof Error ? error.message : 'Unknown error'}`],
        };
    }
}

/**
 * Run inference using a fine-tuned model
 * 
 * @param endpointUrl - W&B inference endpoint URL
 * @param message - User message/prompt
 * @param modelName - Optional model name (defaults to 'latest')
 * @returns Model response
 */
export async function runInference(
    endpointUrl: string,
    message: string,
    modelName: string = 'latest'
): Promise<InferenceResult> {
    try {
        const client = new OpenAI({
            baseURL: endpointUrl,
            apiKey: process.env.WANDB_API_KEY || '',
        });

        const response = await client.chat.completions.create({
            model: modelName,
            messages: [
                {
                    role: 'system',
                    content: 'You are a financial expert. Provide accurate, concise answers based on your training.',
                },
                {
                    role: 'user',
                    content: message,
                },
            ],
            max_tokens: 512,
            temperature: 0.7,
        });

        return {
            output: response.choices[0].message.content || '',
            usage: response.usage ? {
                prompt_tokens: response.usage.prompt_tokens,
                completion_tokens: response.usage.completion_tokens,
                total_tokens: response.usage.total_tokens,
            } : undefined,
        };
    } catch (error) {
        console.error(`[OpenPipe] Inference error: ${error}`);
        throw new Error(`Inference failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
}

/**
 * Stop a running fine-tuning job
 * 
 * @param jobId - Job ID to stop
 * @returns Success status
 */
export async function stopJob(jobId: string): Promise<{ success: boolean }> {
    const process = activeProcesses.get(jobId);

    if (!process) {
        console.warn(`[OpenPipe] Job ${jobId} not found in active processes`);
        return { success: false };
    }

    console.log(`[OpenPipe] Stopping job: ${jobId}`);
    process.kill('SIGTERM');
    activeProcesses.delete(jobId);

    return { success: true };
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Parse W&B run state to our JobStatus state
 */
function parseWandbState(wandbState: string): JobStatus['state'] {
    switch (wandbState) {
        case 'running':
            return 'RUNNING';
        case 'finished':
            return 'COMPLETED';
        case 'failed':
        case 'crashed':
            return 'FAILED';
        default:
            return 'PENDING';
    }
}

/**
 * Get list of all active jobs
 */
export function getActiveJobs(): string[] {
    return Array.from(activeProcesses.keys());
}

/**
 * Clean up all active processes (useful for graceful shutdown)
 */
export function cleanup(): void {
    console.log(`[OpenPipe] Cleaning up ${activeProcesses.size} active processes`);

    for (const [jobId, process] of activeProcesses.entries()) {
        console.log(`[OpenPipe] Stopping job: ${jobId}`);
        process.kill('SIGTERM');
    }

    activeProcesses.clear();
}
