export interface TrainRequestPayload {
    dataset_url: string;
    model_name: string;
    user_id: string;
}

export interface InferenceRequestPayload {
    endpoint_url?: string;
    job_id?: string;
    message: string;
    stream?: boolean;
}

export interface JobStatusPayload {
    state: string;
    progress: number;
    message?: string;
    logs: string[];
    endpointUrl?: string | null;
    datasetUrl?: string | null;
    datasetFileName?: string | null;
    modelName?: string | null;
}
