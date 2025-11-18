# FinSureAI

FinSureAI is a full-stack platform for uploading proprietary financial datasets, triggering GPU finetuning jobs on Modal, and chatting with the resulting Qwen or Llama checkpoints. The frontend runs on Next.js (App Router) + Tailwind + shadcn UI, authentication and storage are handled by Supabase, and long-running training/inference workloads live entirely inside Modal serverless functions.

## Architecture

- **Frontend**: Next.js 14 App Router, Tailwind CSS, shadcn primitives, Supabase Auth (email/password), protected dashboard with drag-and-drop uploader, model selector, job progress, and chat UI.
- **APIs**: Vercel-ready serverless routes for dataset upload, finetune trigger, job status polling, and inference proxy.
- **Backend tooling**: Python notebooks, benchmarking scripts, and inference agents under `backend/` for experimentation.
- **Modal jobs**: Production GPU workflows defined under `modal/` for finetuning (LoRA + GRPO) and HTTPS inference endpoints.

```
FinSureAI/
├── app/                    # Next.js App Router structure
├── backend/                # Legacy research assets and datasets
├── lib/                    # Shared utilities (env, Supabase, Modal)
├── modal/                  # Modal finetune + inference jobs
├── tests/                  # Vitest suites exercising API helpers
└── README.md
```

## Getting Started

### 1. Install dependencies

```bash
pnpm install   # or npm install / yarn install
```

### 2. Configure environment variables

Duplicate `.env.local.example` into `.env.local` and add the credentials from Supabase, Modal, and your JWT secret. The schema is validated by `lib/env.ts` during builds to prevent missing keys on Vercel.

### 3. Run the development server

```bash
pnpm dev
```

Navigate to `http://localhost:3000`, create an account with Supabase Auth, upload a `.json/.jsonl/.zip` dataset, pick a base model, start finetuning, and chat with the deployed endpoint after it completes.

### 4. Execute smoke tests

Vitest ensures that dataset uploads, finetune triggers, and inference proxy logic behave as expected without hitting external services.

```bash
pnpm test
```

## Modal Jobs

- `modal/finetune_job.py`: Downloads the dataset from Supabase Storage, runs LoRA SFT followed by GRPO (ART) reinforcement learning, saves artifacts to a shared Modal Volume, and returns the inference endpoint URL.
- `modal/inference_job.py`: Serves the fine-tuned weights behind a Modal `web_endpoint`, supporting streaming prompts from the Next.js dashboard.
- `modal/reward.py`: Reward function used during GRPO, identical to the legacy implementation.
- `modal/Dockerfile` + `modal/requirements.txt`: Deterministic environment definition for standalone GPU containers.

Deploy from the repo root:

```bash
modal deploy modal/finetune_job.py
modal deploy modal/inference_job.py
```

## Supabase Schema

Create a `jobs` table to store Modal job metadata and endpoint URLs:

| Column        | Type      | Notes                           |
| ------------- | --------- | --------------------------------|
| `job_id`      | text (PK) | Modal job identifier            |
| `user_id`     | uuid      | Supabase user                   |
| `dataset_url` | text      | Public Supabase Storage URL     |
| `model_name`  | text      | Base model slug                 |
| `status`      | text      | RUNNING / COMPLETED / FAILED    |
| `endpoint_url`| text      | Modal inference endpoint        |

## Deployment

1. **Vercel**: Connect the repo, add the env vars from `.env.local.example`, and deploy. All API routes are serverless-compatible and rely solely on Supabase + Modal HTTP calls.
2. **Modal**: Deploy `finetune_job` and `inference_endpoint` functions. Provide the resulting endpoint URL via the job return payload.
3. **Supabase Storage**: Create a bucket named `datasets` (or update `SUPABASE_BUCKET_NAME`) with `public` access for inference downloads.

## Testing Matrix

| Test                                | Location                  | Purpose                                      |
| ----------------------------------- | ------------------------- | -------------------------------------------- |
| Dataset upload helper               | `tests/api/upload.test.ts`| Verifies `.jsonl` uploads reach Supabase     |
| Finetune job trigger + metadata     | `tests/api/finetune.test.ts` | Ensures Modal jobs are recorded in DB    |
| Inference proxy                     | `tests/api/inference.test.ts` | Validates endpoint forwarding logic    |

## Scripts

- `pnpm dev` – start Next.js dev server
- `pnpm build && pnpm start` – production build
- `pnpm lint` – run ESLint
- `pnpm test` – run Vitest suite

## Notes

- Never store Supabase service keys on the client. All admin calls go through server routes.
- Modal workloads are isolated; Vercel doesn’t execute Python.
- Tailwind + shadcn UI components live under `app/components/ui` and power FileUpload, ModelSelect, TrainButton, ProgressBar, and ChatBox.
- Existing research notebooks and scripts remain untouched under `backend/` for reference.
