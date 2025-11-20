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

Copy `.env.example` to `.env.local` and add the credentials from Supabase and Modal:

```bash
cp .env.example .env.local
```

Then edit `.env.local` with your actual credentials.

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

### Prerequisites

Before deploying, ensure you have:
- A Supabase account with a project created
- A Modal account with API credentials
- A Vercel account (free tier works)
- A GitHub account

### Step 1: Supabase Setup

1. Create a new Supabase project at [supabase.com](https://supabase.com)
2. Create a `jobs` table with the following schema:

```sql
CREATE TABLE jobs (
  job_id TEXT PRIMARY KEY,
  user_id UUID NOT NULL,
  dataset_url TEXT NOT NULL,
  model_name TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'RUNNING',
  endpoint_url TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);
```

3. Create a storage bucket named `datasets`:
   - Go to Storage in your Supabase dashboard
   - Create a new bucket called `datasets`
   - Set it to `public` access for inference downloads

4. Note down your credentials:
   - `NEXT_PUBLIC_SUPABASE_URL`: Project URL from Settings > API
   - `NEXT_PUBLIC_SUPABASE_ANON_KEY`: Anon/public key from Settings > API
   - `SUPABASE_SERVICE_ROLE_KEY`: Service role key from Settings > API (keep this secret!)

### Step 2: Modal Setup

1. Sign up at [modal.com](https://modal.com)
2. Install Modal CLI: `pip install modal`
3. Authenticate: `modal token new`
4. Deploy the Modal functions:

```bash
cd modal
modal deploy finetune_job.py
modal deploy inference_job.py
```

5. Note down your API credentials from the Modal dashboard

### Step 3: GitHub Repository

1. Initialize git (if not already done):

```bash
git init
git add .
git commit -m "Initial commit"
```

2. Create a new repository on GitHub
3. Push your code:

```bash
git remote add origin https://github.com/your-username/finsureai.git
git branch -M main
git push -u origin main
```

### Step 4: Deploy to Vercel

1. Go to [vercel.com](https://vercel.com) and sign in
2. Click "New Project" and import your GitHub repository
3. Configure your project:
   - **Framework Preset**: Next.js
   - **Build Command**: `pnpm build` (or leave default)
   - **Install Command**: `pnpm install` (or leave default)

4. Add environment variables (from `.env.example`):
   - `NEXT_PUBLIC_SUPABASE_URL`
   - `NEXT_PUBLIC_SUPABASE_ANON_KEY`
   - `SUPABASE_SERVICE_ROLE_KEY`
   - `SUPABASE_BUCKET` (set to `datasets`)
   - `MODAL_API_TOKEN`
   - `MODAL_API_SECRET`

5. Click "Deploy" and wait for the build to complete

### Step 5: Post-Deployment

1. Visit your deployed URL
2. Create a test account
3. Upload a sample `.jsonl` dataset
4. Start a finetuning job and monitor progress
5. Chat with your fine-tuned model once training completes

### Continuous Deployment

Vercel automatically deploys:
- **Production**: Every push to the `main` branch
- **Preview**: Every pull request

### Troubleshooting

- **Build failures**: Check environment variables are set correctly
- **Database errors**: Verify Supabase credentials and table schema
- **Modal errors**: Ensure Modal functions are deployed and credentials are correct
- **Upload issues**: Check Supabase storage bucket permissions

### Local Development vs Production

For local development:
1. Copy `.env.example` to `.env.local`
2. Fill in your credentials
3. Run `pnpm dev`

For production, all environment variables are managed through Vercel's dashboard.

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
