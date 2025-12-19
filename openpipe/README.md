# OpenPipe ART Fine-Tuning for FinSureAI

This directory contains the OpenPipe ART (Adaptive Reinforcement Training) setup for fine-tuning financial Q&A models using reinforcement learning.

---

## 🎯 What This Does

**OpenPipe ART** trains AI models using **reinforcement learning** instead of traditional supervised learning. The model learns by:
1. Generating answers to financial questions
2. Getting feedback (rewards) on answer quality
3. Improving over time to maximize rewards

**Think of it like**: Teaching a student by grading their practice tests, not just showing them the answers.

---

## ✅ What's Already Done

### **1. Complete Training Setup**
- ✅ Docker environment with CUDA support
- ✅ Python training script (`finetune_job.py`)
- ✅ Reward function for evaluating answers (`reward.py`)
- ✅ Dataset integration (`virattt/financial-qa-10K`)
- ✅ W&B logging and monitoring

### **2. Backend Integration**
- ✅ Next.js API routes to trigger training
- ✅ Real-time progress monitoring via W&B
- ✅ Dashboard displays training metrics
- ✅ Inference API for chatting with trained models

### **3. Project Structure**
```
openpipe/
├── finetune_job.py         # Main training script
├── reward.py               # Reward calculation logic
├── Dockerfile              # Docker environment
├── docker-compose.yml      # Docker configuration
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
├── .env                    # Your credentials (gitignored)
└── README.md               # This file
```

---

## 🚀 Quick Start

### **Prerequisites**
1. Docker Desktop installed and running
2. W&B account (free): https://wandb.ai/signup
3. Supabase project set up (for FinSureAI app)

### **Step 1: Configure Environment**
```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your W&B API key
nano .env
```

Get your W&B API key from: https://wandb.ai/authorize

### **Step 2: Build Docker Image**
```bash
docker compose build
```
⏱️ Takes 5-10 minutes on first run (downloads CUDA, PyTorch, etc.)

### **Step 3: Run Training**

**Option A: Quick Test (2 steps, ~10 minutes)**
```bash
docker compose run --rm openpipe-training python3 finetune_job.py \
  "virattt/financial-qa-10K" \
  "OpenPipe/Qwen3-14B-Instruct" \
  "test-user"
```

**Option B: Full Training (20 steps, 2-4 hours)**
```bash
docker compose up
```

**Option C: Via Web UI**
1. Start FinSureAI app: `cd .. && pnpm dev`
2. Go to dashboard
3. Upload dataset
4. Click "Start Fine-Tuning"

---

## 💰 Cost Breakdown

### **What's Free**
- ✅ OpenPipe ART code (open-source)
- ✅ W&B dashboard and logging
- ✅ Docker setup
- ✅ All the code we built

### **What Costs Money**
- ❌ GPU time for training: **~$2-12 per training run**
  - Uses W&B Training (serverless GPUs)
  - Automatically provisions and releases GPUs
  - You only pay for actual training time

- ❌ Inference API calls: **~$0.001-0.01 per request**
  - After training, model is hosted on W&B
  - Pay per API call (like OpenAI)
  - Much cheaper than OpenAI for your own models

### **Free Alternative**
Use your own GPU (if you have one):
1. Edit `finetune_job.py` line 204
2. Change `ServerlessBackend()` to `LocalBackend()`
3. Training runs on your GPU (free!)

---

## 📊 How It Works

### **Training Flow**
```
1. Load financial dataset (1000 Q&A pairs)
   ↓
2. Model generates answers
   ↓
3. Reward function scores answers
   - Semantic similarity
   - Key term matching
   - Length appropriateness
   - Hallucination detection
   ↓
4. Model learns from rewards
   ↓
5. Repeat for 20 steps
   ↓
6. Final model saved to W&B
```

### **Reward Function** (`reward.py`)
Evaluates answer quality based on:
- **60%**: Semantic similarity to ground truth
- **30%**: Key financial terms and numbers
- **10%**: Penalties for length, hallucinations, repetition

**Example**:
```python
compute_reward(
    "What was Apple's Q1 revenue?",
    "Apple's Q1 revenue was $119.58B",  # Good answer
    "Apple reported $119.58 billion"
)
# Returns: 0.85 (high reward)
```

---

## 📈 Monitoring Training

### **W&B Dashboard**
Visit: https://wandb.ai/finsureai

**You'll see**:
- 📊 Reward curve (should go up)
- 📉 Loss curve (should go down)
- 📝 Individual trajectories
- 💾 Model checkpoints
- ⏱️ Training time and GPU usage

### **FinSureAI Dashboard**
Your Next.js app shows:
- Current training step
- Progress percentage
- Latest reward and loss
- Link to W&B dashboard
- Training logs

---

## 🔧 Configuration

### **Environment Variables** (`.env`)
```bash
# Required
WANDB_API_KEY=your_wandb_key_here

# Optional
OPENPIPE_API_KEY=
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
```

### **Training Parameters** (`finetune_job.py`)
```python
num_steps = 20              # Number of training iterations
rollouts_per_step = 18      # Parallel rollouts per step
learning_rate = 1e-5        # Learning rate
TRAIN_SIZE = 1000           # Dataset size
```

### **Supported Models**
- `OpenPipe/Qwen3-14B-Instruct` (recommended)
- `Qwen/Qwen3-4B-Instruct-2507` (faster, smaller)
- `meta-llama/Llama-3.1-70B-Instruct` (largest, best quality)

---

## 🐛 Troubleshooting

### **Docker daemon not running**
```bash
# Start Docker Desktop app
# Wait for whale icon in menu bar
docker info  # Verify it's running
```

### **W&B API key not working**
```bash
# Get new key from https://wandb.ai/authorize
# Update .env file
# Rebuild: docker compose build
```

### **Training fails with GPU error**
- W&B Training handles GPUs automatically
- If using LocalBackend, ensure you have a GPU
- Check Docker has GPU access: `docker run --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi`

### **Out of memory**
- Reduce `rollouts_per_step` (try 12 instead of 18)
- Use smaller model (Qwen3-4B instead of 14B)
- Reduce `TRAIN_SIZE` (try 500 instead of 1000)

---

## 📋 What's Next

### **Immediate Next Steps**
1. ✅ **Fix Supabase Setup**
   - Create `.env.local` in project root
   - Add Supabase credentials
   - Restart dev server: `pnpm dev`

2. ✅ **Create W&B Account**
   - Sign up at https://wandb.ai/signup
   - Get API key from https://wandb.ai/authorize
   - Add to `openpipe/.env`

3. ✅ **Run Test Training**
   - Build Docker: `docker compose build`
   - Run quick test (2 steps)
   - Verify W&B logging works

### **After Testing Works**
4. **Run Full Training**
   - Train on complete dataset (20 steps)
   - Monitor W&B dashboard
   - Wait 2-4 hours for completion

5. **Test Inference**
   - Use trained model in chat
   - Compare with baseline model
   - Evaluate answer quality

6. **Production Deployment**
   - Deploy to cloud (AWS, GCP, etc.)
   - Set up automated training pipeline
   - Monitor model performance

### **Future Enhancements**
- **Unsloth Quantization**: Compress model for faster inference
- **Custom Datasets**: Train on your own financial data
- **Multi-Model Training**: Train multiple models simultaneously
- **A/B Testing**: Compare different reward functions

---

## 🎓 Learning Resources

### **OpenPipe ART**
- GitHub: https://github.com/openpipe-ai/art
- Docs: https://docs.openpipe.ai/features/art
- Example: 2048 game training notebook

### **Weights & Biases**
- Dashboard: https://wandb.ai
- Docs: https://docs.wandb.ai
- Training: https://docs.wandb.ai/guides/launch

### **Reinforcement Learning**
- GRPO (Group Relative Policy Optimization)
- Reward shaping for LLMs
- Policy gradient methods

---

## 📞 Support

**Issues?**
- Check troubleshooting section above
- Review W&B logs for errors
- Check Docker logs: `docker logs finsureai-openpipe`

**Questions?**
- OpenPipe Discord: https://discord.gg/openpipe
- W&B Community: https://wandb.ai/community

---

## 📊 Summary

**Status**: ✅ **Ready to Train**

**What You Have**:
- Complete training setup
- Docker environment
- Backend integration
- Dashboard monitoring

**What You Need**:
1. W&B API key (free)
2. Supabase credentials (for app)
3. Docker running

**Cost**: ~$2-12 per training run (or free with your own GPU)

**Time**: 2-4 hours for full training

**Result**: Fine-tuned financial Q&A model hosted on W&B

---

🚀 **Ready to start? Run `docker compose build` and let's train!**
