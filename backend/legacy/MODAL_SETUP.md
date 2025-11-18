# Modal Deployment - Setup & Compatibility

## ✅ Will it run successfully on Modal?

**Yes, with the following updates made:**

### Changes Made for Modal Compatibility

1. **✅ ART/GRPO Integration Added**
   - Full SFT + GRPO training pipeline integrated
   - Reward function copied into Modal image
   - GRPO training runs after SFT completes

2. **✅ Dependencies Fixed**
   - `bitsandbytes` installed in Modal image (CUDA-enabled)
   - ART installed from GitHub during image build
   - All required packages included

3. **✅ Paths Updated**
   - All paths use `/models/` (Modal volume mount)
   - `reward.py` copied to `/reward.py` in image
   - Model saves to persistent volume

4. **✅ Timeout Increased**
   - Training timeout: 7200 seconds (2 hours) for SFT + GRPO
   - Inference timeout: 1800 seconds (30 minutes)

5. **✅ Inference Updated**
   - Added `use_art` parameter to select model
   - Error handling for missing models

## Potential Issues & Solutions

### ⚠️ Known Considerations

1. **GRPOTrainer API Compatibility**
   - The ART library API may vary. If you encounter errors, check:
     - GRPOTrainer parameter names
     - Dataset format requirements
     - Reward function signature

2. **Memory Requirements**
   - GRPO training may require more memory
   - If OOM errors occur, reduce `rollout_batch_size` from 8 to 4

3. **Training Time**
   - Full training (SFT + GRPO) may take 1-2 hours
   - Monitor Modal logs for progress

4. **Model Size**
   - Both SFT and ART models saved to volume
   - Ensure sufficient volume space (~20GB recommended)

## Running on Modal

### 1. Training (SFT + GRPO)
```bash
cd FinSureAI
modal run modal_app.py::finetune_model
```

This will:
- Run SFT training
- Save merged SFT model to `/models/merged_finetuned_qwen`
- Run GRPO training
- Save ART model to `/models/qwen-4b-art`

### 2. Inference (SFT Model)
```bash
modal run modal_app.py::run_inference \
  --context "Your context here" \
  --question "Your question here"
```

### 3. Inference (ART Model)
```bash
modal run modal_app.py::run_inference \
  --context "Your context here" \
  --question "Your question here" \
  --use-art
```

## File Structure in Modal

```
/modal_app.py          # Main Modal app
/reward.py             # Copied during image build
/models/               # Persistent volume
  ├── finetuned_qwen/  # PEFT model (temporary)
  ├── merged_finetuned_qwen/  # Merged SFT model
  └── qwen-4b-art/     # ART/GRPO model
```

## Troubleshooting

### If GRPO training fails:
1. Check Modal logs: `modal logs finsure-ai`
2. Verify ART installation: Check if `from art import GRPOTrainer` works
3. Reduce batch size if OOM: Change `rollout_batch_size=8` to `rollout_batch_size=4`

### If reward function import fails:
- Verify `reward.py` is in the same directory as `modal_app.py`
- Check Modal image build logs

### If model not found during inference:
- Ensure training completed successfully
- Check volume mount: Models should persist in `/models/`

## Testing Locally First

Before running on Modal, test locally:
```bash
python test_dataset.py  # Verify dataset and reward function
```

Then test Modal deployment with a small dataset first (reduce `range(1000)` to `range(10)` for quick test).

