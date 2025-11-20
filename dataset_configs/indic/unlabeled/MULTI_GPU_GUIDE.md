# Multi-GPU Processing Guide for 8 GPUs

This guide explains how to efficiently utilize 8 GPUs for maximum-speed processing of mixed-language audio.

## 🚀 Quick Start (8 GPUs)

```bash
# Ensure all 8 GPUs are visible
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Run the optimized configuration
python main.py \
  --config-path="dataset_configs/indic/unlabeled" \
  --config-name="config_8gpu.yaml" \
  workspace_dir="/data/audio" \
  model_hi="/models/hindi.nemo" \
  model_te="/models/telugu.nemo" \
  model_ta="/models/tamil.nemo" \
  model_bn="/models/bengali.nemo" \
  model_ml="/models/malayalam.nemo" \
  model_kn="/models/kannada.nemo" \
  model_mr="/models/marathi.nemo" \
  model_gu="/models/gujarati.nemo" \
  model_pa="/models/punjabi.nemo" \
  model_or="/models/odia.nemo" \
  model_as="/models/assamese.nemo" \
  model_en="/models/english.nemo"
```

## 📊 GPU Distribution Strategy

### Optimal 8-GPU Layout

```
┌─────────┬───────────────────────────────┬──────────────────┐
│ GPU ID  │ Task                          │ Workload         │
├─────────┼───────────────────────────────┼──────────────────┤
│ GPU 0   │ NeMo AmberNet LangID          │ LangID (Model 1) │
│ GPU 1   │ SpeechBrain VoxLingua107      │ LangID (Model 2) │
│ GPU 2   │ Faster Whisper Large-V3       │ LangID (Model 3) │
├─────────┼───────────────────────────────┼──────────────────┤
│ GPU 3   │ Hindi + Telugu + Tamil ASR    │ ASR Batch 1      │
│ GPU 4   │ Bengali + Malayalam + Kannada │ ASR Batch 2      │
│ GPU 5   │ Marathi + Gujarati + Punjabi  │ ASR Batch 3      │
│ GPU 6   │ Odia + Assamese + English     │ ASR Batch 4      │
├─────────┼───────────────────────────────┼──────────────────┤
│ GPU 7   │ Spare / Overflow              │ Available        │
└─────────┴───────────────────────────────┴──────────────────┘
```

### Why This Distribution?

1. **Parallel Language ID (GPUs 0-2)**: All 3 LangID models run simultaneously
2. **Parallel ASR (GPUs 3-6)**: Languages processed sequentially within each GPU, but 4 GPUs work in parallel
3. **Load Balancing**: Each ASR GPU handles 3 languages, distributing workload evenly
4. **Spare GPU**: GPU 7 available for overflow or other tasks

## ⚡ Performance Comparison

### Processing Time for 100 Hours of Audio

| Setup | Time | Speed |
|-------|------|-------|
| 1 GPU (Sequential) | ~10-12 hours | 1x |
| 4 GPUs | ~3-4 hours | 3x faster |
| **8 GPUs** | **~1.5-2 hours** | **6-8x faster** ⚡ |

### Bottlenecks Addressed

- ✅ **LangID Parallelization**: 3 models run simultaneously instead of sequentially
- ✅ **ASR Parallelization**: 4 language groups processed in parallel
- ✅ **Increased Batch Sizes**: 64 for ASR, 128 for LangID (vs 32 single-GPU)
- ✅ **GPU Memory**: Each model gets dedicated GPU memory

## 🔧 Configuration Options

### Adjust Batch Sizes Based on GPU Memory

```yaml
# For 16GB GPUs (e.g., Tesla T4)
asr_batch_size: 32
langid_batch_size: 64

# For 24GB GPUs (e.g., RTX 3090, A5000)
asr_batch_size: 64
langid_batch_size: 128

# For 40GB+ GPUs (e.g., A100)
asr_batch_size: 128
langid_batch_size: 256
```

### Custom GPU Assignment

Override default GPU assignments:

```bash
python main.py \
  --config-name="config_8gpu.yaml" \
  gpu_langid_nemo=0 \
  gpu_langid_speechbrain=1 \
  gpu_langid_whisper=2 \
  gpu_asr_batch1=3 \
  gpu_asr_batch2=4 \
  gpu_asr_batch3=5 \
  gpu_asr_batch4=6 \
  ...
```

### Use Specific GPUs Only

If you only want to use specific GPUs:

```bash
# Use only GPUs 2, 3, 4, 5, 6, 7 (skip 0, 1)
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,0,1

# Or use a subset of 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Then adjust config for 4 GPUs
```

## 📈 Monitoring GPU Usage

### Real-Time Monitoring

Run this in a separate terminal while processing:

```bash
cd dataset_configs/indic/unlabeled/scripts
./monitor_8gpu.sh
```

This displays:
- GPU utilization percentage
- Memory usage
- Temperature
- Power consumption
- Active processes

### Manual Monitoring

```bash
# Watch GPU stats (updates every 1 second)
watch -n 1 nvidia-smi

# Log GPU usage to file
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
  --format=csv -l 1 > gpu_usage.log
```

## 🎯 Optimization Tips

### 1. **Pre-load Models**

All models are loaded once at the start. No need to reload between batches.

### 2. **Balanced Language Distribution**

The pipeline automatically distributes languages based on detection. If you know some languages are more common, you can adjust GPU assignments.

### 3. **Pipeline Stages**

The pipeline runs in stages:
1. **Stage 1 (Segmentation)**: CPU-bound, no GPU needed
2. **Stage 2 (LangID)**: Uses GPUs 0-2 in parallel
3. **Stage 3 (ASR)**: Uses GPUs 3-6 in parallel
4. **Stage 4-5 (Post-processing)**: CPU-bound

### 4. **Disk I/O**

Ensure fast storage (NVMe SSD) for reading audio files. Disk I/O can become a bottleneck with 8 GPUs processing simultaneously.

## 🔥 Advanced: Different GPU Counts

### 4 GPUs Configuration

```yaml
# Simplified assignment for 4 GPUs
gpu_langid_nemo: 0
gpu_langid_speechbrain: 0  # Share GPU 0
gpu_langid_whisper: 1
gpu_asr_batch1: 2          # Handle 6 languages
gpu_asr_batch2: 3          # Handle 6 languages
```

### 2 GPUs Configuration

```yaml
# All LangID on GPU 0
gpu_langid_nemo: 0
gpu_langid_speechbrain: 0
gpu_langid_whisper: 0
# All ASR on GPU 1
gpu_asr_batch1: 1
gpu_asr_batch2: 1
gpu_asr_batch3: 1
gpu_asr_batch4: 1
```

## 🐛 Troubleshooting

### Out of Memory Errors

```bash
# Reduce batch sizes
asr_batch_size=16
langid_batch_size=32
```

### GPU Not Detected

```bash
# Check GPU availability
nvidia-smi

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Uneven GPU Utilization

This is normal! Different stages use different GPUs:
- LangID stage: GPUs 0-2 busy, GPUs 3-7 idle
- ASR stage: GPUs 3-6 busy, GPUs 0-2 idle

### One GPU Overloaded

If one language dominates, redistribute:
```yaml
# Move high-frequency language to dedicated GPU
gpu_asr_batch1: 3  # Hindi only
gpu_asr_batch2: 4  # Telugu, Tamil
```

## 📝 Best Practices

1. ✅ **Monitor First Run**: Use `monitor_8gpu.sh` to verify all GPUs are utilized
2. ✅ **Adjust Batch Sizes**: Start conservative, increase if GPU utilization < 80%
3. ✅ **Balance Workload**: If one language dominates, give it a dedicated GPU
4. ✅ **Check Disk Speed**: Ensure storage keeps up with GPU processing
5. ✅ **Temperature Check**: Ensure adequate cooling for sustained 8-GPU operation

## 🎓 Example Workflow

```bash
# Terminal 1: Start monitoring
cd dataset_configs/indic/unlabeled/scripts
./monitor_8gpu.sh

# Terminal 2: Run processing
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python main.py \
  --config-name="config_8gpu.yaml" \
  workspace_dir="/data/audio" \
  asr_batch_size=64 \
  langid_batch_size=128 \
  model_hi="/models/hindi.nemo" \
  ... [other models]

# Terminal 3: Monitor progress
watch -n 5 "ls -lh /data/audio/manifests/*.json | tail -5"
```

## 📚 Additional Resources

- **config_8gpu.yaml**: The optimized 8-GPU configuration
- **monitor_8gpu.sh**: Real-time GPU monitoring script
- **README.md**: General documentation for Indic processing
- **LANGID_COMPARISON.md**: Details on the 3 LangID models

---

**Pro Tip**: With 8 GPUs, you can process 100 hours of mixed-language audio in under 2 hours! 🚀

