# Troubleshooting Guide

## Common Errors and Solutions

### 1. `AttributeError: module 'torchaudio' has no attribute 'list_audio_backends'`

**Error:**
```
AttributeError: module 'torchaudio' has no attribute 'list_audio_backends'
```

**Cause:** 
- SpeechBrain tries to call `torchaudio.list_audio_backends()` which was removed in torchaudio 2.1.0+
- You have a newer version of torchaudio

**Solution:**
✅ **Already Fixed!** The code now includes an automatic compatibility patch that:
1. Detects if `list_audio_backends` is missing
2. Monkey-patches it with a dummy function
3. Allows SpeechBrain to work with newer torchaudio

If you still see this error, update your code to the latest version.

**Alternative Solution (if patch fails):**
```bash
# Downgrade torchaudio to older version
pip install torchaudio==2.0.2
```

---

### 1a. Warning: "SpeechBrain could not find any working torchaudio backend"

**Warning:**
```
SpeechBrain could not find any working torchaudio backend. Audio files may fail to load.
```

**Cause:** 
- SpeechBrain can't find audio I/O libraries
- Missing `soundfile` or `sox` backend

**Solution:**
```bash
# Install soundfile (recommended)
pip install soundfile

# Or install sox
# On Ubuntu/Debian:
sudo apt-get install libsox-dev
pip install sox

# Or use our llm requirements (already includes soundfile)
pip install -r requirements/llm.txt
```

✅ **Note:** The warning is harmless if the model loads successfully. The code now sets the backend automatically.

---

### 2. Warning: "SpeechBrain could not find any working torchaudio backend"

**Warning:**
```
SpeechBrain could not find any working torchaudio backend. Audio files may fail to load.
```

**Cause:** 
- SpeechBrain can't find audio I/O libraries
- Missing `soundfile` or `sox` backend

**Solution:**
```bash
# Install soundfile (recommended)
pip install soundfile

# Or install sox
# On Ubuntu/Debian:
sudo apt-get install libsox-dev
pip install sox

# Or use our llm requirements (already includes soundfile)
pip install -r requirements/llm.txt
```

✅ **Note:** The warning is harmless if the model loads successfully. The code now sets the backend automatically.

---

### 3. `InterpolationKeyError: 'model_te' not found`

**Error:**
```
omegaconf.errors.InterpolationKeyError: Interpolation key 'model_te' not found
```

**Cause:** 
You didn't provide all 12 model paths when running the pipeline.

**Solution:**
Provide ALL model paths (even for languages not in `target_languages`):

```bash
python main.py \
  target_languages='["hi","en"]' \
  model_hi=/path/to/hi.nemo \
  model_en=/path/to/en.nemo \
  model_te=/path/to/hi.nemo \
  model_ta=/path/to/hi.nemo \
  model_bn=/path/to/hi.nemo \
  model_ml=/path/to/hi.nemo \
  model_kn=/path/to/hi.nemo \
  model_mr=/path/to/hi.nemo \
  model_gu=/path/to/hi.nemo \
  model_pa=/path/to/hi.nemo \
  model_or=/path/to/hi.nemo \
  model_as=/path/to/hi.nemo
```

**Note:** Unused models are never loaded! Just point them to any existing .nemo file.

---

### 4. `RuntimeError: Error(s) in loading state_dict for EncDecFrameClassificationModel: Unexpected key(s) in state_dict: "loss.weight"`

**Error:**
```
RuntimeError: Error(s) in loading state_dict for EncDecFrameClassificationModel:
        Unexpected key(s) in state_dict: "loss.weight".
```

**Cause:** 
- NeMo VAD model checkpoint contains extra keys not expected by the current model definition
- Checkpoint format mismatch between NeMo versions

**Solution:**
✅ **Already Fixed!** The code now includes an automatic patch that:
1. Detects VAD/classification models
2. Loads them with `strict=False` to ignore extra keys
3. Allows the model to load successfully

The patch is automatically applied when you run the pipeline.

**Alternative Solution (if patch fails):**
Try a different VAD model:
```bash
python main.py \
  vad_model="vad_multilingual_marblenet" \  # Try marblenet instead
  ...
```

---

### 5. `DropOnAttribute.__init__() missing 1 required positional argument: 'key'`

**Error:**
```
TypeError("DropOnAttribute.__init__() missing 1 required positional argument: 'key'")
```

**Cause:** 
Using old configuration with incorrect quality filtering processor.

**Solution:**
✅ **Already Fixed!** The configuration now uses `PreserveByValue` instead of `DropOnAttribute` for numeric quality thresholds.

Update to the latest `config_cross_validate_multilang_asr.yaml`.

---

### 4. CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

**Option 1: Reduce Batch Sizes**
```bash
python main.py \
  asr_batch_size=16 \
  langid_batch_size=32 \
  ...
```

**Option 2: Use CPU for Some Steps**
```bash
python main.py \
  vad_device=cpu \
  langid_device=cpu \
  ...
```

**Option 3: Process Fewer Languages**
```bash
python main.py \
  target_languages='["hi"]' \
  ...
```

**Option 4: Use 8-GPU Configuration**
```bash
python main.py \
  --config-name=config_8gpu \
  ...
```

---

### 5. VAD Segmentation Creates Too Many/Few Segments

**Issue:** Audio is split into too many tiny segments or not split enough.

**Solutions:**

**Adjust VAD Parameters:**
```bash
python main.py \
  vad_onset=0.8 \          # Higher = more strict (fewer segments)
  vad_offset=0.6 \         # Lower = more permissive
  min_segment_duration=2.0 \   # Minimum segment length
  max_segment_duration=20.0 \  # Maximum segment length
  ...
```

**Parameter Guide:**
- `vad_onset`: 0.5-0.9 (higher = detect speech more conservatively)
- `vad_offset`: 0.3-0.7 (lower = end speech segments later)
- `min_segment_duration`: 1.0-5.0 seconds
- `max_segment_duration`: 15.0-30.0 seconds

---

### 6. Language Detection Accuracy Issues

**Issue:** LangID is misclassifying languages.

**Solutions:**

**Option 1: Adjust Confidence Threshold**
```bash
python main.py \
  nemo_langid_min_confidence=0.8 \      # Higher = more conservative
  whisper_langid_min_confidence=0.7 \
  speechbrain_langid_min_confidence=0.6 \
  crossval_agreement_threshold=0.6 \    # Require 2/3 models to agree
  ...
```

**Option 2: Use Single Model (Faster)**
If cross-validation is too slow, use just NeMo LangID:
```bash
# Edit config to skip steps 10, 11, 12 (WhisperLangId, SpeechBrainLangId, CrossValidateLangId)
# Use only step 9 (AudioLid/NeMo)
```

**Option 3: Review Disagreements**
```bash
# Check where models disagreed
jq 'select(.nemo_langid != .whisper_langid)' manifests/12_whisper_langid.json | less
```

---

### 7. Quality Filtering Too Aggressive

**Issue:** Too many samples filtered out by quality metrics.

**Solutions:**

**Lower Quality Thresholds:**
```bash
python main.py \
  min_pesq=1.5 \      # Default: 2.0 (range 1.0-4.5)
  min_stoi=0.7 \      # Default: 0.8 (range 0-1)
  min_sisdr=5.0 \     # Default: 10.0 (dB)
  min_snr=10.0 \      # Default: 15.0 (dB)
  min_bandwidth=4000 \ # Default: 6000 (Hz)
  ...
```

**Check Distributions Before Filtering:**
```bash
# View metric distributions
jq -r '.pesq' manifests/18_with_squim_metrics.json | sort -n | uniq -c
jq -r '.stoi' manifests/18_with_squim_metrics.json | sort -n | uniq -c
```

---

### 8. Punctuation Restoration Failing

**Issue:** LLM punctuation is slow or fails.

**Solutions:**

**Option 1: Use NeMo Model Instead**
```bash
python main.py \
  enable_punctuation=true \
  punctuation_method=nemo \
  punctuation_model=punctuation_en_distilbert \
  ...
```

**Option 2: Increase GPU Resources for LLM**
```bash
python main.py \
  punctuation_method=llm \
  tensor_parallel_size=4 \    # Use 4 GPUs for LLM
  max_tokens=100 \            # Limit output length
  ...
```

**Option 3: Disable Punctuation**
```bash
python main.py \
  enable_punctuation=false \
  ...
```

---

### 9. Pipeline Stops at Specific Step

**Issue:** Pipeline crashes or hangs at a particular processor.

**Debug Steps:**

1. **Check Logs:**
```bash
tail -100 /shared/workspace/logs/*.log
```

2. **Inspect Intermediate Manifest:**
```bash
# Check the last successfully created manifest
ls -lth /shared/workspace/manifests/ | head
jq . /shared/workspace/manifests/XX_*.json | less
```

3. **Run Specific Processor Only:**
```bash
python main.py \
  processors_to_run="10:11" \  # Run only step 10
  ...
```

4. **Enable Debug Mode:**
```bash
HYDRA_FULL_ERROR=1 python main.py ...
```

---

### 10. Import Errors (Missing Dependencies)

**Error:**
```
ModuleNotFoundError: No module named 'vllm'
```

**Solutions:**

Install the appropriate requirements:

```bash
# For LLM punctuation
pip install -r requirements/llm.txt

# For quality metrics
pip install -r requirements/quality.txt

# For all features
pip install -r requirements/main.txt
pip install -r requirements/llm.txt
pip install -r requirements/quality.txt
```

---

## Getting Help

If you encounter an error not listed here:

1. **Check Dask Dashboard**: http://127.0.0.1:8787/status
2. **Read full logs**: `/shared/workspace/logs/*.log`
3. **Check intermediate manifests**: `/shared/workspace/manifests/`
4. **Enable verbose logging**: `HYDRA_FULL_ERROR=1`
5. **Share error traceback and config parameters**

## Performance Optimization Tips

### Single GPU System
```bash
python main.py \
  --config-name=config_cross_validate_multilang_asr \
  asr_batch_size=16 \
  langid_batch_size=32 \
  num_workers=4
```

### 8 GPU System
```bash
python main.py \
  --config-name=config_8gpu \
  asr_batch_size=64 \
  langid_batch_size=128 \
  num_workers=8
```

### CPU Only (Slow but Possible)
```bash
python main.py \
  vad_device=cpu \
  langid_device=cpu \
  asr_device=cpu \
  asr_batch_size=1 \
  num_workers=1
```

### Faster Processing (Skip Quality/Punctuation)
```bash
python main.py \
  enable_quality_filtering=false \
  enable_punctuation=false \
  target_languages='["hi"]'  # Process only 1 language
```

