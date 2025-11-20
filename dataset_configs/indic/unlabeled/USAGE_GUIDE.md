# Usage Guide: Multi-Language ASR Pipeline

## Quick Start

### Process All 12 Languages

```bash
python /path/to/speech-data-processor/main.py \
  --config-path=/path/to/dataset_configs/indic/unlabeled \
  --config-name=config_cross_validate_multilang_asr \
  workspace_dir=/shared/workspace \
  raw_audio_dir=/path/to/audio \
  model_hi=/shared/models/vachana_hybrid_xxl_hi_in.nemo \
  model_te=/shared/models/te_asr.nemo \
  model_ta=/shared/models/ta_asr.nemo \
  model_bn=/shared/models/bn_asr.nemo \
  model_ml=/shared/models/ml_asr.nemo \
  model_kn=/shared/models/kn_asr.nemo \
  model_mr=/shared/models/mr_asr.nemo \
  model_gu=/shared/models/gu_asr.nemo \
  model_pa=/shared/models/pa_asr.nemo \
  model_or=/shared/models/or_asr.nemo \
  model_as=/shared/models/as_asr.nemo \
  model_en=/shared/models/vachana_tdt_xl_en_in.nemo
```

### Process Specific Languages (e.g., Hindi and English only)

```bash
python /path/to/speech-data-processor/main.py \
  --config-path=/path/to/dataset_configs/indic/unlabeled \
  --config-name=config_cross_validate_multilang_asr \
  workspace_dir=/shared/workspace \
  raw_audio_dir=/path/to/audio \
  target_languages='["hi","en"]' \
  model_hi=/shared/models/vachana_hybrid_xxl_hi_in.nemo \
  model_te=/shared/models/te_asr.nemo \
  model_ta=/shared/models/ta_asr.nemo \
  model_bn=/shared/models/bn_asr.nemo \
  model_ml=/shared/models/ml_asr.nemo \
  model_kn=/shared/models/kn_asr.nemo \
  model_mr=/shared/models/mr_asr.nemo \
  model_gu=/shared/models/gu_asr.nemo \
  model_pa=/shared/models/pa_asr.nemo \
  model_or=/shared/models/or_asr.nemo \
  model_as=/shared/models/as_asr.nemo \
  model_en=/shared/models/vachana_tdt_xl_en_in.nemo
```

## ⚠️ Important: Model Paths Required for All Languages

**Even if you only want to process 2 languages, you must provide paths for ALL 12 model parameters.**

### Why?
- Hydra (the configuration framework) resolves all variables at startup
- It will fail with `InterpolationKeyError` if any model path is missing
- **Don't worry:** Models for unused languages are NEVER actually loaded!

### What Actually Happens
1. You set `target_languages='["hi","en"]'`
2. LangID runs on all audio segments
3. **Filter step**: Only Hindi and English segments are kept
4. ASR runs ONLY on Hindi and English segments
5. Other language ASR models are never instantiated or loaded

### Solution: Point All Models to the Same File

If you don't have all 12 models, just reuse existing ones:

```bash
python /path/to/speech-data-processor/main.py \
  --config-path=/path/to/dataset_configs/indic/unlabeled \
  --config-name=config_cross_validate_multilang_asr \
  workspace_dir=/shared/workspace \
  raw_audio_dir=/path/to/audio \
  target_languages='["hi","en"]' \
  model_hi=/shared/models/vachana_hybrid_xxl_hi_in.nemo \
  model_en=/shared/models/vachana_tdt_xl_en_in.nemo \
  model_te=/shared/models/vachana_hybrid_xxl_hi_in.nemo \
  model_ta=/shared/models/vachana_hybrid_xxl_hi_in.nemo \
  model_bn=/shared/models/vachana_hybrid_xxl_hi_in.nemo \
  model_ml=/shared/models/vachana_hybrid_xxl_hi_in.nemo \
  model_kn=/shared/models/vachana_hybrid_xxl_hi_in.nemo \
  model_mr=/shared/models/vachana_hybrid_xxl_hi_in.nemo \
  model_gu=/shared/models/vachana_hybrid_xxl_hi_in.nemo \
  model_pa=/shared/models/vachana_hybrid_xxl_hi_in.nemo \
  model_or=/shared/models/vachana_hybrid_xxl_hi_in.nemo \
  model_as=/shared/models/vachana_hybrid_xxl_hi_in.nemo
```

Since only Hindi and English will be processed, the Telugu/Tamil/etc models will never be loaded.

## Custom Quality Thresholds

Adjust audio quality filtering for TTS:

```bash
python /path/to/speech-data-processor/main.py \
  --config-path=/path/to/dataset_configs/indic/unlabeled \
  --config-name=config_cross_validate_multilang_asr \
  workspace_dir=/shared/workspace \
  raw_audio_dir=/path/to/audio \
  target_languages='["hi"]' \
  min_pesq=2.5 \
  min_stoi=0.85 \
  min_sisdr=10.0 \
  min_snr=15.0 \
  min_bandwidth=6000 \
  model_hi=/shared/models/hi.nemo \
  model_en=/shared/models/en.nemo \
  model_te=/shared/models/hi.nemo \
  model_ta=/shared/models/hi.nemo \
  model_bn=/shared/models/hi.nemo \
  model_ml=/shared/models/hi.nemo \
  model_kn=/shared/models/hi.nemo \
  model_mr=/shared/models/hi.nemo \
  model_gu=/shared/models/hi.nemo \
  model_pa=/shared/models/hi.nemo \
  model_or=/shared/models/hi.nemo \
  model_as=/shared/models/hi.nemo
```

## Enable Punctuation Restoration

### Using LLM (Gemma-2-27B)

```bash
python /path/to/speech-data-processor/main.py \
  --config-path=/path/to/dataset_configs/indic/unlabeled \
  --config-name=config_cross_validate_multilang_asr \
  workspace_dir=/shared/workspace \
  raw_audio_dir=/path/to/audio \
  enable_punctuation=true \
  punctuation_method=llm \
  punctuation_model=google/gemma-2-27b-it \
  tensor_parallel_size=2 \
  target_languages='["hi","en"]' \
  model_hi=/shared/models/hi.nemo \
  [... all other models ...]
```

## Language Codes Reference

| Code | Language   |
|------|------------|
| hi   | Hindi      |
| te   | Telugu     |
| ta   | Tamil      |
| bn   | Bengali    |
| ml   | Malayalam  |
| kn   | Kannada    |
| mr   | Marathi    |
| gu   | Gujarati   |
| pa   | Punjabi    |
| or   | Odia       |
| as   | Assamese   |
| en   | English    |

## 8-GPU Configuration

For faster processing on 8 GPUs, use `config_8gpu.yaml` instead. Same parameter requirements apply.

```bash
python /path/to/speech-data-processor/main.py \
  --config-path=/path/to/dataset_configs/indic/unlabeled \
  --config-name=config_8gpu \
  [... same parameters as above ...]
```

## Common Errors

### Error: `AttributeError: module 'torchaudio' has no attribute 'list_audio_backends'`

**Cause:** SpeechBrain incompatibility with torchaudio 2.1.0+

**Solution:** ✅ **Already Fixed!** The code includes an automatic compatibility patch. Just restart the pipeline.

### Error: `InterpolationKeyError: 'model_te' not found`

**Cause:** You didn't provide all 12 model paths.

**Solution:** Provide all model parameters, even if using `target_languages`. See "Solution: Point All Models to the Same File" above.

### Error: `DropOnAttribute.__init__() missing 1 required positional argument: 'key'`

**Cause:** Using old configuration with incorrect quality filtering.

**Solution:** ✅ **Already Fixed!** Make sure you're using the updated `config_cross_validate_multilang_asr.yaml`.

---

**For more troubleshooting, see:** `TROUBLESHOOTING.md`

