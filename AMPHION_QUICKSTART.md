# Amphion-Style Pipeline - Quick Start Guide

## 🚀 Quick Start (5 Minutes)

### 1. Install Dependencies

```bash
cd /Users/thoshith/Desktop/Gnani/speech-data-processor
pip install -r requirements_amphion_pipeline.txt
```

### 2. Download Source Separation Model

```bash
# Create models directory
mkdir -p /shared/models/source_separation

# Download UVR-MDX-NET model
wget https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_3.onnx \
  -O /shared/models/source_separation/UVR-MDX-NET-Inst_HQ_3.onnx
```

### 3. Get HuggingFace Token

1. Go to: https://huggingface.co/settings/tokens
2. Create a new token (read access)
3. Grant access at: https://huggingface.co/pyannote/speaker-diarization-3.1
4. Copy the token (starts with `hf_`)

### 4. Run the Pipeline

```bash
python main.py \
  --config-path="dataset_configs/indic/unlabeled" \
  --config-name="config_amphion_style_multilang_asr.yaml" \
  processors_to_run="0:" \
  workspace_dir="/shared/workspace" \
  separation_model_path="/shared/models/source_separation/UVR-MDX-NET-Inst_HQ_3.onnx" \
  hf_token="hf_YOUR_TOKEN_HERE" \
  target_languages='["hi","en"]' \
  model_hi="/shared/models/vachana_hybrid_xxl_hi_in.nemo" \
  model_en="/shared/models/vachana_tdt_xl_en_in.nemo"
```

## 📂 New Processor Files Created

```
sdp/processors/
├── inference/
│   ├── audio_processing/
│   │   ├── __init__.py
│   │   └── source_separation.py          # NEW: Source separation
│   ├── vad/
│   │   ├── __init__.py
│   │   └── silero_vad.py                 # NEW: Silero VAD
│   ├── diarization/
│   │   ├── __init__.py
│   │   └── speaker_diarization.py        # NEW: Speaker diarization
│   ├── nlp/
│   │   └── whisper_langid.py             # NEW: Whisper language detection
│   └── asr/
│       └── language_router.py            # NEW: Language-based ASR routing
```

## 📋 Configuration File

```
dataset_configs/indic/unlabeled/
└── config_amphion_style_multilang_asr.yaml    # NEW: Amphion-style config
```

## 🔄 Pipeline Flow

```
Input Audio Files
    ↓
[Stage 1] Source Separation
    ↓ Clean audio (no BGM/noise)
[Stage 2] Speaker Diarization  
    ↓ Speaker-labeled segments
[Stage 3] Silero VAD
    ↓ Fine-grained 3-30s segments
[Stage 4] Whisper Language Detection
    ↓ Per-segment language labels
[Stage 5] Language-Based ASR
    ↓ Batched transcription by language
[Stage 6] Post-processing
    ↓
Final Manifest with Transcriptions
```

## 📊 Expected Output Structure

```
/shared/workspace/
├── manifests/
│   ├── 00_initial_manifest.json
│   ├── 01_with_duration.json
│   ├── 01a_separated.json           # After source separation
│   ├── 02_diarized.json             # After speaker diarization
│   ├── 03_vad_segments.json         # After VAD segmentation
│   ├── 04_lang_detected.json        # After language detection
│   ├── 05_transcribed.json          # After ASR
│   ├── 06_split_segments.json       # Individual segment entries
│   └── final_manifest.json          # Final cleaned dataset
├── separated_audio/                  # Source-separated audio files
└── final_segments/                   # Individual segment audio files
```

## 🎯 Key Features

✅ **Handles Mixed Languages** - Detects language per segment, not per file  
✅ **Clean Audio** - Removes BGM and noise before processing  
✅ **No CUDA Issues** - Whisper runs on CPU, avoids multiprocessing problems  
✅ **Efficient Batching** - Groups segments by language for fast ASR  
✅ **Per-Segment Processing** - Fine-grained language detection and transcription  

## 🔧 Common Customizations

### Process Fewer Languages

```bash
target_languages='["hi","en"]'  # Only Hindi and English
```

### Adjust Segment Length

```yaml
min_segment_duration: 2.0   # Shorter segments
max_segment_duration: 40.0  # Longer segments
```

### Use CPU for Source Separation (Slower but Stable)

```yaml
- _target_: sdp.processors.SourceSeparation
  device: cpu  # Change from cuda to cpu
```

### Skip Source Separation (If Audio is Clean)

```bash
# Start from stage 2 (skip source separation)
processors_to_run="2:"
```

## 📚 Documentation

- **Full Documentation:** `AMPHION_PIPELINE_README.md`
- **Processor Details:** See docstrings in each processor file
- **Original Amphion Code:** `Amphion/preprocessors/Emilia/`

## ⚠️ Troubleshooting

### Issue: `FileNotFoundError: UVR-MDX-NET-Inst_HQ_3.onnx not found`

**Solution:** Download the model (see step 2 above)

### Issue: `ValueError: hf_token must start with 'hf'`

**Solution:** Get HuggingFace token (see step 3 above)

### Issue: CUDA out of memory

**Solution:** Reduce batch size:
```bash
asr_batch_size=16  # Lower from default 32
```

### Issue: Too many small segments

**Solution:** Increase minimum duration:
```bash
min_segment_duration=5.0  # Increase from default 3.0
```

## 🎉 Success Criteria

After running, you should see:

1. ✅ Separated audio files in `separated_audio/`
2. ✅ Manifest with speaker segments in `02_diarized.json`
3. ✅ Fine-grained segments in `03_vad_segments.json`
4. ✅ Language-labeled segments in `04_lang_detected.json`
5. ✅ Transcribed segments in `05_transcribed.json`
6. ✅ Final dataset in `final_manifest.json`

Check logs for:
```
[SDP I] Processed X entries with speaker diarization
[SDP I] VAD segmentation: X → Y merged → Z after filtering
[SDP I] Language detection complete: X total segments, Y valid
[SDP I] ASR complete: X segments transcribed
[SDP I] Language distribution: {'hi': 100, 'en': 50}
```

## 🚀 Next Steps

1. **Review Output:** Check `final_manifest.json` for quality
2. **Adjust Parameters:** Fine-tune based on your data
3. **Add Quality Filters:** Add DNSMOS, SQUIM, etc. (see original config)
4. **Scale Up:** Process your full dataset

---

**Need Help?** See `AMPHION_PIPELINE_README.md` for detailed documentation.

