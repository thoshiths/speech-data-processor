# Indic Blind Audio Processing - Quick Start Guide

This guide will help you process blind Indic audio files (without transcriptions) in 5 minutes.

## Prerequisites

✅ NeMo ASR model for your target language (`.nemo` file)  
✅ Audio files (WAV, MP3, or FLAC format)  
✅ GPU with CUDA (recommended) or CPU  
✅ SDP installed with requirements

## Step 1: Setup Workspace (1 minute)

```bash
# Choose your language (hi=Hindi, te=Telugu, ta=Tamil, etc.)
LANG_CODE="hi"
WORKSPACE="/data/${LANG_CODE}_audio"

# Create directory structure
bash dataset_configs/indic/unlabeled/scripts/setup_workspace.sh ${WORKSPACE} ${LANG_CODE}
```

## Step 2: Add Your Audio Files (1 minute)

```bash
# Copy your audio files to the raw_audio directory
cp /path/to/your/audio/*.wav ${WORKSPACE}/raw_audio/

# Verify files are copied
ls ${WORKSPACE}/raw_audio/ | head -5
```

## Step 3: Run Processing (2-10 minutes depending on data size)

```bash
# Update these paths
MODEL_PATH="/path/to/your/${LANG_CODE}_model.nemo"
SDP_ROOT="/path/to/speech-data-processor"

# Run SDP pipeline
cd ${SDP_ROOT}

python main.py \
  --config-path="dataset_configs/indic/unlabeled" \
  --config-name="config.yaml" \
  workspace_dir="${WORKSPACE}" \
  language_code="${LANG_CODE}" \
  nemo_model_path="${MODEL_PATH}"
```

## Step 4: Check Output (1 minute)

```bash
# View final manifest
head -n 3 ${WORKSPACE}/manifests/${LANG_CODE}_final_manifest.json

# Run verification
python dataset_configs/indic/unlabeled/scripts/verify_output.py \
  --workspace ${WORKSPACE} \
  --language ${LANG_CODE} \
  --show-samples 5
```

## Expected Output

You should see a manifest file like:

```json
{"audio_filepath": "/data/hi_audio/raw_audio/audio_001.wav", "text": "यह एक परीक्षण है", "duration": 3.5, "pred_text": "यह एक परीक्षण है", "num_words": 4}
{"audio_filepath": "/data/hi_audio/raw_audio/audio_002.wav", "text": "हिंदी भाषा", "duration": 2.1, "pred_text": "हिंदी भाषा", "num_words": 2}
```

## Troubleshooting

### Problem: Out of GPU Memory

**Solution:** Reduce batch size:
```bash
python main.py ... asr_batch_size=8
```

### Problem: No audio files found

**Solution:** Check paths and extensions:
```bash
ls ${WORKSPACE}/raw_audio/*.wav
# If files have different extension:
python main.py ... audio_extension=mp3
```

### Problem: Empty transcriptions

**Solution:** 
1. Verify model path is correct
2. Check audio quality
3. Try with a few test files first

## Next Steps

✅ **Verify Quality**: Review sample transcriptions  
✅ **Process More Languages**: Repeat for other Indic languages  
✅ **Train Models**: Use generated data for model training  
✅ **Iterate**: Improve models and regenerate transcriptions  

## Full Example Commands

### Hindi

```bash
python main.py \
  --config-path="dataset_configs/indic/unlabeled" \
  --config-name="config.yaml" \
  workspace_dir="/data/hindi_audio" \
  language_code="hi" \
  nemo_model_path="/models/hindi_conformer.nemo"
```

### Telugu

```bash
python main.py \
  --config-path="dataset_configs/indic/unlabeled" \
  --config-name="config.yaml" \
  workspace_dir="/data/telugu_audio" \
  language_code="te" \
  nemo_model_path="/models/telugu_conformer.nemo"
```

### Tamil

```bash
python main.py \
  --config-path="dataset_configs/indic/unlabeled" \
  --config-name="config.yaml" \
  workspace_dir="/data/tamil_audio" \
  language_code="ta" \
  nemo_model_path="/models/tamil_conformer.nemo"
```

## Need Help?

- 📖 Read: `dataset_configs/indic/unlabeled/README.md`
- 📖 Read: `dataset_configs/indic/README.md`
- 🔧 Check: Configuration parameters in `config.yaml`
- 🐛 Debug: Check logs in `${WORKSPACE}/manifests/`

## Success Checklist

- [ ] Workspace created with correct structure
- [ ] Audio files copied to `raw_audio/` directory
- [ ] NeMo model path is correct and file exists
- [ ] SDP pipeline runs without errors
- [ ] Final manifest contains expected transcriptions
- [ ] Verification script shows reasonable statistics

**Congratulations!** You've successfully processed Indic blind audio files! 🎉

