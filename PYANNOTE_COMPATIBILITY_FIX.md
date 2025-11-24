# PyAnnote.audio Compatibility Fix

## Problem

If you see this error:
```
AttributeError: module 'torchaudio' has no attribute 'AudioMetaData'
```

This is a version incompatibility between `pyannote.audio` and `torchaudio`.

## Your Current Setup

Based on the error:
- **torchaudio:** 2.9.0+cu128 (very new)
- **pyannote.audio:** Likely 3.1.x (older)

## Solution

### Option 1: Upgrade PyAnnote (Recommended)

```bash
pip install --upgrade "pyannote.audio>=3.3.0"
```

This upgrades pyannote.audio to a version compatible with torchaudio 2.9.0.

### Option 2: Use Compatible Versions

If upgrading doesn't work, install specific compatible versions:

```bash
# Uninstall current versions
pip uninstall -y pyannote.audio torchaudio torch

# Install compatible versions
pip install torch==2.2.0 torchaudio==2.2.0
pip install "pyannote.audio>=3.1.0,<3.3.0"
```

## What I Fixed

I updated the speaker diarization processor to use **lazy imports**:
- Imports are now done inside the `prepare()` method
- Won't fail at module load time
- Only imports when you actually use speaker diarization

This means the error will only occur when you run the SpeakerDiarization processor, not at startup.

## Test the Fix

Try running again:

```bash
python3 main.py \
  --config-path="dataset_configs/indic/unlabeled" \
  --config-name="config_amphion_style_multilang_asr.yaml" \
  processors_to_run="0:" \
  hf_token="hf_YOUR_TOKEN_HERE"
```

If you still get the error when it reaches the SpeakerDiarization step, run:

```bash
pip install --upgrade pyannote.audio
```

## Alternative: Skip Speaker Diarization

If you don't need speaker diarization, you can skip that stage:

```yaml
# In your config, comment out or skip the diarization processor
processors_to_run: "0:1,3:"  # Skip processor at index 2 (diarization)
```

Or run without speaker diarization by using a simpler pipeline:

```bash
# Use original pipeline without Amphion-style diarization
python3 main.py \
  --config-path="dataset_configs/indic/unlabeled" \
  --config-name="config_cross_validate_multilang_asr.yaml" \
  processors_to_run="0:"
```

## Verification

After fixing, verify the installation:

```bash
python3 -c "
from pyannote.audio import Pipeline
import torchaudio
print(f'✅ pyannote.audio: Successfully imported')
print(f'✅ torchaudio: {torchaudio.__version__}')
"
```

If this runs without errors, you're all set!

