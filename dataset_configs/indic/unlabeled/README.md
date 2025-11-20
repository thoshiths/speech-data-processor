# Indic Unlabeled Audio Processing

This directory contains configurations for processing blind audio files (without ground truth transcriptions) for 12 Indic languages using custom NeMo ASR models.

## Supported Languages

| Language | Code | Script |
|----------|------|--------|
| Hindi | `hi` | Devanagari |
| Telugu | `te` | Telugu |
| Tamil | `ta` | Tamil |
| Bengali | `bn` | Bengali |
| Malayalam | `ml` | Malayalam |
| Kannada | `kn` | Kannada |
| Marathi | `mr` | Devanagari |
| Gujarati | `gu` | Gujarati |
| Punjabi | `pa` | Gurmukhi |
| Odia | `or` | Odia |
| Assamese | `as` | Bengali |
| Urdu | `ur` | Perso-Arabic |

## Configuration Files

### Standard Configurations

### 1. `config.yaml` - Standard Pipeline
Basic pipeline for processing single-language blind audio files.

**Use when:** You have audio files from a single known language.

**Note:** Does NOT use language identification - faster but assumes single language.

---

### Language Identification Configurations

### 2. `config_with_langid.yaml` - NeMo AmberNet LangID
Pipeline using NeMo's `langid_ambernet` model for language detection.

**Use when:** Mixed-language files (different files in different languages)

**Model:** NeMo AmberNet (100+ languages)  
**See:** `LANGID_INFO.md` for details

### 3. `config_whisper_langid.yaml` - Whisper Large V3 LangID ⭐
Pipeline using Faster Whisper Large V3's built-in language detection.

**Use when:** Using Whisper for ASR, or need highest accuracy

**Model:** Whisper Large V3 (99+ languages)  
**Advantages:** State-of-the-art accuracy, built into Whisper

### 4. `config_speechbrain_langid.yaml` - SpeechBrain VoxLingua107
Pipeline using SpeechBrain's specialized language ID model.

**Use when:** Speed is critical, low memory footprint needed

**Model:** VoxLingua107 ECAPA (107 languages)  
**Advantages:** Fastest inference, purpose-built for LangID

### 5. `config_cross_validate_langid.yaml` - Ensemble (Highest Accuracy) 🏆
Pipeline using ALL THREE models for cross-validation.

**Use when:** Maximum accuracy required, short-medium audio

**Models:** NeMo + Whisper + SpeechBrain ensemble  
**Advantages:** 97-99% accuracy, identifies uncertain cases  
**See:** `LANGID_COMPARISON.md` for detailed comparison

### 5b. `config_cross_validate_longaudio.yaml` - Ensemble for Long Audio ⭐ ULTIMATE
Combined VAD segmentation + 3-model cross-validation for long audio.

**Use when:** Long audio + mixed languages + maximum accuracy required

**Process:** VAD → Segment → 3-model LangID per segment → ASR  
**Advantages:** 98-99% accuracy on long mixed-language recordings  
**See:** `LONGAUDIO_CROSSVAL_GUIDE.md` for complete guide

---

### Long Audio / Mixed Language Configurations

### 6. `config_mixed_lang_segments.yaml` - Segment-Level Processing ⭐
Advanced pipeline for LONG audio with multiple languages in the same file.

**Use when:** 
- Long recordings (>1 minute) with code-switching
- Different speakers use different languages
- Conference calls, multilingual conversations

**Process:** VAD → Segment → LangID per segment → Filter → ASR  
**See:** `MIXED_LANGUAGE_GUIDE.md`

### 7. `config_multilingual_asr.yaml` - Multilingual ASR Approach
Alternative using a single multilingual ASR model.

**Use when:** Have multilingual ASR model, want simpler pipeline  
**See:** `MIXED_LANGUAGE_GUIDE.md`

---

### Example Configurations

### 8. `example_hindi.yaml`
Ready-to-use example for Hindi audio processing.

## Prerequisites

1. **NeMo ASR Models**: You need trained NeMo ASR models (`.nemo` files) for your target languages
2. **Audio Files**: Raw audio files in a supported format (WAV, MP3, FLAC, etc.)
3. **GPU**: Recommended for ASR inference (CPU will be much slower)

## Directory Structure Setup

```
workspace_dir/
├── raw_audio/              # Place your audio files here
│   ├── audio_001.wav
│   ├── audio_002.wav
│   └── ...
└── manifests/              # Will be auto-created
    ├── 00_initial_manifest.json
    ├── 01_with_duration.json
    ├── ...
    └── final_manifest.json
```

## Quick Start

### Single Language Processing

```bash
# Step 1: Create workspace and copy audio files
mkdir -p /data/hindi_audio/raw_audio
cp your_audio_files/*.wav /data/hindi_audio/raw_audio/

# Step 2: Run SDP pipeline
python main.py \
  --config-path="dataset_configs/indic/unlabeled" \
  --config-name="config.yaml" \
  processors_to_run="0:" \
  workspace_dir="/data/hindi_audio" \
  language_code="hi" \
  nemo_model_path="/models/hindi_conformer.nemo"

# Step 3: Check output
cat /data/hindi_audio/manifests/hi_final_manifest.json
```

### Mixed Language Processing (with LangID)

```bash
python main.py \
  --config-path="dataset_configs/indic/unlabeled" \
  --config-name="config_with_langid.yaml" \
  processors_to_run="0:" \
  workspace_dir="/data/mixed_audio" \
  language_code="te" \
  nemo_model_path="/models/telugu_conformer.nemo"
```

### Long Audio with Multiple Languages (Code-Switching)

```bash
# For long audio files containing multiple languages
python main.py \
  --config-path="dataset_configs/indic/unlabeled" \
  --config-name="config_mixed_lang_segments.yaml" \
  workspace_dir="/data/long_mixed_audio" \
  language_code="hi" \
  nemo_model_path="/models/hindi_conformer.nemo"

# This will:
# 1. Segment the long audio using VAD
# 2. Detect language per segment
# 3. Transcribe only Hindi segments
```

## Configuration Parameters

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `workspace_dir` | Root directory for audio and outputs | `/data/hindi_audio` |
| `language_code` | Two-letter language code | `hi`, `te`, `ta` |
| `nemo_model_path` | Path to your .nemo model file | `/models/hindi.nemo` |

### Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `audio_extension` | `wav` | Audio file extension to search for |
| `min_duration` | `1.0` | Minimum audio duration (seconds) |
| `max_duration` | `30.0` | Maximum audio duration (seconds) |
| `asr_batch_size` | `32` | Batch size for ASR inference |

## Processing Steps

The pipeline performs the following operations:

1. **Discovery** - Finds all audio files in `raw_audio_dir`
2. **Duration Calculation** - Computes duration for each file
3. **Filtering** - Removes files outside duration range
4. **ASR Inference** - Generates transcriptions using your NeMo model
5. **Text Cleaning** - Normalizes whitespace and formatting
6. **Quality Filtering** - Removes empty/invalid transcriptions
7. **Word Rate Check** - Filters unrealistic word rates
8. **Final Export** - Creates manifest with essential fields

## Output Format

The final manifest file contains one JSON object per line:

```json
{"audio_filepath": "/data/hindi_audio/raw_audio/audio_001.wav", "text": "यह एक उदाहरण है", "duration": 3.5, "pred_text": "यह एक उदाहरण है", "num_words": 4}
{"audio_filepath": "/data/hindi_audio/raw_audio/audio_002.wav", "text": "हिंदी भाषा", "duration": 2.1, "pred_text": "हिंदी भाषा", "num_words": 2}
```

### Fields Description

- `audio_filepath`: Full path to audio file
- `text`: Cleaned transcription text
- `duration`: Audio duration in seconds
- `pred_text`: Raw ASR prediction (before cleaning)
- `num_words`: Word count in transcription

## Batch Processing All Languages

See `scripts/process_all_languages.sh` for an example of processing multiple languages in batch.

## Customization

### Adjust Duration Filters

```bash
python main.py \
  --config-path="dataset_configs/indic/unlabeled" \
  --config-name="config.yaml" \
  ... \
  min_duration=0.5 \
  max_duration=60.0
```

### Change Audio Format

```bash
python main.py \
  --config-path="dataset_configs/indic/unlabeled" \
  --config-name="config.yaml" \
  ... \
  audio_extension=mp3
```

### Adjust Batch Size (for GPU memory)

```bash
python main.py \
  --config-path="dataset_configs/indic/unlabeled" \
  --config-name="config.yaml" \
  ... \
  asr_batch_size=16  # Reduce if out of memory
```

## Troubleshooting

### Out of GPU Memory

Reduce batch size:
```bash
asr_batch_size=8
```

### No Audio Files Found

- Check `raw_audio_dir` path is correct
- Verify `audio_extension` matches your files
- Ensure files have proper extensions

### Empty Transcriptions

- Check model is correct for the language
- Verify audio quality is good
- Check audio format is supported by NeMo

### Slow Processing

- Use GPU instead of CPU
- Increase `asr_batch_size` if you have GPU memory
- Process in parallel for multiple languages

## Advanced Usage

### Run Specific Steps Only

```bash
# Run only steps 0-3 (discovery to filtering)
processors_to_run="0:4"

# Run only ASR inference step
processors_to_run="3"

# Run from ASR onwards
processors_to_run="3:"
```

### Custom Output Location

```bash
final_manifest="/custom/path/output.json"
```

## Quality Checks

After processing, verify your data:

```python
import json

# Load manifest
with open('/data/hindi_audio/manifests/hi_final_manifest.json') as f:
    data = [json.loads(line) for line in f]

# Check statistics
print(f"Total samples: {len(data)}")
print(f"Total duration: {sum(d['duration'] for d in data)/3600:.2f} hours")
print(f"Avg words per sample: {sum(d['num_words'] for d in data)/len(data):.2f}")

# Check for issues
empty = [d for d in data if not d['text'].strip()]
print(f"Empty transcriptions: {len(empty)}")
```

## Next Steps

After generating transcriptions:

1. **Manual Review** - Sample and verify transcription quality
2. **Iterative Training** - Use generated data to train better models
3. **Confidence Filtering** - Add confidence scores to filter high-quality samples
4. **Data Augmentation** - Use for semi-supervised learning

## Support

For issues or questions:
- Check SDP documentation: https://nvidia.github.io/NeMo-speech-data-processor/
- Review NeMo ASR docs: https://docs.nvidia.com/nemo-framework/

