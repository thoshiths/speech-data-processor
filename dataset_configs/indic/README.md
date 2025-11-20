# Indic Language Dataset Configurations

This directory contains Speech Data Processor (SDP) configurations for processing Indic language audio datasets.

## Overview

The Indic language family includes several major languages spoken primarily in the Indian subcontinent. This repository provides tools and configurations to process speech data for 12 Indic languages using custom NeMo ASR models.

## Supported Languages

| Language | Code | Native Name | Script | Speakers (millions) |
|----------|------|-------------|--------|---------------------|
| Hindi | `hi` | हिन्दी | Devanagari | 600+ |
| Bengali | `bn` | বাংলা | Bengali | 265+ |
| Telugu | `te` | తెలుగు | Telugu | 95+ |
| Marathi | `mr` | मराठी | Devanagari | 83+ |
| Tamil | `ta` | தமிழ் | Tamil | 80+ |
| Gujarati | `gu` | ગુજરાતી | Gujarati | 60+ |
| Kannada | `kn` | ಕನ್ನಡ | Kannada | 50+ |
| Malayalam | `ml` | മലയാളം | Malayalam | 38+ |
| Odia | `or` | ଓଡ଼ିଆ | Odia | 38+ |
| Punjabi | `pa` | ਪੰਜਾਬੀ | Gurmukhi | 33+ |
| Assamese | `as` | অসমীয়া | Bengali | 15+ |
| English | `en` | English | Latin | 1500+ |

## Directory Structure

```
indic/
├── README.md                    # This file
└── unlabeled/                   # Configurations for blind audio processing
    ├── config.yaml              # Standard processing pipeline
    ├── config_with_langid.yaml  # With language detection
    ├── README.md                # Detailed usage instructions
    └── scripts/                 # Helper scripts
        ├── process_all_languages.sh
        ├── setup_workspace.sh
        └── verify_output.py
```

## Use Cases

### 1. Unlabeled Audio Processing (`unlabeled/`)

Process blind audio files without ground truth transcriptions:
- Generate pseudo-labels using ASR
- Prepare data for semi-supervised learning
- Create training data from in-the-wild audio

**Use when:**
- You have audio recordings without transcriptions
- You want to leverage existing ASR models for data creation
- You're doing iterative pseudo-labeling

## Quick Start

### Process Single Language

```bash
# 1. Setup workspace
bash dataset_configs/indic/unlabeled/scripts/setup_workspace.sh /data/hindi_audio hi

# 2. Copy your audio files
cp your_audio/*.wav /data/hindi_audio/raw_audio/

# 3. Run processing
python main.py \
  --config-path="dataset_configs/indic/unlabeled" \
  --config-name="config.yaml" \
  workspace_dir="/data/hindi_audio" \
  language_code="hi" \
  nemo_model_path="/models/hindi_conformer.nemo"
```

### Process All Languages

```bash
# Configure paths in the script, then run:
bash dataset_configs/indic/unlabeled/scripts/process_all_languages.sh
```

## Configuration Options

All configurations support these common parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `workspace_dir` | Root directory for processing | Required |
| `language_code` | Two-letter language code | Required |
| `nemo_model_path` | Path to .nemo ASR model | Required |
| `audio_extension` | Audio file format | `wav` |
| `min_duration` | Min audio length (seconds) | `1.0` |
| `max_duration` | Max audio length (seconds) | `30.0` |
| `asr_batch_size` | ASR inference batch size | `32` |

## Model Requirements

### NeMo ASR Models

You need trained NeMo ASR models (`.nemo` files) for your target languages. Options include:

1. **Custom Trained Models**: Your own models trained on Indic data
2. **AI4Bharat Models**: Pre-trained Conformer models
3. **NVIDIA NGC**: Models from NGC catalog

Example model names:
- `ai4bharat/indicconformer_hi` (Hindi)
- `ai4bharat/indicconformer_te` (Telugu)
- etc.

## Output Format

Processed manifests contain one JSON object per line:

```json
{"audio_filepath": "/path/audio.wav", "text": "transcription", "duration": 3.5, "pred_text": "transcription", "num_words": 4}
```

## Quality Verification

Use the verification script to check output quality:

```bash
python dataset_configs/indic/unlabeled/scripts/verify_output.py \
  --workspace /data/hindi_audio \
  --language hi \
  --verify-script \
  --show-samples 10
```

## Common Issues and Solutions

### Issue: Out of GPU Memory

**Solution:**
```bash
asr_batch_size=8  # Reduce batch size
```

### Issue: No Audio Files Found

**Solution:**
- Check `raw_audio_dir` path
- Verify `audio_extension` matches your files
- Ensure files have proper extensions

### Issue: Poor Transcription Quality

**Solution:**
- Verify model is correct for the language
- Check audio quality and format
- Try different ASR models
- Adjust duration filters

### Issue: Mixed Language Content

**Solution:**
Use `config_with_langid.yaml` to filter by language first

## Advanced Features

### Custom Text Processing

Add language-specific text normalization:

```yaml
- _target_: sdp.processors.SubRegex
  regex_params_list:
    - {"pattern": "specific_pattern", "repl": "replacement"}
```

### Script Validation

Ensure text uses correct Indic script:

```yaml
- _target_: sdp.processors.DropNonAlphabet
  alphabet: "अआइईउऊ..."  # Your language's alphabet
```

### Confidence Filtering

Add confidence-based filtering (requires model support):

```yaml
- _target_: sdp.processors.DropOnAttribute
  attribute_key: "confidence"
  operator: "lt"
  threshold: 0.7
```

## Contributing

To add support for new Indic languages or datasets:

1. Create appropriate config files
2. Add language-specific processors if needed
3. Update documentation
4. Test thoroughly with sample data

## Resources

- **SDP Documentation**: https://nvidia.github.io/NeMo-speech-data-processor/
- **NeMo Framework**: https://docs.nvidia.com/nemo-framework/
- **AI4Bharat**: https://ai4bharat.org/
- **Indic NLP Resources**: https://github.com/AI4Bharat/IndicNLP

## Support

For issues or questions:
1. Check the specific configuration's README
2. Review SDP documentation
3. File an issue on GitHub

## License

These configurations follow the same Apache 2.0 license as the main SDP repository.

