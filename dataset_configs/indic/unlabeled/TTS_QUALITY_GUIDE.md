# TTS Audio Quality Filtering Guide

This guide explains the audio quality metrics used to filter data for high-quality TTS (Text-to-Speech) model training.

## 📊 Quality Metrics Explained

### 1. **PESQ** (Perceptual Evaluation of Speech Quality)
- **Scale**: 1.0 to 5.0 (higher is better)
- **What it measures**: Overall perceptual quality of speech
- **Originally designed for**: Detecting codec distortions
- **For TTS**:
  - **Excellent**: PESQ > 4.0
  - **Good**: PESQ > 3.5
  - **Acceptable**: PESQ > 3.0 ✅ (default threshold)
  - **Poor**: PESQ < 3.0

### 2. **STOI** (Short-Time Objective Intelligibility)
- **Scale**: 0.0 to 1.0 (higher is better)
- **What it measures**: Speech intelligibility (how understandable the speech is)
- **Evaluates**: Speech envelope integrity across time
- **For TTS**:
  - **Excellent**: STOI > 0.9
  - **Good**: STOI > 0.85
  - **Acceptable**: STOI > 0.75 ✅ (default threshold)
  - **Poor**: STOI < 0.75

### 3. **SI-SDR** (Scale-Invariant Signal-to-Distortion Ratio)
- **Scale**: Decibels (dB, higher is better)
- **What it measures**: Speech signal strength vs. distortion
- **Interpretation**:
  - **0 dB**: Signal and distortion have equal energy
  - **15-20 dB**: Clean speech (standard for TTS)
- **For TTS**:
  - **Excellent**: SI-SDR > 20 dB
  - **Good**: SI-SDR > 15 dB
  - **Acceptable**: SI-SDR > 12 dB ✅ (default threshold)
  - **Poor**: SI-SDR < 12 dB

### 4. **SNR** (Signal-to-Noise Ratio)
- **Scale**: Decibels (dB, higher is better)
- **What it measures**: Speech loudness compared to background noise
- **For TTS**:
  - **Excellent**: SNR > 25 dB
  - **Good**: SNR > 20 dB
  - **Acceptable**: SNR > 15 dB ✅ (default threshold)
  - **Moderate**: SNR 10-15 dB
  - **Noisy**: SNR < 10 dB

### 5. **Bandwidth** (Frequency Range)
- **Scale**: Hertz (Hz, higher is better)
- **What it measures**: Effective audio frequency range
- **Categories**:
  - **Telephone**: < 3400 Hz (poor for TTS)
  - **Narrowband**: 4000-8000 Hz (acceptable for TTS) ✅
  - **Wideband**: 8000-16000 Hz (ideal for TTS)
  - **Super-wideband**: > 16000 Hz (excellent)
- **Default threshold**: 4000 Hz (narrowband minimum)

---

## 🎯 Default Quality Thresholds

These are the default thresholds set in the configurations:

```yaml
min_pesq: 3.0              # Perceptual quality
min_stoi: 0.75             # Intelligibility
min_sisdr: 12.0            # Signal-to-distortion (dB)
min_snr: 15.0              # Signal-to-noise (dB)
min_bandwidth: 4000        # Minimum bandwidth (Hz)
```

**Result**: Only segments meeting **ALL** quality thresholds are kept.

---

## 🔧 Customizing Thresholds

### Strict Filtering (High-Quality TTS Only)

For premium TTS models, use stricter thresholds:

```bash
python main.py \
  --config-name="config_cross_validate_multilang_asr.yaml" \
  min_pesq=4.0 \
  min_stoi=0.85 \
  min_sisdr=15.0 \
  min_snr=20.0 \
  min_bandwidth=8000 \
  ...
```

### Relaxed Filtering (More Data, Lower Quality)

For larger datasets with acceptable quality:

```bash
python main.py \
  --config-name="config_cross_validate_multilang_asr.yaml" \
  min_pesq=2.5 \
  min_stoi=0.65 \
  min_sisdr=10.0 \
  min_snr=12.0 \
  min_bandwidth=3000 \
  ...
```

### Disable Quality Filtering

If you want to skip quality filtering entirely:

```bash
# Remove or comment out quality filtering steps in YAML
# Or set very low thresholds:
python main.py \
  --config-name="config_cross_validate_multilang_asr.yaml" \
  min_pesq=1.0 \
  min_stoi=0.1 \
  min_sisdr=-10.0 \
  min_snr=0.0 \
  min_bandwidth=0 \
  ...
```

---

## 📈 Expected Data Retention

Quality filtering will reduce your dataset size. Here's what to expect:

| Threshold Level | Data Retained | Quality | Use Case |
|----------------|---------------|---------|----------|
| **Very Strict** | ~10-20% | Excellent | Premium TTS, studio quality |
| **Strict** | ~20-40% | Very Good | High-quality TTS |
| **Default** | ~40-60% | Good | Standard TTS training |
| **Relaxed** | ~60-80% | Acceptable | Large-scale TTS, data-hungry models |
| **Very Relaxed** | ~80-95% | Mixed | Research, low-resource languages |

---

## 🔍 Analyzing Quality Metrics

### Check Quality Distribution

After processing, analyze the quality metrics:

```bash
# Extract quality metrics
jq -r '[.pesq, .stoi, .sisdr, .snr_db, .bandwidth_hz] | @csv' \
  manifests/20_with_snr.json > quality_metrics.csv

# Summary statistics
jq -r '.pesq' manifests/20_with_snr.json | \
  awk '{sum+=$1; sumsq+=$1*$1} END {print "Mean:", sum/NR, "StdDev:", sqrt(sumsq/NR - (sum/NR)**2)}'
```

### View Samples by Quality

```bash
# Best quality samples
jq -r 'select(.pesq > 4.0 and .stoi > 0.9 and .sisdr > 18) | 
  {audio: .audio_filepath, pesq, stoi, sisdr}' \
  manifests/20_with_snr.json | head -20

# Worst quality samples (failed thresholds)
jq -r 'select(.pesq < 3.0 or .stoi < 0.75) | 
  {audio: .audio_filepath, pesq, stoi, sisdr}' \
  manifests/20_with_snr.json | head -20
```

---

## 🎛️ Quality vs. Quantity Trade-offs

### Scenario 1: Low-Resource Language
**Problem**: Limited audio data available  
**Solution**: Use relaxed thresholds to maximize data

```yaml
min_pesq: 2.5
min_stoi: 0.65
min_sisdr: 10.0
min_snr: 12.0
min_bandwidth: 3000
```

### Scenario 2: Production TTS
**Problem**: Need highest quality for commercial deployment  
**Solution**: Use strict thresholds

```yaml
min_pesq: 4.0
min_stoi: 0.9
min_sisdr: 18.0
min_snr: 22.0
min_bandwidth: 8000
```

### Scenario 3: Research/Experimentation
**Problem**: Want to study impact of quality on TTS  
**Solution**: Keep all data, filter later based on metrics

```yaml
# Keep all data initially
enable_quality_metrics: true
min_pesq: 1.0  # Don't filter, just tag
min_stoi: 0.0
min_sisdr: -50.0
min_snr: 0.0
min_bandwidth: 0
```

Then filter during training based on quality metrics in the manifest.

---

## 🚨 Common Issues

### Issue 1: Too Much Data Filtered Out

**Symptoms**: < 20% of data retained  
**Possible causes**:
- Noisy source audio
- Low-quality recordings
- Codec compression artifacts

**Solutions**:
1. Relax thresholds gradually
2. Check source audio quality
3. Investigate failed samples
4. Consider better audio sources

### Issue 2: Quality Metrics All Zero

**Symptoms**: All quality metrics show 0.0  
**Possible causes**:
- CUDA/GPU issues
- Missing dependencies
- Audio loading failures

**Solutions**:
```bash
# Install TTS requirements
pip install -r requirements/tts.txt

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Check audio loading
python -c "import librosa; librosa.load('test.wav')"
```

### Issue 3: Slow Quality Assessment

**Symptoms**: Quality metrics stage very slow  
**Solutions**:
- Use GPU for SQUIM (default in config)
- Increase batch size
- Process subset first to estimate time

---

## 📊 Quality Metrics by Language

Different languages may have different quality distributions. Monitor per-language:

```bash
# Quality by language
jq -r '[.detected_lang, .pesq, .stoi, .sisdr] | @csv' \
  manifests/20_with_snr.json | \
  sort | \
  awk -F',' '{sum[$1]+=$2; count[$1]++} 
   END {for (lang in sum) print lang, sum[lang]/count[lang]}'
```

---

## 💡 Best Practices

1. **Start Conservative**: Use default thresholds first, then adjust
2. **Monitor Distribution**: Check quality histograms before filtering
3. **Language-Specific**: Different languages may need different thresholds
4. **Iterative Approach**: Process small batch → analyze → adjust → full run
5. **Keep Metrics**: Always keep quality scores in final manifest for debugging
6. **GPU Utilization**: Use GPU for SQUIM (much faster than CPU)
7. **Documentation**: Document threshold decisions for reproducibility

---

## 📚 References

- **PESQ**: ITU-T Recommendation P.862
- **STOI**: Taal et al. (2011) - "An Algorithm for Intelligibility Prediction of Time-Frequency Weighted Noisy Speech"
- **SI-SDR**: Le Roux et al. (2019) - "SDR – Half-baked or Well Done?"
- **SQUIM**: Kumar et al. (2023) - TorchAudio implementation

---

## 🎯 Quick Decision Matrix

| Your Goal | Recommended Thresholds |
|-----------|------------------------|
| Premium commercial TTS | PESQ>4.0, STOI>0.9, SI-SDR>18, SNR>22, BW>8k |
| Standard TTS training | PESQ>3.0, STOI>0.75, SI-SDR>12, SNR>15, BW>4k ✅ |
| Large-scale TTS | PESQ>2.5, STOI>0.65, SI-SDR>10, SNR>12, BW>3k |
| Low-resource language | PESQ>2.0, STOI>0.6, SI-SDR>8, SNR>10, BW>2k |
| Research/baseline | Keep all, analyze later |

---

**Default configuration uses "Standard TTS training" thresholds** - good balance between quality and data quantity for most use cases. ✅

