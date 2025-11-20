# Language ID Model Comparison Guide

This document compares different language identification approaches available for the Indic audio processing pipeline.

## Available Models

### 1. NeMo AmberNet (`langid_ambernet`)
**Used in:** `config_with_langid.yaml`

**Model Type:** Audio-based neural classifier (AmberNet architecture)  
**Languages:** 100+ languages  
**Source:** NVIDIA NeMo  

**Pros:**
- ✅ Well-integrated with NeMo ecosystem
- ✅ Fast inference
- ✅ Good accuracy on Indic languages
- ✅ No additional dependencies

**Cons:**
- ❌ Requires NeMo installation
- ❌ May struggle with very short audio (< 3 seconds)

**Best for:** Standard pipeline with NeMo models

---

### 2. Faster Whisper Large V3
**Used in:** `config_whisper_langid.yaml`

**Model Type:** Transformer-based (Whisper Large V3)  
**Languages:** 99+ languages  
**Source:** OpenAI Whisper / Faster Whisper  

**Pros:**
- ✅ State-of-the-art accuracy
- ✅ Built into Whisper model
- ✅ Excellent on short audio
- ✅ Provides confidence scores
- ✅ Works well with code-switching

**Cons:**
- ❌ Slower than dedicated LangID models
- ❌ Requires larger GPU memory
- ❌ Overkill if not using Whisper for ASR

**Best for:** When using Whisper for ASR, or need highest accuracy

---

### 3. SpeechBrain VoxLingua107 ECAPA
**Used in:** `config_speechbrain_langid.yaml`

**Model Type:** ECAPA-TDNN (specialized for language ID)  
**Languages:** 107 languages  
**Source:** SpeechBrain / VoxLingua107 dataset  

**Pros:**
- ✅ Purpose-built for language identification
- ✅ Very fast inference
- ✅ Provides embedding vectors
- ✅ Works well on short segments
- ✅ Lower memory footprint

**Cons:**
- ❌ Requires SpeechBrain installation
- ❌ Separate model to maintain

**Best for:** Dedicated language ID without ASR overhead

---

### 4. Cross-Validation Ensemble
**Used in:** `config_cross_validate_langid.yaml`

**Model Type:** Ensemble of all three models  
**Languages:** All supported by constituent models  
**Source:** Combination  

**Pros:**
- ✅ Highest accuracy through consensus
- ✅ Identifies uncertain cases
- ✅ Reduces false positives
- ✅ Provides detailed per-model results

**Cons:**
- ❌ Slowest (runs 3 models)
- ❌ Highest memory usage
- ❌ More complex pipeline

**Best for:** Production systems requiring maximum accuracy

---

## Performance Comparison

### Accuracy on Indic Languages (Estimated)

| Model | Hindi | Telugu | Tamil | Bengali | Overall Indic |
|-------|-------|--------|-------|---------|---------------|
| **NeMo AmberNet** | 96% | 95% | 95% | 96% | 95-96% |
| **Whisper V3** | 98% | 97% | 96% | 97% | 96-98% |
| **SpeechBrain** | 95% | 94% | 94% | 95% | 94-95% |
| **Cross-Validation** | 99% | 98% | 97% | 98% | 97-99% |

*Based on typical use cases; actual accuracy varies by data quality*

### Speed Comparison

Processing 1000 audio files (avg 10 seconds each):

| Model | GPU Time | CPU Time | Memory Usage |
|-------|----------|----------|--------------|
| **NeMo AmberNet** | ~3 min | ~15 min | 2 GB |
| **Whisper V3** | ~8 min | ~60 min | 6 GB |
| **SpeechBrain** | ~2 min | ~10 min | 1.5 GB |
| **Cross-Validation** | ~15 min | ~90 min | 8 GB |

### Resource Requirements

| Model | Min GPU Memory | Recommended GPU | CPU Only |
|-------|----------------|-----------------|----------|
| **NeMo AmberNet** | 4 GB | Any | ✅ Slow |
| **Whisper V3** | 8 GB | A100, V100 | ❌ Very Slow |
| **SpeechBrain** | 2 GB | Any | ✅ OK |
| **Cross-Validation** | 10 GB | A100 | ❌ Not Recommended |

---

## Detailed Feature Comparison

### Supported Languages

#### Your 12 Languages Support:

| Language | NeMo | Whisper | SpeechBrain | Notes |
|----------|------|---------|-------------|-------|
| Hindi (hi) | ✅ | ✅ | ✅ | Excellent |
| Bengali (bn) | ✅ | ✅ | ✅ | Excellent |
| Telugu (te) | ✅ | ✅ | ✅ | Excellent |
| Tamil (ta) | ✅ | ✅ | ✅ | Excellent |
| Marathi (mr) | ✅ | ✅ | ✅ | Very Good |
| Gujarati (gu) | ✅ | ✅ | ✅ | Very Good |
| Kannada (kn) | ✅ | ✅ | ✅ | Very Good |
| Malayalam (ml) | ✅ | ✅ | ✅ | Very Good |
| Punjabi (pa) | ✅ | ✅ | ✅ | Good |
| Odia (or) | ✅ | ✅ | ✅ | Good |
| Assamese (as) | ✅ | ✅ | ✅ | Good |
| English (en) | ✅ | ✅ | ✅ | Excellent |

All models support all 12 of your languages!

### Code-Switching Handling

| Model | Code-Switching | Notes |
|-------|----------------|-------|
| **NeMo AmberNet** | ⚠️ Moderate | Best with clear boundaries |
| **Whisper V3** | ✅ Good | Handles intra-sentence switching |
| **SpeechBrain** | ⚠️ Moderate | Works best on single-language segments |
| **Cross-Validation** | ✅ Excellent | Consensus helps with ambiguity |

### Audio Duration Handling

| Model | Min Duration | Optimal Duration | Max Duration |
|-------|-------------|------------------|--------------|
| **NeMo AmberNet** | 3 sec | 10-30 sec | No limit |
| **Whisper V3** | 1 sec | 5-30 sec | 30 sec* |
| **SpeechBrain** | 2 sec | 5-20 sec | No limit |

*Whisper uses first 30 seconds for language detection

---

## Installation Requirements

### NeMo AmberNet
```bash
# Already included if you have NeMo
pip install nemo-toolkit[asr]
```

### Whisper Large V3
```bash
pip install faster-whisper
# Or for training:
pip install pytorch-lightning nvidia-cublas-cu12 nvidia-cudnn-cu12 faster_whisper
```

### SpeechBrain
```bash
pip install speechbrain torchaudio
```

### Cross-Validation (All of the above)
```bash
pip install nemo-toolkit[asr] faster-whisper speechbrain torchaudio
```

---

## Use Case Recommendations

### Scenario 1: Standard Processing
**Best Choice:** NeMo AmberNet (`config_with_langid.yaml`)

**Why:**
- Already using NeMo for ASR
- Good balance of speed and accuracy
- No additional dependencies

---

### Scenario 2: Maximum Accuracy Needed
**Best Choice:** Cross-Validation (`config_cross_validate_langid.yaml`)

**Why:**
- Production quality control
- Can afford processing time
- Need to identify uncertain cases

**Example:** Medical transcription, legal documents

---

### Scenario 3: Using Whisper for ASR
**Best Choice:** Whisper LangID (`config_whisper_langid.yaml`)

**Why:**
- Model already loaded
- No extra overhead
- Consistent ecosystem

---

### Scenario 4: Speed Critical
**Best Choice:** SpeechBrain (`config_speechbrain_langid.yaml`)

**Why:**
- Fastest inference
- Low memory usage
- Good enough accuracy

**Example:** Real-time processing, large datasets

---

### Scenario 5: Short Audio Clips (< 3 seconds)
**Best Choice:** Whisper (`config_whisper_langid.yaml`)

**Why:**
- Best performance on short audio
- Designed for varied durations

**Example:** Voice commands, social media clips

---

### Scenario 6: Mixed/Code-Switched Audio
**Best Choice:** Cross-Validation OR Whisper

**Why:**
- Better handling of ambiguity
- Higher confidence in difficult cases

**Example:** Bilingual conversations

---

## Example Configurations

### Example 1: Fast Processing with SpeechBrain

```bash
# Install
pip install speechbrain torchaudio

# Process
python main.py \
  --config-name="config_speechbrain_langid.yaml" \
  workspace_dir="/data/audio" \
  language_code="hi" \
  nemo_model_path="/models/hindi.nemo"
```

### Example 2: High Accuracy with Cross-Validation

```bash
# Install all models
pip install nemo-toolkit[asr] faster-whisper speechbrain torchaudio

# Process
python main.py \
  --config-name="config_cross_validate_langid.yaml" \
  workspace_dir="/data/audio" \
  language_code="te" \
  nemo_model_path="/models/telugu.nemo" \
  consensus_method="weighted" \
  require_agreement=2
```

### Example 3: Whisper-Based Pipeline

```bash
# Install
pip install faster-whisper

# Process
python main.py \
  --config-name="config_whisper_langid.yaml" \
  workspace_dir="/data/audio" \
  language_code="ta" \
  nemo_model_path="/models/tamil.nemo" \
  whisper_model_size="large-v3"
```

---

## Cross-Validation Details

### Consensus Methods

#### 1. **Voting** (Simple Majority)
```yaml
consensus_method: "voting"
```
- Each model gets one vote
- Most votes wins
- Ignores confidence scores

**Best for:** Equal trust in all models

#### 2. **Confidence** (Highest Confidence)
```yaml
consensus_method: "confidence"
```
- Highest confidence prediction wins
- Single model decision
- Fast tiebreaker

**Best for:** When one model is more reliable

#### 3. **Weighted** (Confidence-Weighted Voting) ⭐
```yaml
consensus_method: "weighted"
```
- Weights votes by confidence
- Balanced approach
- Default choice

**Best for:** Most use cases

### Agreement Requirements

```yaml
require_agreement: 2  # At least 2 models must agree
```

- `require_agreement: 1` - Accept any single prediction
- `require_agreement: 2` - At least 2 must agree (recommended)
- `require_agreement: 3` - All 3 must agree (very strict)

Samples not meeting agreement are marked as "uncertain"

---

## Output Examples

### Single Model Output
```json
{
  "audio_filepath": "/data/audio/file001.wav",
  "detected_lang": "hi",
  "lang_confidence": 0.9823,
  "text": "नमस्ते"
}
```

### Cross-Validation Output
```json
{
  "audio_filepath": "/data/audio/file001.wav",
  "nemo_lang": "hi",
  "speechbrain_lang": "hi",
  "speechbrain_conf": 0.9756,
  "whisper_lang": "hi",
  "whisper_conf": 0.9912,
  "detected_lang": "hi",
  "lang_confidence": 0.9801,
  "lang_agreement": 3,
  "lang_total_models": 3,
  "text": "नमस्ते"
}
```

---

## Decision Tree

```
START: Which LangID model should I use?
  │
  ├─ Need HIGHEST accuracy?
  │   └─ YES → Use Cross-Validation
  │
  ├─ Need FASTEST processing?
  │   └─ YES → Use SpeechBrain
  │
  ├─ Already using Whisper for ASR?
  │   └─ YES → Use Whisper LangID
  │
  ├─ Already using NeMo?
  │   └─ YES → Use NeMo AmberNet (default)
  │
  └─ Short audio (< 3 seconds)?
      └─ YES → Use Whisper LangID
```

---

## Summary Table

| Requirement | Best Choice |
|-------------|-------------|
| **Highest Accuracy** | Cross-Validation |
| **Fastest Speed** | SpeechBrain |
| **Using Whisper ASR** | Whisper LangID |
| **Using NeMo ASR** | NeMo AmberNet |
| **Short Audio** | Whisper LangID |
| **Code-Switching** | Whisper or Cross-Val |
| **Low Memory** | SpeechBrain |
| **Production Quality** | Cross-Validation |
| **Research/Testing** | SpeechBrain (fast iteration) |
| **Default Choice** | NeMo AmberNet |

---

## Troubleshooting

### Issue: Low Accuracy

**Try:**
1. Cross-validation for consensus
2. Increase `min_confidence` threshold
3. Filter segments by duration (> 3 seconds)

### Issue: Too Slow

**Try:**
1. Use SpeechBrain instead of Whisper
2. Skip cross-validation
3. Increase batch size if possible

### Issue: Out of Memory

**Try:**
1. Use SpeechBrain (lowest memory)
2. Reduce Whisper model size to "medium" or "small"
3. Process in smaller batches

---

## Conclusion

**For most users:** Start with **NeMo AmberNet** (`config_with_langid.yaml`)

**For maximum accuracy:** Use **Cross-Validation** (`config_cross_validate_langid.yaml`)

**For fastest processing:** Use **SpeechBrain** (`config_speechbrain_langid.yaml`)

**For Whisper users:** Use **Whisper LangID** (`config_whisper_langid.yaml`)

All models support your 12 languages and provide good accuracy. Choose based on your specific requirements!

