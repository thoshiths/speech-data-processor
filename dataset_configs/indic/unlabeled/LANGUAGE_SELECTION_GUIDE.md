# Language Selection Guide

This guide explains how to selectively process specific languages instead of all 12 languages.

## 🎯 Quick Start

### Process Only 3 Languages (Hindi, Telugu, English)

```bash
python main.py \
  --config-name="config_cross_validate_multilang_asr.yaml" \
  workspace_dir="/data/audio" \
  target_languages='["hi","te","en"]' \
  model_hi="/models/hindi.nemo" \
  model_te="/models/telugu.nemo" \
  model_en="/models/english.nemo"
```

**Note**: You only need to provide models for languages in `target_languages`!

---

## 📋 Supported Languages

| Code | Language | Script |
|------|----------|--------|
| `hi` | Hindi | Devanagari |
| `te` | Telugu | Telugu |
| `ta` | Tamil | Tamil |
| `bn` | Bengali | Bengali |
| `ml` | Malayalam | Malayalam |
| `kn` | Kannada | Kannada |
| `mr` | Marathi | Devanagari |
| `gu` | Gujarati | Gujarati |
| `pa` | Punjabi | Gurmukhi |
| `or` | Odia | Odia |
| `as` | Assamese | Bengali |
| `en` | English | Latin |

---

## 🔧 Configuration Options

### Default: All 12 Languages

```yaml
target_languages: ["hi", "te", "ta", "bn", "ml", "kn", "mr", "gu", "pa", "or", "as", "en"]
```

### Custom: Select Specific Languages

```yaml
target_languages: ["hi", "en"]  # Only Hindi and English
```

```yaml
target_languages: ["hi", "te", "ta"]  # Top 3 Dravidian + Hindi
```

```yaml
target_languages: ["hi", "mr", "gu", "en"]  # Indo-Aryan + English
```

---

## 💡 Use Cases

### Use Case 1: Pilot Testing
**Goal**: Test pipeline with 2-3 languages before full deployment

```bash
python main.py \
  --config-name="config_cross_validate_multilang_asr.yaml" \
  workspace_dir="/data/pilot" \
  target_languages='["hi","en"]' \
  model_hi="/models/hindi.nemo" \
  model_en="/models/english.nemo"
```

### Use Case 2: Regional Focus
**Goal**: Process only South Indian languages

```bash
python main.py \
  --config-name="config_cross_validate_multilang_asr.yaml" \
  workspace_dir="/data/south_india" \
  target_languages='["te","ta","ml","kn"]' \
  model_te="/models/telugu.nemo" \
  model_ta="/models/tamil.nemo" \
  model_ml="/models/malayalam.nemo" \
  model_kn="/models/kannada.nemo"
```

### Use Case 3: High-Resource Languages Only
**Goal**: Process languages with best models

```bash
python main.py \
  --config-name="config_cross_validate_multilang_asr.yaml" \
  workspace_dir="/data/high_resource" \
  target_languages='["hi","en","te","bn"]' \
  model_hi="/models/hindi_premium.nemo" \
  model_en="/models/english_premium.nemo" \
  model_te="/models/telugu_premium.nemo" \
  model_bn="/models/bengali_premium.nemo"
```

### Use Case 4: Single Language
**Goal**: Extract and process only one language

```bash
python main.py \
  --config-name="config_cross_validate_multilang_asr.yaml" \
  workspace_dir="/data/hindi_only" \
  target_languages='["hi"]' \
  model_hi="/models/hindi.nemo"
```

---

## 🔄 How It Works

### 1. **Full Pipeline** (All Languages)
```
Long Audio
    ↓
VAD Segmentation → 1000 segments
    ↓
Cross-Validated LangID
    ↓ (Detects: 300 hi, 250 te, 200 en, 150 ta, 100 other)
Keep ALL Languages
    ↓
Route to 12 ASR models
    ↓
Final: All segments transcribed
```

### 2. **Selective Pipeline** (target_languages=['hi','te','en'])
```
Long Audio
    ↓
VAD Segmentation → 1000 segments
    ↓
Cross-Validated LangID
    ↓ (Detects: 300 hi, 250 te, 200 en, 150 ta, 100 other)
Filter: Keep ONLY hi, te, en ⭐
    ↓ (Kept: 750 segments, Dropped: 250 segments)
Route to 3 ASR models (hi, te, en)
    ↓
Final: 750 segments transcribed in 3 languages
```

**Key Benefit**: You don't need to provide models for languages you're not processing!

---

## 📊 Expected Results

### Example: Audio with Mixed Languages

| Language | Segments Detected | Kept (All) | Kept (hi,te,en only) |
|----------|-------------------|------------|----------------------|
| Hindi | 350 | ✅ 350 | ✅ 350 |
| Telugu | 250 | ✅ 250 | ✅ 250 |
| English | 200 | ✅ 200 | ✅ 200 |
| Tamil | 100 | ✅ 100 | ❌ 0 |
| Kannada | 50 | ✅ 50 | ❌ 0 |
| Other | 50 | ✅ 50 | ❌ 0 |
| **Total** | **1000** | **1000** | **800** |

---

## ⚙️ Advanced Configuration

### Dynamic Language List from File

Create a file `languages.txt`:
```
hi
te
en
```

Use in script:
```bash
LANGS=$(cat languages.txt | paste -sd "," - | sed 's/,/","/g')
LANGS='["'$LANGS'"]'

python main.py \
  --config-name="config_cross_validate_multilang_asr.yaml" \
  workspace_dir="/data/audio" \
  target_languages="$LANGS" \
  model_hi="/models/hindi.nemo" \
  model_te="/models/telugu.nemo" \
  model_en="/models/english.nemo"
```

### Process Different Language Sets in Parallel

```bash
# Terminal 1: Indo-Aryan languages
python main.py \
  --config-name="config_8gpu.yaml" \
  workspace_dir="/data/indo_aryan" \
  target_languages='["hi","mr","gu","pa","as","bn"]' \
  ...

# Terminal 2: Dravidian languages
python main.py \
  --config-name="config_8gpu.yaml" \
  workspace_dir="/data/dravidian" \
  target_languages='["te","ta","ml","kn"]' \
  ...

# Terminal 3: English + Odia
python main.py \
  --config-name="config_8gpu.yaml" \
  workspace_dir="/data/other" \
  target_languages='["en","or"]' \
  ...
```

---

## 🔍 Analyzing Language Distribution

### Before Filtering (After LangID)

Check what languages were detected:

```bash
jq -r '.detected_lang' manifests/09_with_consensus_lang.json | \
  sort | uniq -c | sort -rn

# Output:
#  350 hi
#  250 te
#  200 en
#  100 ta
#   50 kn
#   50 other
```

### After Filtering

Check what was kept:

```bash
jq -r '.detected_lang' manifests/09a_target_languages_only.json | \
  sort | uniq -c | sort -rn

# Output (with target_languages=["hi","te","en"]):
#  350 hi
#  250 te
#  200 en
```

---

## 🚨 Common Issues

### Issue 1: Missing Model for Target Language

**Error**: `model_ta` not provided but `ta` in `target_languages`

**Solution**: Either provide the model or remove language from list:

```bash
# Option 1: Provide the model
model_ta="/models/tamil.nemo"

# Option 2: Remove from target list
target_languages='["hi","te","en"]'  # Removed "ta"
```

### Issue 2: No Segments After Filtering

**Symptom**: `09a_target_languages_only.json` is empty

**Possible causes**:
- No segments detected in target languages
- LangID models don't recognize your languages
- Audio quality too poor for detection

**Solution**:
```bash
# Check detected languages
jq -r '.detected_lang' manifests/09_with_consensus_lang.json | sort | uniq -c

# If your language is detected with different code, use that code
# E.g., if "hin" instead of "hi", adjust target_languages
```

### Issue 3: Model Paths Not Found

**Error**: Model file not found

**Solution**: Use absolute paths:
```bash
model_hi="/absolute/path/to/models/hindi.nemo"
# Not: model_hi="models/hindi.nemo"
```

---

## 💰 Cost Optimization

Processing fewer languages = Lower costs!

| Languages | GPU Hours (100h audio) | Cost Savings |
|-----------|------------------------|--------------|
| All 12 | ~15 GPU-hours | Baseline |
| 6 languages | ~8 GPU-hours | 47% savings |
| 3 languages | ~5 GPU-hours | 67% savings |
| 1 language | ~3 GPU-hours | 80% savings |

**Recommendation**: Start with 2-3 languages for pilot, then expand based on results.

---

## 📚 Examples by Region

### North India
```yaml
target_languages: ["hi", "pa", "mr", "gu", "en"]
```

### South India  
```yaml
target_languages: ["te", "ta", "kn", "ml", "en"]
```

### East India
```yaml
target_languages: ["bn", "as", "or", "hi", "en"]
```

### West India
```yaml
target_languages: ["mr", "gu", "hi", "en"]
```

### Major Languages Only
```yaml
target_languages: ["hi", "te", "bn", "mr", "ta", "en"]
```

---

## 🎯 Best Practices

1. **Start Small**: Test with 2-3 languages first
2. **Check Distribution**: Analyze LangID results before filtering
3. **Absolute Paths**: Use absolute paths for model files
4. **Document Choices**: Keep track of which languages you processed
5. **Validate Models**: Ensure models work before large-scale processing
6. **Monitor Quality**: Check quality metrics per language
7. **Batch by Family**: Process related languages together (e.g., all Dravidian)

---

## 🔧 Quick Reference

### Syntax
```bash
target_languages='["lang1","lang2","lang3"]'
```

### All Languages
```bash
target_languages='["hi","te","ta","bn","ml","kn","mr","gu","pa","or","as","en"]'
```

### Single Language
```bash
target_languages='["hi"]'
```

### Check Current Setting
```bash
grep "^target_languages:" config_cross_validate_multilang_asr.yaml
```

---

**Pro Tip**: You can always re-run the pipeline with a different `target_languages` list to extract additional languages from the same LangID results! The language detection step doesn't need to be re-run. 🚀

