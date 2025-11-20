# Punctuation Restoration Guide

This guide explains how to add punctuation and capitalization to ASR transcriptions using LLM models or traditional approaches.

## 🎯 Overview

ASR models typically produce lowercase text without punctuation:
```
Input: hello how are you doing today
```

This processor adds punctuation and capitalization:
```
Output: Hello, how are you doing today?
```

The pipeline creates **duplicate entries**:
- **Original** with `"pnc": "no"` (no punctuation)
- **Punctuated** with `"pnc": "yes"` (with punctuation)

This gives you both versions for different use cases!

---

## 🚀 Quick Start

### Default Configuration (LLM-based)

The configurations are already set up with Qwen2.5-7B-Instruct:

```bash
python main.py \
  --config-name="config_cross_validate_multilang_asr.yaml" \
  workspace_dir="/data/audio" \
  target_languages='["hi","te","en"]' \
  punctuation_model="Qwen/Qwen2.5-7B-Instruct" \
  ...
```

**Result**: Each entry is duplicated:
```json
{"text": "hello how are you", "pnc": "no", ...}
{"text": "Hello, how are you?", "pnc": "yes", ...}
```

---

## 🔧 Punctuation Methods

### Method 1: LLM-based (Recommended for Multilingual)

**Pros**:
- ✅ Supports multiple languages (Indic, English, etc.)
- ✅ Context-aware punctuation
- ✅ Handles complex sentences
- ✅ Better capitalization

**Cons**:
- ❌ Slower than NeMo
- ❌ Requires GPU memory for LLM

```yaml
# In config
punctuation_method: "llm"
punctuation_model: "Qwen/Qwen2.5-7B-Instruct"
```

### Method 2: NeMo Model (Fast for English)

**Pros**:
- ✅ Very fast
- ✅ Lower memory usage
- ✅ Production-ready

**Cons**:
- ❌ English only (mostly)
- ❌ Less context-aware

```yaml
# In config
punctuation_method: "nemo"
punctuation_model: "punctuation_en_bert"
```

### Method 3: Disable Punctuation

```yaml
# In config
enable_punctuation: false
```

Or remove the punctuation processor from the pipeline.

---

## 🤖 Supported LLM Models

### Multilingual Models (Recommended for Indic)

| Model | Languages | Size | Speed | Quality |
|-------|-----------|------|-------|---------|
| **Qwen/Qwen2.5-7B-Instruct** ✅ | 100+ incl. Indic | 7B | Fast | Excellent |
| **Qwen/Qwen2.5-14B-Instruct** | 100+ incl. Indic | 14B | Medium | Excellent |
| **google/gemma-2-9b-it** | 50+ incl. some Indic | 9B | Fast | Very Good |
| **ai4bharat/Airavata** | Indic + English | 7B | Fast | Excellent (Indic) |
| **meta-llama/Llama-3.1-8B-Instruct** | 50+ | 8B | Fast | Very Good |

### English-only Models

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| **microsoft/Phi-3-mini-4k-instruct** | 3.8B | Very Fast | Good |
| **mistralai/Mistral-7B-Instruct-v0.3** | 7B | Fast | Excellent |

---

## 💡 Custom Prompt Templates

### Default Prompt (Works for most cases)
```python
punctuation_prompt: "Add proper punctuation and capitalization to this text. Return only the punctuated text:\n\n{text}"
```

### Language-Specific Prompts

#### For Hindi
```yaml
punctuation_prompt: "इस टेक्स्ट में विराम चिन्ह और कैपिटलाइज़ेशन जोड़ें। केवल विराम चिन्हों वाला टेक्स्ट लौटाएँ:\n\n{text}"
```

#### For Mixed Language
```yaml
punctuation_prompt: "Add punctuation (periods, commas, question marks) and proper capitalization to this text. Preserve the original language. Return ONLY the punctuated text:\n\n{text}"
```

#### For Formal Style
```yaml
punctuation_prompt: "Add formal punctuation and capitalization to this text for professional documentation:\n\n{text}"
```

#### For Conversational Style
```yaml
punctuation_prompt: "Add natural conversational punctuation to this spoken text:\n\n{text}"
```

---

## 📊 Output Format

### Before Punctuation (Single Entry)
```json
{
  "audio_filepath": "segment_001.wav",
  "text": "hello how are you doing today",
  "duration": 2.5,
  "detected_lang": "en"
}
```

### After Punctuation (Duplicated)
```json
{
  "audio_filepath": "segment_001.wav",
  "text": "hello how are you doing today",
  "duration": 2.5,
  "detected_lang": "en",
  "pnc": "no"
}
{
  "audio_filepath": "segment_001.wav",
  "text": "Hello, how are you doing today?",
  "duration": 2.5,
  "detected_lang": "en",
  "pnc": "yes"
}
```

**Dataset Size**: 2x original (1000 segments → 2000 entries)

---

## 🔧 Configuration Examples

### Example 1: Use Qwen for Indic Languages
```bash
python main.py \
  --config-name="config_cross_validate_multilang_asr.yaml" \
  workspace_dir="/data/audio" \
  punctuation_model="Qwen/Qwen2.5-7B-Instruct" \
  punctuation_prompt="Add punctuation and capitalization. Return only the punctuated text:\n\n{text}" \
  ...
```

### Example 2: Use Airavata for Indic (Specialized)
```bash
python main.py \
  --config-name="config_cross_validate_multilang_asr.yaml" \
  workspace_dir="/data/audio" \
  punctuation_model="ai4bharat/Airavata" \
  punctuation_prompt="Add proper punctuation marks to this text:\n\n{text}" \
  ...
```

### Example 3: Use NeMo for English Only
```yaml
# Edit config file, replace LLMPunctuationRestoration with:
- _target_: sdp.processors.NeMoPunctuationRestoration
  output_manifest_file: ${manifest_dir}/26_with_punctuation.json
  input_text_key: text
  output_text_key: text
  model_name: "punctuation_en_bert"
  batch_size: 64
  keep_original: true
```

### Example 4: Disable Punctuation
```yaml
# Comment out the punctuation processor in config
# Or skip that step:
processors_to_run: "0:26,28:"  # Skip step 27 (punctuation)
```

---

## ⚡ Performance Comparison

### Processing 1000 Segments

| Method | Time | GPU Memory | Quality (Indic) | Quality (English) |
|--------|------|------------|-----------------|-------------------|
| **Qwen 7B** | ~5 min | 16GB | Excellent | Excellent |
| **Gemma 9B** | ~6 min | 18GB | Very Good | Excellent |
| **Airavata 7B** | ~5 min | 16GB | Excellent | Good |
| **NeMo BERT** | ~30 sec | 2GB | N/A | Very Good |
| **No Punctuation** | 0 sec | 0GB | N/A | N/A |

---

## 📈 Use Cases

### Use Case 1: TTS Training Data
**Need**: Clean, punctuated text for TTS models  
**Solution**: Use LLM with high quality
```yaml
punctuation_model: "Qwen/Qwen2.5-14B-Instruct"  # Best quality
keep_original: true  # Keep both versions
```

### Use Case 2: Subtitles/Captions
**Need**: Fast processing with reasonable quality  
**Solution**: Use NeMo for English or smaller LLM
```yaml
punctuation_model: "microsoft/Phi-3-mini-4k-instruct"  # Fast
```

### Use Case 3: Research/Analysis
**Need**: Multiple punctuation versions for comparison  
**Solution**: Run multiple models and compare
```bash
# Run once with Qwen
python main.py ... punctuation_model="Qwen/Qwen2.5-7B-Instruct"

# Run again with Airavata
python main.py ... punctuation_model="ai4bharat/Airavata"

# Compare results
```

### Use Case 4: Low-Resource Deployment
**Need**: Minimal GPU memory  
**Solution**: Use NeMo or disable
```yaml
punctuation_method: "nemo"  # Only 2GB VRAM
# Or disable entirely
enable_punctuation: false
```

---

## 🎯 Filtering by Punctuation Flag

### Keep Only Punctuated Versions
```bash
jq 'select(.pnc == "yes")' final_manifest.json > punctuated_only.json
```

### Keep Only Original Versions
```bash
jq 'select(.pnc == "no")' final_manifest.json > no_punctuation.json
```

### Split by Language and Punctuation
```bash
# Hindi with punctuation
jq 'select(.detected_lang == "hi" and .pnc == "yes")' final_manifest.json > hindi_punctuated.json

# English without punctuation
jq 'select(.detected_lang == "en" and .pnc == "no")' final_manifest.json > english_no_pnc.json
```

---

## 🐛 Troubleshooting

### Issue 1: Out of Memory

**Symptom**: CUDA out of memory error

**Solutions**:
```yaml
# Option 1: Use smaller model
punctuation_model: "microsoft/Phi-3-mini-4k-instruct"

# Option 2: Use NeMo instead
# Comment out LLMPunctuationRestoration, use NeMoPunctuationRestoration

# Option 3: Disable punctuation
enable_punctuation: false
```

### Issue 2: Slow Processing

**Symptom**: Punctuation stage takes too long

**Solutions**:
1. Use smaller/faster model (Phi-3, Qwen 7B instead of 14B)
2. Increase `tensor_parallel_size` if you have multiple GPUs
3. Use NeMo for English-only

### Issue 3: Poor Punctuation Quality

**Symptom**: Incorrect or missing punctuation

**Solutions**:
1. **Improve prompt**: Be more specific
```yaml
punctuation_prompt: "Add punctuation marks (., ?, !, ,) and capitalize the first letter of sentences. Do not change the text otherwise. Input:\n\n{text}\n\nOutput:"
```

2. **Try different model**: Qwen or Airavata for Indic
3. **Add examples** to prompt (few-shot):
```yaml
punctuation_prompt: |
  Add punctuation to text.
  
  Example 1:
  Input: hello how are you
  Output: Hello, how are you?
  
  Example 2:
  Input: what is your name
  Output: What is your name?
  
  Now punctuate:
  Input: {text}
  Output:
```

### Issue 4: Wrong Language in Output

**Symptom**: LLM translates instead of just adding punctuation

**Solution**: Emphasize preservation in prompt
```yaml
punctuation_prompt: "Add ONLY punctuation and capitalization. DO NOT translate or change any words. Keep the original language. Text:\n\n{text}"
```

---

## 📚 Best Practices

1. **✅ Test First**: Run on small sample before full dataset
2. **✅ Monitor Quality**: Check first 10-20 punctuated samples manually
3. **✅ Keep Both Versions**: Always set `keep_original: true`
4. **✅ Language-Specific**: Use Airavata/Qwen for Indic, NeMo for English
5. **✅ Prompt Engineering**: Customize prompt for your use case
6. **✅ Resource Planning**: Reserve GPU for punctuation if using 8-GPU config

---

## 🔗 Model Resources

- **Qwen**: https://huggingface.co/Qwen
- **Airavata**: https://huggingface.co/ai4bharat/Airavata
- **Gemma**: https://huggingface.co/google/gemma-2-9b-it
- **NeMo Models**: https://catalog.ngc.nvidia.com/models?filters=&orderBy=weightPopularDESC&query=punctuation

---

## 💡 Pro Tips

1. **Batch Processing**: Punctuation runs on all segments at once (efficient)
2. **GPU 7**: In 8-GPU config, GPU 7 is reserved for punctuation
3. **Quality Check**: `pnc: "yes"` entries are ready for TTS training
4. **Flexibility**: You can always re-run just the punctuation step later
5. **Cost**: Keep originals (`pnc: "no"`) for free re-processing with different models

---

**Default: Qwen2.5-7B-Instruct provides excellent multilingual punctuation for Indic languages!** ✅

