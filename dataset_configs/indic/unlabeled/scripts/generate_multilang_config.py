#!/usr/bin/env python3
"""
Generate Multi-Language ASR Routing Configuration

This script generates a complete configuration file for processing
long audio with automatic ASR routing based on detected language.

Usage:
    python generate_multilang_config.py \
        --languages hi,te,ta,en \
        --output config_my_languages.yaml
"""

import argparse
from pathlib import Path


HEADER_TEMPLATE = '''documentation: |
  Cross-Validated Multi-Language ASR Router
  ##########################################
  
  AUTO-GENERATED configuration for {num_languages} languages: {language_list}
  
  This configuration automatically routes segments to appropriate ASR models
  based on cross-validated language detection.
  
  **Process:**
  1. VAD segments the long audio
  2. Cross-validate language detection (3 models) per segment
  3. Route each segment to its detected language's ASR model
  4. Combine all transcriptions
  
  **Example Usage:**
  
  .. code-block:: bash
  
      python main.py \\
        --config-path="dataset_configs/indic/unlabeled" \\
        --config-name="{config_name}" \\
        processors_to_run="0:" \\
        workspace_dir="/data/mixed_audio" \\
{model_params}

processors_to_run: "0:"

workspace_dir: ???

# ASR Model paths for each language
{model_definitions}

# Optional parameters
audio_extension: wav
min_segment_duration: 1.0
max_segment_duration: 30.0
asr_batch_size: 32

# VAD parameters
vad_model: "vad_multilingual_frame_marblenet"
vad_onset: 0.3
vad_offset: 0.3
vad_min_duration: 0.2

# Cross-validation parameters
consensus_method: "weighted"
require_agreement: 2
langid_min_confidence: 0.6
langid_segment_duration: 15
langid_num_segments: 1

# Derived paths
manifest_dir: ${{workspace_dir}}/manifests
raw_audio_dir: ${{workspace_dir}}/raw_audio
segmented_audio_dir: ${{workspace_dir}}/segmented_audio
vad_output_dir: ${{manifest_dir}}/vad_output
final_manifest: ${{manifest_dir}}/all_languages_final_manifest.json

processors:
'''

SEGMENTATION_STAGE = '''  # ============================================================
  # STAGE 1: SEGMENTATION
  # ============================================================
  
  - _target_: sdp.processors.CreateInitialManifestByExt
    raw_data_dir: ${raw_audio_dir}
    extension: ${audio_extension}
    output_file_key: audio_filepath
    output_manifest_file: ${manifest_dir}/00_initial_manifest.json

  - _target_: sdp.processors.GetAudioDuration
    audio_filepath_key: audio_filepath
    duration_key: duration
    output_manifest_file: ${manifest_dir}/01_with_duration.json

  - _target_: sdp.processors.Subprocess
    cmd: 'rm -rf ${vad_output_dir}/*'

  - _target_: sdp.processors.Subprocess
    input_manifest_file: ${manifest_dir}/01_with_duration.json
    output_manifest_file: ${vad_output_dir}
    input_manifest_arg: "manifest_filepath"
    output_manifest_arg: "output_dir"
    cmd: 'python sdp/processors/inference/asr/nemo/utils/speech_to_text_with_vad.py audio_type=${audio_extension} vad_model=${vad_model} vad_config=sdp/processors/inference/asr/nemo/utils/frame_vad_infer_postprocess.yaml'

  - _target_: sdp.processors.RenameFields
    input_manifest_file: ${vad_output_dir}/temp_manifest_vad_rttm-onset${vad_onset}-offset${vad_offset}-pad_onset0.2-pad_offset0.2-min_duration_on${vad_min_duration}-min_duration_off${vad_min_duration}-filter_speech_firstTrue.json
    output_manifest_file: ${manifest_dir}/02_vad_output.json
    rename_fields: {"audio_filepath": "source_filepath"}

  - _target_: sdp.processors.GetRttmSegments
    output_manifest_file: ${manifest_dir}/03_with_segments.json
    rttm_key: rttm_file
    output_file_key: audio_segments
    duration_key: duration
    duration_threshold: ${max_segment_duration}

  - _target_: sdp.processors.SplitAudioFile
    output_manifest_file: ${manifest_dir}/04_segmented_audio.json
    splited_audio_dir: ${segmented_audio_dir}
    segments_key: audio_segments
    duration_key: duration
    input_file_key: source_filepath
    output_file_key: audio_filepath

  - _target_: sdp.processors.DropHighLowDuration
    output_manifest_file: ${manifest_dir}/05_filtered_segments.json
    high_duration_threshold: ${max_segment_duration}
    low_duration_threshold: ${min_segment_duration}

'''

LANGID_STAGE = '''  # ============================================================
  # STAGE 2: CROSS-VALIDATED LANGUAGE IDENTIFICATION
  # ============================================================

  - _target_: sdp.processors.AudioLid
    output_manifest_file: ${manifest_dir}/06_with_nemo_langid.json
    input_audio_key: audio_filepath
    output_lang_key: nemo_lang
    device: cuda
    pretrained_model: "langid_ambernet"
    segment_duration: ${langid_segment_duration}
    num_segments: ${langid_num_segments}

  - _target_: sdp.processors.SpeechBrainLangId
    output_manifest_file: ${manifest_dir}/07_with_speechbrain_langid.json
    input_audio_key: audio_filepath
    output_lang_key: speechbrain_lang
    output_confidence_key: speechbrain_conf
    model_source: "speechbrain/lang-id-voxlingua107-ecapa"
    min_confidence: ${langid_min_confidence}
    device: cuda
    use_dask: False  # Disable Dask (custom batch processing avoids Dask overhead)
    max_workers: 1  # Serial processing (custom batch method handles batching)

  - _target_: sdp.processors.WhisperLangId
    output_manifest_file: ${manifest_dir}/08_with_whisper_langid.json
    input_audio_key: audio_filepath
    output_lang_key: whisper_lang
    output_confidence_key: whisper_conf
    model_size: "large-v3"
    min_confidence: ${langid_min_confidence}
    device: cuda

  - _target_: sdp.processors.CrossValidateLangId
    output_manifest_file: ${manifest_dir}/09_with_consensus_lang.json
    lang_keys: ["nemo_lang", "speechbrain_lang", "whisper_lang"]
    confidence_keys: ["nemo_conf", "speechbrain_conf", "whisper_conf"]
    output_lang_key: detected_lang
    output_confidence_key: lang_confidence
    method: ${consensus_method}
    require_agreement: ${require_agreement}

'''

ASR_ROUTING_TEMPLATE = '''  # Process {lang_name} segments
  - _target_: sdp.processors.PreserveByValue
    input_manifest_file: ${{manifest_dir}}/09_with_consensus_lang.json
    output_manifest_file: ${{manifest_dir}}/10_{lang_code}_segments.json
    input_value_key: detected_lang
    target_value: "{lang_code}"

  - _target_: sdp.processors.ASRInference
    input_manifest_file: ${{manifest_dir}}/10_{lang_code}_segments.json
    output_manifest_file: ${{manifest_dir}}/11_{lang_code}_transcribed.json
    pretrained_model: ${{model_{lang_code}}}
    batch_size: ${{asr_batch_size}}

'''

POSTPROCESSING_TEMPLATE = '''  # ============================================================
  # STAGE 4: COMBINE ALL LANGUAGES
  # ============================================================

  - _target_: sdp.processors.CreateCombinedManifests
    output_manifest_file: ${{manifest_dir}}/12_all_transcribed.json
    manifest_list:
{manifest_list}

  # ============================================================
  # STAGE 5: POST-PROCESSING
  # ============================================================

  - _target_: sdp.processors.DuplicateFields
    output_manifest_file: ${{manifest_dir}}/13_with_text.json
    duplicate_fields: {{"pred_text": "text"}}

  - _target_: sdp.processors.SubRegex
    output_manifest_file: ${{manifest_dir}}/14_cleaned_text.json
    regex_params_list:
      - {{"pattern": "\\\\s+", "repl": " "}}
      - {{"pattern": "^\\\\s+|\\\\s+$", "repl": ""}}

  - _target_: sdp.processors.DropIfRegexMatch
    output_manifest_file: ${{manifest_dir}}/15_filtered_empty.json
    regex_patterns: 
      - "^\\\\s*$"
      - "^.{{1,2}}$"

  - _target_: sdp.processors.CountNumWords
    output_manifest_file: ${{manifest_dir}}/16_with_word_count.json
    text_key: text
    output_key: num_words

  - _target_: sdp.processors.DropHighLowWordrate
    output_manifest_file: ${{manifest_dir}}/17_filtered_wordrate.json
    high_wordrate_threshold: 6.0
    low_wordrate_threshold: 0.3

  - _target_: sdp.processors.KeepOnlySpecifiedFields
    output_manifest_file: ${{final_manifest}}
    fields_to_keep:
      - "audio_filepath"
      - "text"
      - "duration"
      - "pred_text"
      - "detected_lang"
      - "lang_confidence"
      - "num_words"
      - "source_filepath"
      - "nemo_lang"
      - "speechbrain_lang"
      - "speechbrain_conf"
      - "whisper_lang"
      - "whisper_conf"
      - "lang_agreement"
      - "lang_total_models"
'''

LANGUAGE_NAMES = {
    'hi': 'Hindi',
    'te': 'Telugu',
    'ta': 'Tamil',
    'bn': 'Bengali',
    'ml': 'Malayalam',
    'kn': 'Kannada',
    'mr': 'Marathi',
    'gu': 'Gujarati',
    'pa': 'Punjabi',
    'or': 'Odia',
    'as': 'Assamese',
    'en': 'English',
}


def generate_config(languages, output_file):
    """Generate complete configuration for specified languages."""
    
    # Prepare language list
    language_list = ', '.join([f"{LANGUAGE_NAMES.get(lang, lang)} ({lang})" for lang in languages])
    
    # Generate model parameters for example
    model_params = '\n'.join([f'        model_{lang}="/models/{lang}_asr.nemo" \\' for lang in languages])
    model_params = model_params.rstrip(' \\')
    
    # Generate model definitions
    model_definitions = '\n'.join([
        f'model_{lang}: ???  # {LANGUAGE_NAMES.get(lang, lang)}'
        for lang in languages
    ])
    
    # Generate header
    config_name = Path(output_file).stem
    header = HEADER_TEMPLATE.format(
        num_languages=len(languages),
        language_list=language_list,
        config_name=config_name,
        model_params=model_params,
        model_definitions=model_definitions
    )
    
    # Generate ASR routing section
    asr_routing = '  # ============================================================\n'
    asr_routing += '  # STAGE 3: LANGUAGE-SPECIFIC ASR ROUTING\n'
    asr_routing += '  # ============================================================\n\n'
    
    for lang in languages:
        asr_routing += ASR_ROUTING_TEMPLATE.format(
            lang_name=LANGUAGE_NAMES.get(lang, lang),
            lang_code=lang
        )
    
    # Generate manifest list for combining
    manifest_list = '\n'.join([
        f'      - ${{manifest_dir}}/11_{lang}_transcribed.json'
        for lang in languages
    ])
    
    postprocessing = POSTPROCESSING_TEMPLATE.format(manifest_list=manifest_list)
    
    # Combine all sections
    full_config = header + SEGMENTATION_STAGE + LANGID_STAGE + asr_routing + postprocessing
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(full_config)
    
    print(f"✓ Generated configuration for {len(languages)} languages:")
    for lang in languages:
        print(f"  - {LANGUAGE_NAMES.get(lang, lang)} ({lang})")
    print(f"\n✓ Saved to: {output_file}")
    print(f"\n📖 Usage:")
    print(f"    python main.py \\")
    print(f"      --config-name=\"{config_name}.yaml\" \\")
    print(f"      workspace_dir=\"/data/audio\" \\")
    for lang in languages:
        print(f"      model_{lang}=\"/models/{lang}.nemo\" \\")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Generate multi-language ASR routing configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate config for 4 languages
  python generate_multilang_config.py \\
    --languages hi,te,ta,en \\
    --output config_4lang.yaml

  # Generate config for all 12 Indic languages
  python generate_multilang_config.py \\
    --languages hi,te,ta,bn,ml,kn,mr,gu,pa,or,as,en \\
    --output config_all_indic.yaml
        '''
    )
    
    parser.add_argument(
        '--languages',
        required=True,
        help='Comma-separated list of language codes (e.g., hi,te,ta,en)'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='Output configuration file name'
    )
    
    args = parser.parse_args()
    
    # Parse languages
    languages = [lang.strip() for lang in args.languages.split(',')]
    
    # Validate languages
    unknown = [lang for lang in languages if lang not in LANGUAGE_NAMES]
    if unknown:
        print(f"⚠️  Warning: Unknown language codes: {', '.join(unknown)}")
        print(f"   Known codes: {', '.join(sorted(LANGUAGE_NAMES.keys()))}")
    
    # Generate configuration
    generate_config(languages, args.output)


if __name__ == '__main__':
    main()

